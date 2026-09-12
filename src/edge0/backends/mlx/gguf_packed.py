"""Direct GGML block execution in Metal (float32 activations/accumulation).

Decoding formulas and IQ tables derived from GGML commit
6c84c7d5d8833c6e0df69628f75a0f599797934e, copyright the ggml authors.
MIT license: see LICENSE.ggml. No projection is decoded into a weight array.
"""
from functools import lru_cache
import math

from edge0.checkpoints.gguf import QUANT_SIZES


def _header(kind):
    # These are fixed codebooks, never checkpoint data or decoded weights.
    from edge0.checkpoints.quant import IQ1_S, IQ2_XXS, IQ4_NL
    tables = ''
    if kind in (16, 19):
        cls = IQ2_XXS if kind == 16 else IQ1_S
        cls.init_grid()
        tables = 'constant short iqgrid[] = {' + ','.join(str(int(v)) for v in cls.grid.flat) + '};\n'
    if kind == 16:
        tables += 'constant uchar signs[] = {' + ','.join(map(str, IQ2_XXS.ksigns)) + '};\n'
    if kind == 20:
        tables += 'constant short values[] = {' + ','.join(map(str, IQ4_NL.kvalues)) + '};\n'
    bodies = {
        0: 'return as_type<float>(u32(p + 4*i));',
        30: 'return as_type<float>(u16(p + 2*i) << 16);',
        8: 'return h(p) * float(as_type<char>(p[2+i]));',
        12: '''uint g=i/32, j=i%32;
            uint sc=g<4 ? p[4+g]&63 : (p[8+g]&15)|((p[g]>>6)<<4);
            uint mn=g<4 ? p[8+g]&63 : (p[8+g]>>4)|((p[4+g]>>6)<<4);
            uint q=(p[16+(g/2)*32+j]>>((g%2)*4))&15;
            return (h(p)*float(sc))*float(q)-h(p+2)*float(mn);''',
        13: '''uint g=i/32, j=i%32;
            uint sc=g<4 ? p[4+g]&63 : (p[8+g]&15)|((p[g]>>6)<<4);
            uint mn=g<4 ? p[8+g]&63 : (p[8+g]>>4)|((p[4+g]>>6)<<4);
            uint q=((p[48+(g/2)*32+j]>>((g%2)*4))&15)|(((p[16+j]>>g)&1)<<4);
            return (h(p)*float(sc))*float(q)-h(p+2)*float(mn);''',
        14: '''uint segment=i/128, j=i%128;
            uint lo=(p[segment*64+j%64]>>((j/64)*4))&15;
            uint hi=(p[128+segment*32+j%32]>>((j/32)*2))&3;
            return (h(p+208)*float(as_type<char>(p[192+i/16])))*float(int(lo|(hi<<4))-32);''',
        16: '''uint g=i/32, s=(i%32)/8, j=i%8;
            uint a=u32(p+2+g*8+4);
            float d=h(p)*(0.5f+float(a>>28))*0.25f;
            uint sign=(signs[(a>>(7*s))&127]>>j)&1;
            return d*float(iqgrid[uint(p[2+g*8+s])*8+j])*(sign ? -1.0f : 1.0f);''',
        19: '''uint g=i/32, s=(i%32)/8, j=i%8, a=u16(p+34+2*g);
            uint idx=uint(p[2+g*4+s])|(((a>>(3*s))&7)<<8);
            float d=h(p)*float(2*((a>>12)&7)+1);
            return d*(float(iqgrid[idx*8+j])+((a&32768) ? -0.125f : 0.125f));''',
        20: 'return h(p)*float(values[(p[2+i%16]>>((i/16)*4))&15]);',
    }
    return tables + '''
inline uint u16(const device uchar* p) { return uint(p[0])|(uint(p[1])<<8); }
inline uint u32(const device uchar* p) { return u16(p)|(u16(p+2)<<16); }
inline float h(const device uchar* p) { return float(as_type<half>(ushort(u16(p)))); }
inline float unpack(const device uchar* p, uint i) {
''' + bodies[kind] + '\n}\n'


@lru_cache(maxsize=None)
def _kernel(kind, experts, rows_only=False):
    import mlx.core as mx
    block, size = QUANT_SIZES[kind]
    names = [f'w{i}' for i in range(experts)]
    choose = 'const device uchar* w=w0;\n' + ''.join(
        f'if (expert == {i}) w=w{i};\n' for i in range(1, experts))
    if rows_only:
        source = f'''uint k=thread_position_in_grid.x;
        if (k>=N*K) return;
        out[k]=unpack(w0+(k/{block})*{size}, k%{block});'''
        inputs = names
    else:
        source = f'''uint lane=thread_index_in_simdgroup;
        uint row=thread_position_in_grid.x/32;
        uint token=thread_position_in_grid.y;
        if(row>=N || token>=M) return;
        uint expert=ids[token];
        {choose}
        float sum=0.0f;
        for(uint k=lane;k<K;k+=32) {{
            uint index=row*K+k;
            float v=unpack(w+(index/{block})*{size},index%{block});
            sum += x[token*K+k]*v;
        }}
        sum=simd_sum(sum);
        if(lane==0) out[token*N+row]=sum;'''
        inputs = names + ['x', 'ids']
    return mx.fast.metal_kernel(name=f'gguf_{kind}_{experts}_{int(rows_only)}',
        input_names=inputs, output_names=['out'], source=source, header=_header(kind))


def matmul(buffers, kind, x, ids, rows):
    import mlx.core as mx
    width = x.shape[-1]
    tokens = math.prod(x.shape[:-1])
    return _kernel(kind, len(buffers))(inputs=[*buffers, x.astype(mx.float32), ids],
        template=[('K', width), ('N', rows), ('M', tokens)],
        grid=(rows*32, tokens, 1), threadgroup=(128, 1, 1),
        output_shapes=[(*x.shape[:-1], rows)], output_dtypes=[mx.float32])[0]


def decode_rows(buffer, kind, rows, width):
    import mlx.core as mx
    return _kernel(kind, 1, True)(inputs=[buffer],
        template=[('N', rows), ('K', width)], grid=(rows*width, 1, 1),
        threadgroup=(256, 1, 1), output_shapes=[(rows, width)],
        output_dtypes=[mx.float32])[0]
