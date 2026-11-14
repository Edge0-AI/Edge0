// metal/kernels/quant.metal - the weight formats as "units": a unit is UNIT consecutive weights of one row that one
// SIMD lane decodes at a time into raw values q[0..UNIT) and one scale s, so that the unit's dot with x is
// s * sum(q[j] * x[j]).  Every GEMV kernel (moe.metal, gemv.metal) is written once over these.
//
// Decode rules transcribed from ggml-quants.c dequantize_row_* (3cf03257) and Strata's Q2_0 contract
// (docs/q2_0-contract.md); the grids and sign tables are ggml-common.h's (prepended by the runtime).  The tables a
// format needs are copied to threadgroup memory once per threadgroup (constant-memory lookups with divergent indices
// serialise on Apple GPUs).
//
// Blocks start at 2-byte aligned addresses (18, 66, 74, 82, 98, 110 bytes), so every multi-byte field is read as
// ushorts.

#define T_Q4_0 2
#define T_Q3_K 11
#define T_BF16 30
#define T_F32 0
#define T_Q5_0 6
#define T_Q5_1 7
#define T_Q8_0 8
#define T_Q4_K 12
#define T_Q5_K 13
#define T_Q6_K 14
#define T_IQ2_XXS 16
#define T_IQ2_XS 17
#define T_IQ3_XXS 18
#define T_IQ4_NL 20
#define T_IQ3_S 21
#define T_IQ1_S 19   // Unsloth UD-IQ1_S / UD-IQ1_M experts (50-byte blocks)
#define T_IQ1_M 29   // the 2.00 bpw pack's 1.75-bit type
#define T_IQ2_S 22
#define T_IQ4_XS 23
#define T_Q2_0 42
#define T_Q2_0R 142   // Q2_0 read with raw activations (codes - 1 in the decoder; no prepared x needed)
#define T_TQ2T 144    // agentionai's trellis types (Gyro): 128 values a block, 34 / 26 / 30 bytes, Hadamard-folded experts
#define T_TQK6 145
#define T_TQK7 146

template <int F> struct fmt;
// QK block elements, BB block bytes, UNIT weights per unit, TAB words of threadgroup table
template <> struct fmt<T_Q2_0>    { enum { QK = 64,  BB = 18,  UNIT = 16, TAB = 1 }; };
template <> struct fmt<T_Q2_0R>   { enum { QK = 64,  BB = 18,  UNIT = 16, TAB = 1 }; };
template <> struct fmt<T_IQ3_XXS> { enum { QK = 256, BB = 98,  UNIT = 16, TAB = 256 + 32 }; };
template <> struct fmt<T_IQ2_XXS> { enum { QK = 256, BB = 66,  UNIT = 16, TAB = 512 + 32 }; };
template <> struct fmt<T_IQ1_M> { enum { QK = 256, BB = 56, UNIT = 16, TAB = 1 }; };
template <> struct fmt<T_IQ1_S> { enum { QK = 256, BB = 50, UNIT = 16, TAB = 1 }; };
template <> struct fmt<T_IQ2_XS>  { enum { QK = 256, BB = 74,  UNIT = 16, TAB = 1024 + 32 }; };
template <> struct fmt<T_IQ2_S>   { enum { QK = 256, BB = 82,  UNIT = 16, TAB = 2048 + 32 }; };
template <> struct fmt<T_IQ3_S>   { enum { QK = 256, BB = 110, UNIT = 16, TAB = 512 + 32 }; };
template <> struct fmt<T_IQ4_NL>  { enum { QK = 32,  BB = 18,  UNIT = 32, TAB = 256 }; };
template <> struct fmt<T_IQ4_XS>  { enum { QK = 256, BB = 136, UNIT = 32, TAB = 256 }; };
template <> struct fmt<T_Q3_K>    { enum { QK = 256, BB = 110, UNIT = 32, TAB = 1 }; };
template <> struct fmt<T_Q4_K>    { enum { QK = 256, BB = 144, UNIT = 32, TAB = 1 }; };
template <> struct fmt<T_Q5_K>    { enum { QK = 256, BB = 176, UNIT = 32, TAB = 1 }; };
template <> struct fmt<T_Q6_K>    { enum { QK = 256, BB = 210, UNIT = 32, TAB = 1 }; };
template <> struct fmt<T_BF16>    { enum { QK = 16,  BB = 32,  UNIT = 16, TAB = 1 }; };
template <> struct fmt<T_F32>     { enum { QK = 16,  BB = 64,  UNIT = 16, TAB = 1 }; };
template <> struct fmt<T_Q8_0>    { enum { QK = 32,  BB = 34,  UNIT = 32, TAB = 1 }; };
template <> struct fmt<T_Q4_0>    { enum { QK = 32,  BB = 18,  UNIT = 32, TAB = 1 }; };
template <> struct fmt<T_Q5_0>    { enum { QK = 32,  BB = 22,  UNIT = 32, TAB = 1 }; };
template <> struct fmt<T_Q5_1>    { enum { QK = 32,  BB = 24,  UNIT = 32, TAB = 1 }; };
// trellis: a unit is one lane of 4 steps = 16 weights; TAB = the 2048-point codebook as half2 words
template <> struct fmt<T_TQ2T>    { enum { QK = 128, BB = 34,  UNIT = 16, TAB = 2048 }; };
template <> struct fmt<T_TQK6>    { enum { QK = 128, BB = 26,  UNIT = 16, TAB = 2048 }; };
template <> struct fmt<T_TQK7>    { enum { QK = 128, BB = 30,  UNIT = 16, TAB = 2048 }; };

static inline uint rd32(device const uchar* p) {
    device const ushort* s = (device const ushort*)p;
    return uint(s[0]) | (uint(s[1]) << 16);
}
static inline float rdh(device const uchar* p) { return float(*(device const half*)p); }

// ---- threadgroup tables -------------------------------------------------------------------------------------------
// layout in `tab` (uint words): [0, G) the grid, [G, G + 32) ksigns_iq2xs as bytes; IQ4_NL/XS: 16 floats.
template <int F> static inline void load_tab(threadgroup uint* tab, ushort tid, ushort nth);

// QUANT_CTAB (function constant 23): the i-quant grids and the sign table read from ggml-common's constant arrays
// instead of a threadgroup copy (no table load and no barrier wait at the threadgroup's start); the same values.
constant bool QUANT_CTAB [[function_constant(23)]];
constant bool ctab = is_function_constant_defined(QUANT_CTAB) && QUANT_CTAB;
static inline uint ksg(threadgroup const uint* tab, int at, uint i) {
    return ctab ? uint(ksigns_iq2xs[i]) : uint(((threadgroup const uchar*)(tab + at))[i]);
}
static inline uint g_iq3xxs(threadgroup const uint* tab, uint i) { return ctab ? iq3xxs_grid[i] : tab[i]; }
static inline uint g_iq3s(threadgroup const uint* tab, uint i) { return ctab ? iq3s_grid[i] : tab[i]; }
static inline uint g_iq2xxs(threadgroup const uint* tab, uint i, int w) { return ctab ? uint(iq2xxs_grid[i] >> (32 * w)) : tab[2 * i + w]; }
static inline uint g_iq2xs(threadgroup const uint* tab, uint i, int w) { return ctab ? uint(iq2xs_grid[i] >> (32 * w)) : tab[2 * i + w]; }
static inline uint g_iq2s(threadgroup const uint* tab, uint i, int w) { return ctab ? uint(iq2s_grid[i] >> (32 * w)) : tab[2 * i + w]; }

// BEGIN tq_lut (generated by metal/tools/gen_tq_lut.py from third_party/ggml/ggml-common-tq.h; do not edit)
constant uint tq_lut_u[2048] = {
    0xb8f7bd4d, 0x3cd5b6ce, 0x4368bd9e, 0xb8a63dce, 0xb430395e, 0x3dbc3bc1, 0x3c9cb92b, 0x3aaabca2,
    0xb8e13cdb, 0x3dfbbb33, 0xbb183a8d, 0x3da0bd64, 0x3b83bf1a, 0x3659b389, 0xabd7b40f, 0x35a240ee,
    0x3972c057, 0x3996be17, 0xb9b5b926, 0x3c75393b, 0xbdef4151, 0xbd09b5a3, 0x4012b408, 0x3ece3b7f,
    0x4011b688, 0xbe303b17, 0xb4bf3d9b, 0xb8c8b870, 0x38b2b537, 0x34b031be, 0x38713389, 0x38b8ba33,
    0x3b3dbce0, 0x3d3d3367, 0x3cbe3f50, 0xb99130b0, 0x39863733, 0xb6d9b70e, 0xbd98be8c, 0xbe04bcca,
    0x403c36a9, 0x3eaeb527, 0x39823c6a, 0xbd5bb655, 0x35963ebb, 0xbbbd417e, 0xb1d32fe0, 0x31c0b858,
    0xbc10b54e, 0xaf8b3cf0, 0x3f5f38e6, 0xb2f63b87, 0xb675b114, 0x329829bf, 0xb955bb7b, 0xa98434e9,
    0x378fc04a, 0x3c8c31a0, 0x3c682dea, 0x2d292fe1, 0xbd722c33, 0x3e3e379d, 0x3cf2b918, 0x38bfbcbe,
    0xbdd53284, 0xbd013707, 0x38c1bec1, 0x3838b960, 0x30103413, 0x3672376d, 0x3f42302f, 0xbe9a3a61,
    0xbc713c06, 0xbcf2bb57, 0x3c85a9dc, 0xb8b23873, 0x340d2d1e, 0xb73f3661, 0xb8b7bb91, 0x3943b956,
    0x39853958, 0xb5563b26, 0xc2b9ae17, 0xb9ea34e9, 0xb6f433d5, 0xb1af3826, 0xaf98b8a4, 0xbc693a2d,
    0x358e3a25, 0x32fd406d, 0xb11a38a4, 0xb02eb170, 0xb5cbaa0a, 0x3d453c04, 0xb55bb258, 0xbbf5b412,
    0x3f4b3d33, 0x38a9bc6e, 0xb715b8c0, 0xbdd92ec6, 0x2980beb3, 0xbb3c3c54, 0x2d683ba0, 0xb20942f9,
    0x3337b557, 0xbda73870, 0x35bd2d57, 0x385eb8cd, 0xb68e39fe, 0xbd743eda, 0xbd543643, 0x37d43a32,
    0x2f7f38f8, 0x380f37f3, 0xbc231b1e, 0x35c825c7, 0xb82a2e3e, 0x3897b85f, 0x3cb33c0c, 0x377bc13e,
    0x3f09b3a4, 0x3c73c006, 0x3ad1bda5, 0xa3cfb0c6, 0xb680b9d5, 0xb493384b, 0xa9853a3c, 0xba24bcb5,
    0x3c21bd0c, 0x34dc2f68, 0xbdb7bc5d, 0xbddbbbbf, 0xb95ab4c4, 0xbdaab59b, 0xbdce34cf, 0x3f9d3bad,
    0x3bd53a05, 0x2c64b69c, 0x3adb4003, 0x35e83d4c, 0xb5cc308d, 0xbb86bd2d, 0x3cd3b124, 0xb2ab3a0a,
    0x3d4bb507, 0x284cb190, 0xbe83384c, 0xb806bc94, 0xb14631e4, 0x3178bbe0, 0x3819bc9d, 0x2e8f3824,
    0x4150c246, 0x3a43bcf3, 0xbdd53da9, 0x3c47b229, 0xc001400e, 0xb0c33bb2, 0x3822bd46, 0xa4a9b92d,
    0x3cfa3c72, 0x42843550, 0xbc97b9e5, 0x357ab996, 0xaf70ba03, 0xbc0735e7, 0x397bb663, 0x29423c59,
    0xb689adf7, 0x3db8b875, 0xb63c35ea, 0x33493674, 0x3ae03926, 0xb56ebb20, 0xa9ae3f05, 0x3c383ce5,
    0xb1ef40b8, 0xba2bba73, 0x3d3fbdd7, 0x3c093822, 0xbce4bd87, 0x36beba38, 0xbe23b808, 0xbd77ba33,
    0x34663b90, 0xb805c1c5, 0x375b34c0, 0xa2dbb4be, 0x3495b96c, 0xb99ab38a, 0x31d5390a, 0x3e16a910,
    0x3dd62878, 0xbdc03964, 0xb3efb783, 0xb571ba12, 0xc0f2b8ce, 0x3771bd8f, 0x3eba3656, 0x3c39be0e,
    0x371b3ac5, 0x40713469, 0x2e19bd99, 0x3cb83ab2, 0xa92c36d9, 0xbc9bbcb4, 0xb4e2a0fa, 0x31fd34ed,
    0xb0a04033, 0x342ebac8, 0xbccd3df5, 0xbc5fbc5a, 0x3ada3f32, 0x39ec3878, 0x3ce53db1, 0xbe93b891,
    0xb75e3c11, 0xbb27b82c, 0xba6736d6, 0x352cb602, 0xbf3a3507, 0xbcf5b8e9, 0x3773b55e, 0xbd0ab832,
    0x3989bf01, 0x3d733edf, 0x3bc8bb3d, 0xb7e0bf82, 0x393f4053, 0x2f91b07d, 0xb6afbc8b, 0x2a463538,
    0xbbfd3c2f, 0x324ebfc9, 0xb7023a8f, 0xbda9b939, 0x3c3f3ac3, 0x37983875, 0x3e1b3dc4, 0x255eb820,
    0xbe92a7e2, 0xb7c7b63f, 0xbcb438bd, 0xb9583a2f, 0x34fcbdf2, 0xb5a83ba1, 0x30473975, 0x21daaecc,
    0x3bc4b7f7, 0x2eb639f8, 0xbb3abb3f, 0xb13dbd9b, 0xbab237fa, 0x30743794, 0xbad3bd7e, 0x346b3d65,
    0xba613cfd, 0xb8feadd5, 0xb6ecbe09, 0xb544c05b, 0x2fdb360c, 0x39e5b8ba, 0x3e96b8a8, 0x38cf40cf,
    0xac5aba8b, 0xb284b691, 0xb5d5c0b9, 0x3d003864, 0xa3823220, 0xbb8f3422, 0xb85eba85, 0x3eb33854,
    0xb885c110, 0x2e60bdf8, 0xb8ba1cb3, 0x3636b4d3, 0xaf93382a, 0x346abc81, 0x31483d66, 0x36fec0b3,
    0x4001af70, 0x3eac3caf, 0xb614bd48, 0x3de83d04, 0x31d03f2b, 0x30ceb955, 0x2d33b81b, 0xb657b328,
    0x3527b7c2, 0xc255b9f6, 0x3c90bc1a, 0x35f3be9c, 0x409ac11d, 0xbf2f30db, 0x3285be02, 0x2f14b607,
    0x35c23c56, 0x30ac3288, 0xc25c3c1c, 0xbbd63ca4, 0x29c83465, 0x412937a3, 0x32253151, 0xad412ef2,
    0xbfc1bbcb, 0xbe71b0c0, 0xbb59b17d, 0xbefb3cfe, 0x39f0bdb8, 0x28a74014, 0x396ab771, 0xc063bbdc,
    0x2898babe, 0x3de9b683, 0xaa78bd1f, 0x3ded4187, 0x3d7b3e18, 0xab0caf3c, 0xb52438a7, 0xc38fb964,
    0x3d02beda, 0xb25daa74, 0xbe9a2d90, 0xbbb0be23, 0x3a8cbf6a, 0x3d87bbfa, 0x3571b53c, 0xc027385c,
    0xb801b903, 0x3b7b3368, 0x3ab8b70c, 0xb62538b5, 0x3cb1b436, 0x3f2c8d2a, 0x2c53b88e, 0x393bb2c8,
    0x3f40bdbf, 0x39d9bc1b, 0x306dbc3a, 0xb8fcb3ba, 0x3c75c07b, 0xb421b655, 0xb7203d34, 0xb6a037f1,
    0xbec7b338, 0x334eba58, 0xb8103aab, 0xbe3db3d7, 0x40a2b984, 0xba54b5cd, 0xb63e3afa, 0x394dbd80,
    0x2ddaba54, 0xb8e4baec, 0xad382b2a, 0x3886bd82, 0xc015ac6b, 0x3c72368b, 0xbaa72fb3, 0xb2cb38fb,
    0xbfb5335d, 0xa535aba9, 0x38b7afb8, 0x35243d00, 0xaeac3b42, 0xb1a64144, 0xba3ab352, 0x39f32c72,
    0xb82c4079, 0xb45fb9e5, 0xb9f59d20, 0xbd36b2ee, 0x2f11bad5, 0x3e803450, 0xb684213f, 0xbd073200,
    0xbc5eb839, 0x3f5dc217, 0xbe943d9a, 0x33813947, 0xb9439081, 0x37953eac, 0x3915ad61, 0x3d5db810,
    0x33f239cb, 0x41edbfe8, 0xb4a23ee7, 0xb88ab9e5, 0xc0da36b3, 0x406cb57f, 0x3c5cba70, 0x3ae4b936,
    0xb7ff3e2c, 0xb08e35b6, 0x3c3ab8d1, 0xafed227e, 0x310a3a3d, 0xb9262d00, 0x3f76bbe1, 0x3e15b8aa,
    0xc074b31c, 0xbcb640ee, 0x3b033449, 0xbd0138f5, 0x386abc2d, 0xbcfc3f94, 0xb04933a2, 0xb86e3c77,
    0xb91c366c, 0x35c6b832, 0xbc4c36ab, 0x3a04b6b9, 0x3c84b2c9, 0xb7623006, 0x3580330e, 0x39263d32,
    0xbfb238f9, 0x3de6c00a, 0xafad3a35, 0x3d88be69, 0xbe12c30f, 0x3d26b13d, 0xb7a5382a, 0x3ac340fa,
    0xc028b8a9, 0x3e19b979, 0xbdd7bac1, 0xb802bd42, 0xbe1ab631, 0xac7ab8fb, 0x38193ff8, 0x42e83d3f,
    0xb1dcb312, 0xaa42c036, 0xc078bd28, 0x36ddb6f5, 0x31fdbad4, 0xb7e93933, 0x31d439b4, 0xb4aebd13,
    0x3c5d37d3, 0xbafd3925, 0x38753b52, 0xa5812a38, 0xb4503b51, 0x3cccc248, 0x3ad2b1aa, 0xbe9036eb,
    0x34e3423c, 0x3bc13ac2, 0x37493be5, 0x3ceebcd1, 0x40173ac7, 0xbfc43a67, 0xb9a8b7eb, 0xbf08bf94,
    0xa983b711, 0xb6093c0f, 0xb7f73ebd, 0xbca734f4, 0x38b1bac3, 0xb93fb5d0, 0x3f1d3e04, 0x305cb7fd,
    0xbcaaaae3, 0x3d8cbb0f, 0xbd3fbca4, 0xbc08bb3b, 0xb9de3a90, 0xad25377f, 0xbefeb988, 0xbc6cbdf7,
    0x35c3ba23, 0x370abb14, 0xb3a93c45, 0x3c72b889, 0x3e713a36, 0xb8e03a90, 0x34a634cb, 0x3c6c40be,
    0xbc3fbe95, 0x2284ba4d, 0xb73eb96e, 0xbc04384d, 0x3a03b452, 0xc10041a8, 0x2ed23c75, 0x3ab6bc47,
    0x3a633746, 0x3735b62f, 0x3c2fb9dd, 0xb08fbba8, 0x3289b270, 0x3c1e3972, 0xb5ae3844, 0x3d303cee,
    0x3fa2b98f, 0xb1503e16, 0x382c3c9c, 0xae2e3483, 0x3a843b71, 0x32f4bb53, 0x3ab23540, 0xba06b6e0,
    0x3701bcc9, 0x361ab89a, 0x3a74be12, 0x37ae3b56, 0x38eeb6ec, 0xb60ab47a, 0x374a3247, 0x3883394d,
    0x2c41b76f, 0xb862bc07, 0xbba5395e, 0x3865b71f, 0x2cbc3897, 0x35c9ac7a, 0x3ac5b9df, 0xb8b9b63b,
    0x3ff6c064, 0xb8293ba3, 0x3b7eb904, 0xb78eafdb, 0x3c22baec, 0x39a5b53d, 0xc0d4bab6, 0xbddbb47e,
    0xc01cba66, 0xbcb037ec, 0x3ac3b409, 0x3e472b2b, 0xb8e03b8a, 0x3a7738cd, 0xb7243cb0, 0x381fbaef,
    0xb0f9ba7a, 0xb1383940, 0xb8403855, 0x35f439af, 0x2c193b0f, 0x3b833760, 0x3801a813, 0x41d6b801,
    0x3ab33026, 0xa9f5bca3, 0x370ebf1e, 0x3467b82b, 0xbd57b4c3, 0xb4e9ad4c, 0xbad6b8c4, 0xbd66a8f7,
    0x3cd23e70, 0x3421236c, 0x3c22aa8a, 0x9661b9c2, 0xb6223988, 0xb8aab797, 0x305c254c, 0x271bbd53,
    0xbaa93c11, 0x3601b75b, 0xb4e936d1, 0xb15db856, 0xc0b2be96, 0x3e32b7c5, 0x3e22b069, 0x3b41bc18,
    0x31cac16e, 0x35f9365a, 0xb700b491, 0xb41b3a6a, 0x406a2f26, 0xbd54b8a6, 0x39e33b9b, 0xad0fbeae,
    0xb1663b1b, 0x3655afd4, 0xbaa7bde7, 0xbd6c37c1, 0x3cbfbc6d, 0x2dbd3584, 0xb9f93d69, 0xac53b992,
    0xb2a2bf6b, 0xc15735b5, 0x328e331a, 0xb7b3ba14, 0x38093c44, 0x33613c08, 0xbc1c3b93, 0xbd213b53,
    0x39b2b9d1, 0xbad9326b, 0x35e6b1c8, 0xaeff38ff, 0xb734b831, 0xbbe1ba96, 0x36fb33f8, 0xc15ab049,
    0xb7a33b22, 0xbc3db462, 0xb57cbf97, 0xbcaa3c86, 0x3c6ebcb9, 0xb0f539ca, 0xb5bf379f, 0xbcb6b629,
    0x3f0f3aab, 0x34fcc01a, 0xb6be3869, 0x32f4b953, 0xbdb7b85f, 0x411a39f4, 0x203cbce8, 0x2a593824,
    0xa60cb26f, 0x3d96b47f, 0xb53fb8d0, 0xb69f3706, 0x36cfb1c9, 0x37c1ad03, 0xbc903d80, 0xb98eba26,
    0x334b3c5b, 0xb051c196, 0xbd79bd24, 0x3e2f3b38, 0xb64dba82, 0x34ed2a56, 0x2a01b36f, 0xbfbc40e9,
    0xb600b7ce, 0x3a9c3cce, 0xb23c33c6, 0x36c83c33, 0x409ea776, 0x3c35b822, 0x30ef3e43, 0xb9f7b9b2,
    0xba813c79, 0xaedb3973, 0xba33bfe1, 0x3d8a348e, 0xbfa1b5ac, 0x31b7b8d9, 0x34813aee, 0xb56e3d3b,
    0x2c653d98, 0x3d7439e0, 0x3ae2b51a, 0x3bb9ae6c, 0x1d86bdc8, 0x3840b9d9, 0xc0f1bc60, 0x396d3e26,
    0xa4783dc5, 0xc01bbf35, 0x39c8b938, 0x3fd035d1, 0x3d41b3bc, 0x3bf9bca8, 0x32f5bd36, 0x34c83f59,
    0x38543e38, 0xb82e338e, 0xa3ce37b2, 0x37fdb683, 0x3f02c104, 0x38b3b92f, 0x3bcdb454, 0xba4231f4,
    0x3878bf4d, 0x3559bf7f, 0xbc53b8e0, 0x407ebd77, 0xb0c6b69e, 0xbabdacce, 0xb92f3c1b, 0x3a5fb8fa,
    0x3f13b628, 0xbacd3893, 0x3ec8c047, 0xba7fb70b, 0xb140b997, 0xb4662c0c, 0x3ddeba3f, 0x3ae1be75,
    0x3879b466, 0x3c553d68, 0x3c6eb4d9, 0xb803be71, 0x2a523e15, 0xbc373e1b, 0xb7473585, 0x3db4bf22,
    0x31a6bcdb, 0x2e23b415, 0xb884ab25, 0xb5ef329e, 0xb59c3525, 0x39c2ba76, 0xbf1eb6f5, 0x2dadbcdf,
    0x40193e8c, 0xbb4ab92c, 0x2ddb3494, 0x3afd406e, 0xb97a3b91, 0xbce13c1f, 0x3c343eae, 0x38ecbddd,
    0x3e4f390f, 0x3e04bc20, 0xba0b378c, 0xb8aa330d, 0x348e3e34, 0x352fbaa5, 0x3d2bbac3, 0xbe71b982,
    0xc1e1305c, 0xbe12be05, 0x2fc1b70b, 0xbe713c8b, 0x3dcc35ad, 0xb612b581, 0xb50cb42c, 0xb36cb1c9,
    0x3977b443, 0x3d11bd4c, 0x3b1739b8, 0xbd27269a, 0x3d6cbce0, 0xb611b86d, 0xb50c35e7, 0x359cb453,
    0x38bf38c6, 0x39a6b82a, 0x3bb1a8c2, 0x3c8bb020, 0x2dfa2c82, 0xb43ebb37, 0xbbbd2b23, 0xbd253c97,
    0xbb813d21, 0xbe12b8e5, 0x3d6a3845, 0xc0ecb571, 0x34ba3c69, 0x2818b96f, 0x382dbdde, 0x3f333f0c,
    0x3c30b44b, 0xbef5aebd, 0x26833a3b, 0xb7f8aaf0, 0xb4f5405e, 0xb1973c5b, 0xbada359f, 0xbb773227,
    0x3dec3877, 0xb7c634a2, 0xba0e3bde, 0x20ccb758, 0xb8e83760, 0x3cb8bdaf, 0x385d3cf3, 0x3078306f,
    0x34d1ac7a, 0x402fb9b8, 0xbc802d2e, 0xbeefb55d, 0x356a348e, 0x3b683511, 0x32ea35a4, 0x3a133e99,
    0xb8c94002, 0x355a3dc1, 0xbb783556, 0x37ee3560, 0xbc4aba6e, 0x33ebb14d, 0x3c4a22a9, 0x3a27bd5b,
    0xba1dc0ca, 0xbd314053, 0xb87b3c1a, 0x36e338de, 0x2b952663, 0x36853b4f, 0xc020353c, 0x4050be97,
    0xa53abc48, 0xbeea3e4e, 0xbb52ac24, 0x40d2b743, 0x3762b389, 0xbb02b493, 0x342f3cb4, 0x3aceba83,
    0x3923bc28, 0xc0afc004, 0xb842b898, 0x276038d0, 0x405eb83e, 0x4254affb, 0x3c6f3b69, 0x3bedba70,
    0xba70b0de, 0xb6353d9c, 0x3de8bde3, 0x38e53321, 0x40ae369f, 0xbaf236e3, 0xbb9baf7b, 0xc1f3b484,
    0x40973f5a, 0xaead3da5, 0x1e3a3934, 0x409fbc67, 0x3a55b2ed, 0x385a3009, 0xb6ac3166, 0xc3f83832,
    0xa95fb800, 0x2f61bc8a, 0xbeba353c, 0x3ce33284, 0xade83fad, 0xb7c13da0, 0x382db56e, 0x3855385f,
    0x2ff5bd34, 0xb290ba07, 0xb6f4bda4, 0x3ad2b89b, 0xbe273613, 0x35fc3838, 0x3c9628e1, 0xbd623c1e,
    0x356b315a, 0x3b163b05, 0x3ed5baf3, 0x3c65bf42, 0x378a3d3f, 0xbd43bf4e, 0x39643c0d, 0x3c0034c3,
    0xbafe345a, 0xba8bbf4f, 0xb95b4240, 0xc0a23b21, 0x331db6ed, 0x3111b57a, 0xb8ca3594, 0x3d96c16a,
    0x3c86b7a9, 0xb498b949, 0x3c1e2d89, 0x2eebad06, 0xba8abc5b, 0xb0b9be54, 0x3e71bc9a, 0x3ff43ca7,
    0xbbbcbf8b, 0x3fefb894, 0x40c93bda, 0xbb663bc7, 0x3cbbb87e, 0xb1f7bb1e, 0x38f63a3d, 0x3cd9ad22,
    0xbd81bdb4, 0xbd96b005, 0xb91c3d6e, 0xbd5e38f2, 0x2e27b513, 0xbe1c3e5b, 0x366e34b3, 0xb6cdbb24,
    0xbfb436f7, 0x34a3b00c, 0x37dfb1bf, 0x33d3b9d6, 0x3edf396f, 0xb6f7b5d1, 0x38b22aa1, 0x223d36b1,
    0x3d3ebc6a, 0xb7b5b777, 0xb91cb8ed, 0x35f53be6, 0xbd74c077, 0x3966bcdc, 0x3d0bbc12, 0x28cb3312,
    0x4128bd0d, 0xb80ebef1, 0x3eb4b758, 0x352fb6d3, 0x3ca3343d, 0xbaefb733, 0xbc1dbce4, 0x38fa3eac,
    0x3bceb9d1, 0x36ec3953, 0x3985bbb4, 0x3fe24005, 0xb9913f4e, 0xbe2c39d5, 0xb37fb44e, 0x30aa3acf,
    0x40c3b358, 0x3b9b3c1f, 0x39a739ec, 0x41113d01, 0xbb76367d, 0x3a483c75, 0x3c01b547, 0x3ae03d2f,
    0xb837c3b1, 0x36c12e1d, 0xbd8eb2d3, 0x3d80b23a, 0x3c9bbd29, 0x348c37a8, 0xb5aa3c9e, 0x40d33311,
    0x4028256d, 0xa66e2e47, 0xb8de390a, 0x3c033306, 0xbb3fbc58, 0xbc21bc05, 0x28e2376c, 0x3ddab536,
    0x3ebb2427, 0xbdbfb70a, 0xbb103cc4, 0x2d293d33, 0xbfd0c044, 0x3726bfd3, 0xad3432d9, 0x3d32c040,
    0x2b123f66, 0x3ae9377a, 0xc02d3088, 0x38e2384f, 0xacfb3c2b, 0xb9debbda, 0x39452d0a, 0x40ba38f1,
    0x1b913858, 0xba152f58, 0x3a7f3642, 0x387a39d4, 0x3efcb9af, 0xbcc6c007, 0x3938ba9b, 0xaa27b635,
    0x3509a2e3, 0xbb66b38b, 0xb775bbc3, 0x33f5bbc7, 0xb6bac014, 0x340ba91a, 0x3dadc0ad, 0xb70ea94d,
    0xbcf0b9b4, 0xb38a2f73, 0x304e34f1, 0xbfe63cdf, 0x34c3c329, 0xbdc03c80, 0x39ed37bd, 0xaaa3393d,
    0xbd493a94, 0xb4d2c288, 0xbfb62c3e, 0xb93dbca0, 0x3200b9d6, 0x3875ac36, 0x33f73072, 0xb8f73e47,
    0x3116ba54, 0xc0373a12, 0xb48c3ad7, 0xb798323b, 0xbf50c11c, 0xaf0baf13, 0x2efc3c1c, 0xbc683b04,
    0x38fab611, 0x3f8cb74f, 0xbf1a3b22, 0x3cefc394, 0x26143ca6, 0xbe0337c3, 0xc0723cd0, 0xc15f3f2e,
    0x34da3c11, 0x3cf5b4e7, 0x3030c030, 0x3837b32a, 0xbca2c1c0, 0x3ef133dd, 0xac37982a, 0xb7b33c5e,
    0xb9cdb5b6, 0x2c0cb9e9, 0xab2a3832, 0x419aacea, 0x39223458, 0x3c1fc18e, 0xb7622ae8, 0xbe8c393d,
    0xbd47b009, 0x3abd3d9b, 0x3c56b95a, 0x31073cfe, 0xbc5637d9, 0xb846312d, 0x3d2e3752, 0x403041d2,
    0x3bf0b2d1, 0xbc3930c5, 0xb8aab1ee, 0x3d48405a, 0x2a023c11, 0xc184b800, 0x308d3dd2, 0xbd48b975,
    0xb94232b5, 0xbe0e38c6, 0xc06f3688, 0xbc6133b3, 0xc0943910, 0x38f83cb1, 0xc2b8be18, 0xb3de37d6,
    0x3b2eab85, 0x299cb8f6, 0xbdb13676, 0x3f26b87d, 0xc1173068, 0x3a16b7ae, 0x3b4430ea, 0x30a9b347,
    0x39c331a2, 0x302136bf, 0xbc1338f6, 0x338db882, 0x3bf4b892, 0xba38385d, 0x305d3b4e, 0x387b43aa,
    0xb0c436a4, 0xb925b800, 0xbdefb9d7, 0x4244bba6, 0x345236da, 0xc1913b75, 0x318b381f, 0x405a38a3,
    0x3d74ba2a, 0x3666b9b5, 0x38023990, 0x37ccb8ed, 0x36e93697, 0xb3ef3d43, 0x3cbfbaa8, 0xadc4bdf3,
    0xb0b62c73, 0x386abbd5, 0xbecebaaf, 0x3d5835f2, 0x3ca3358e, 0x3d37b987, 0x3cba3770, 0x400f3d8a,
    0x391037a5, 0x31483bce, 0xb1f2b10b, 0x3a6d343f, 0xb733399b, 0x355038b6, 0xb90abe87, 0xbe4bc1ea,
    0xbe8fbe96, 0xb92a3999, 0xb1c83a77, 0x3508366c, 0xbcc4b7aa, 0xb591b03d, 0xb9fcbecf, 0x40f5bb2f,
    0xc0db4073, 0xbf44421c, 0x34cd339f, 0xb4dc37bd, 0xb44cb531, 0x3b383e9d, 0xa9ffba12, 0xb9ce3655,
    0x3441b32e, 0x3c083f6f, 0xb8603a43, 0x3dfebcf7, 0xbc86c07a, 0xbbf63745, 0x36a62987, 0x267f35e0,
    0xb9f1b4b8, 0x3a39b865, 0xbd262f53, 0x388c37b5, 0xb8893b0e, 0xa4223c18, 0x1666b894, 0xaf72b37f,
    0xb0813772, 0xb0eac090, 0x41d033ad, 0x345fb75b, 0xba823de3, 0xb979afd1, 0x36ec3f52, 0x42c7b928,
    0xb54ebc71, 0x356d3b45, 0x3d77b63d, 0x2873a722, 0xac223ab5, 0x3ccbbe38, 0xba9db82f, 0x34f9b88a,
    0xb9973dde, 0x399dc1b1, 0xc19a3d55, 0x3e72b380, 0xbd0ab701, 0xba7fc16c, 0x3df039a8, 0xb595bf04,
    0xb904ba46, 0x35c93ab7, 0xb9dab181, 0x3a40ba29, 0xb257bee1, 0xbabbc262, 0x39fa3d35, 0x39222661,
    0x3c35b5da, 0xbc02ac1e, 0x375bb9c2, 0xb5e8343c, 0xbd4abc03, 0x38402c70, 0x3a48b98c, 0xb987bd06,
    0x3a3fc008, 0xadf03676, 0x437f391e, 0x38df4179, 0x3dd23a92, 0xb870b42f, 0xb0b33e99, 0x3c3b3136,
    0x3b57b98b, 0x3d80af33, 0x3475b578, 0xb6203f58, 0x386c3a60, 0xbc68af9c, 0xb99a3826, 0xb414b8c8,
    0x374bb952, 0xba0c40e7, 0x40723a49, 0xbc2ab76d, 0xaebbbc67, 0xba0d3a0b, 0x3e433f56, 0xbc433d02,
    0x3e82b9a2, 0xb7963a1f, 0xbc5c355d, 0xba8a3a70, 0xbd61333c, 0xb9c73caf, 0x31f0bd93, 0xbce5bd05,
    0xba81b4b9, 0xbbf2bc7d, 0xb9763740, 0xbe7bb53f, 0xb266bced, 0x3bda30be, 0xbab03f03, 0xbfc73bdf,
    0xbf88adc6, 0x2c42b00b, 0xc2c93e4d, 0x4114be94, 0xbcba3b5a, 0xbd313d7c, 0x37ac3726, 0x3a3a397e,
    0xbcd02f06, 0x3cb93d0c, 0x3f3735a6, 0xb16141f6, 0x3e57ba84, 0x3e1eb423, 0x2e083eb8, 0xbc7fb1c0,
    0xc03a3bef, 0xbb16bcc4, 0x3c4a38a5, 0x3bf83b7f, 0x296439c9, 0xb944bddc, 0x3717406a, 0xaef2b80f,
    0xbf1438c2, 0x37eb3633, 0x3d192c27, 0x37d2ba23, 0x382331dd, 0x39a3bb12, 0xb57cb985, 0x36a5b7fa,
    0x3b5bb882, 0x34c03856, 0xb60d3a69, 0x3b6ec03c, 0x219c3b95, 0xbdd6ac3a, 0x356f4008, 0x33cbae4c,
    0x3c41bd84, 0x3b3c3ba0, 0x387ebb56, 0xba143b22, 0x3f8d33cb, 0x18e03508, 0xa89e3b2d, 0x3c7f3c83,
    0xb82a3cd9, 0x391239b5, 0xbb1c39d4, 0xb1deb8ed, 0xbb923b0b, 0x41783be0, 0x31e83716, 0xbc80b3ff,
    0x393eba07, 0xba8739a9, 0x4185bb2d, 0x32fdb616, 0xbbe7b9d6, 0x3cee3645, 0xbcbab4c6, 0xac743e24,
    0xb9ba2b37, 0x31423c42, 0x33e93505, 0xb53ebe07, 0x3b1db609, 0x36e4bd2b, 0x36fb3c8a, 0xb8c339e9,
    0x42953fba, 0xbf1a3766, 0x3ca339e3, 0xb4343fc7, 0xbb95b899, 0xb2e3398e, 0xb5c8bcd6, 0xb44238c5,
    0x407d3cf2, 0xb847b6e2, 0xbdcd3a6a, 0xb34234b0, 0x2d943a83, 0xb6443e96, 0x3e5dbead, 0x39cdb0bb,
    0xba58a92f, 0xbdf93c0b, 0xa45939ad, 0xbd5c3e21, 0x41c83d83, 0xbeb73299, 0xbc3ab260, 0xb98cb6c7,
    0x3d96321a, 0x3a6f3834, 0x3d353920, 0xbe2828a0, 0xb9dbbe36, 0xbbbab7d3, 0x357fbbd1, 0xc114bd91,
    0xae9bb5fd, 0xb85bc078, 0xb4c3c123, 0xb79638b3, 0x3ab8bb1b, 0x37e533b3, 0x3a311a62, 0x3a67b4f1,
    0x3e7bbbce, 0xbc7cb73c, 0x4098408e, 0x342bbe64, 0x39c73da6, 0x1dea4077, 0x3c46af4e, 0xbbe83ebd,
    0x3cd83936, 0x3e99402f, 0xb2d23c05, 0xb1ebbc89, 0x3d842dfa, 0xbb80bed9, 0xbb7238a5, 0xb173bc2c,
    0xb35db84d, 0x3e9fb0ff, 0x3b57bc79, 0x307dbb5b, 0x3703b498, 0x38bebe4a, 0xab2eb878, 0xbacc3b46,
    0x32d83ab4, 0x365bc20b, 0x36ed35ac, 0x2d1b40e5, 0xb0f7b594, 0xbe4cba7a, 0xb070a99c, 0x2dcab21f,
    0x3fd8bf8a, 0x391bb83e, 0x330fbc35, 0xb69834e7, 0x34c1b482, 0xb81836cd, 0xb4493e69, 0x31ab3646,
    0x341e38c2, 0xb9873925, 0x41b938e5, 0xb1acb79d, 0x38f936c6, 0xa977c103, 0xbce0bead, 0xb897bc63,
    0xb8423f5d, 0x38753d63, 0xad0bab3a, 0xbcd625c0, 0xb85138d6, 0xc077b9a9, 0xb4a13be3, 0x39c4c293,
    0x414b32dc, 0x38cc3ada, 0x3b60bad6, 0xb52e3a85, 0x3d6d3d6c, 0xb855b5a4, 0x27f62ccf, 0xbcf635b1,
    0xb4a034df, 0xb62abe85, 0xad0d39d4, 0xab88b542, 0x345a35dd, 0x3951353b, 0x2f10b8f8, 0x363fbaae,
    0x3ca23878, 0x38da35df, 0xbcfdb443, 0x2fdbb87e, 0x36abbc75, 0x3243389a, 0xba7029dc, 0x32502ee6,
    0x3d29b748, 0xbe2140a9, 0xb60dbba6, 0x3c40bbbc, 0xbb7bb5dd, 0x3bf32631, 0x2ad5bfbb, 0xb41735b7,
    0x3831afde, 0x3145b651, 0xb23f3cfa, 0xb56f39ff, 0x3f03bd12, 0x362333d8, 0x3e043401, 0x3f6bbe9f,
    0xa9d730db, 0xb34d31d9, 0x32529f5e, 0x3c75c0f6, 0xb8ec348d, 0x384138d0, 0x20a83abf, 0x2b97c090,
    0xac01bf35, 0x3722b854, 0x344dc074, 0x39723a91, 0x39b9be82, 0xc02ab62c, 0xba61bb3a, 0x3a292fcb,
    0xbb4eba51, 0x3d17b604, 0x3b16363e, 0x3ab93c14, 0x3a9cb80d, 0x3936b0df, 0xb8cab50c, 0x35f3bd85,
    0xb02b2ffc, 0x38fec0b9, 0xba48390d, 0xbc03408f, 0x3646b937, 0xc0c1c22b, 0xbad0bab4, 0x99a8b3f6,
    0xbf40bc8c, 0xbc802074, 0xb89bb927, 0xa9e533be, 0x3229417f, 0x353cb346, 0xb883395d, 0xbadd3d69,
    0xa7433c6c, 0x3466bd97, 0x3977bf85, 0x39763618, 0xb9f83e77, 0xc024be06, 0x325abc84, 0x3d18bb6c,
    0xbca2b930, 0xbfb2b806, 0xb2533578, 0x33f1b48c, 0xbb21b9ac, 0x374ab012, 0x3c0bb0cc, 0x3dc8aca9,
    0x322cb4d9, 0xb993ab2c, 0x3a913aa6, 0x3da642a1, 0x3e593151, 0x2d8e3751, 0x2d9a3365, 0xb77fbaaa,
    0x32eeb7f0, 0xbb783dd2, 0xbe1c3d0e, 0xbd05be19, 0xbebdbcfe, 0x36b0bba1, 0x40063962, 0x370d380e,
    0x3b9c3e08, 0xb2a82bfe, 0xc0383dd4, 0xb32cc00f, 0x39f333b5, 0xb8933691, 0xac1fbb79, 0xbd823d09,
    0xb867afcd, 0xbbb5c01c, 0xb323bbca, 0x3d5db8d8, 0x2db4bbda, 0x38f9b8ca, 0x3a87b615, 0xc099321c,
    0x3f8c3a26, 0x36f130a7, 0xc2a5386f, 0xb157b485, 0x3f94b4d2, 0xb1bb2056, 0x35dbbb33, 0xba5135b5,
    0x845ebe3f, 0xbd67b7cd, 0xb6702dae, 0xc11f399d, 0x378e2c99, 0xbe92b735, 0x3c614021, 0x34c2ba18,
    0xb927b993, 0xb82ab98a, 0xbf68ba9d, 0xb7f9b507, 0xb71840e7, 0x3cd31e42, 0x3a822b48, 0xbf983ef1,
    0x3a56b0ad, 0xbc1234a6, 0x31d63fe4, 0xa3153d63, 0xb81a22f5, 0x32d73b45, 0x36893e26, 0xbc0c3d81,
    0xbcd33d04, 0xb5c836b2, 0xb86b37bb, 0x3c81b658, 0xba20b81b, 0xadad356a, 0x3048be5d, 0x301db9d4,
    0x3d2dadb0, 0xb415aa8b, 0x413fb4b2, 0xb5163c3c, 0x3572aff4, 0xbbf6b682, 0x3bd3bbce, 0x3d0a3a19,
    0x35923729, 0x3a21bc5f, 0xb6673c5b, 0x38c73c4b, 0xaecd3180, 0xb9d9bd82, 0x36d1ac5a, 0x2c34c22c,
    0xb450bcae, 0x388f3f37, 0xb5fdb698, 0x3b4a23d2, 0xae67b721, 0xab023cbc, 0xbd7339d8, 0xbaf198eb,
    0x381e3ad1, 0xbc12b864, 0xc084c0e9, 0x39713831, 0xb2fa3ca9, 0x3708a167, 0xb9ca405f, 0x31ad3ca1,
    0x339d383f, 0x364f389a, 0x3db4b77d, 0xb2bcb56a, 0xb4ddba8f, 0x32b6bf46, 0xb90a381e, 0xbf8a3db5,
    0x37b4bc4e, 0xad30bc0e, 0x3d833c84, 0x3a043a52, 0x4027bc64, 0x3e6d3c18, 0xaf48bcce, 0x325cb052,
    0xbf1db876, 0xaf28bafa, 0x34983a5b, 0xbc54aab7, 0x3d7e3fdb, 0xbf2039e0, 0xaddcbfc8, 0xb83bb815,
    0x3798b793, 0xb6c53915, 0x395bbc75, 0xb960358f, 0xbcedbc4a, 0x34c2bb4c, 0x20223cfd, 0x30cf2cce,
    0xbc4db632, 0x33083a35, 0x38d22f07, 0x3c45bc58, 0xb9adbae1, 0x3910bb37, 0xb91ebf4b, 0xb27e374d,
    0x367bb59a, 0x39fb3af6, 0x3a2abb7c, 0x2d37bf2d, 0x3803b489, 0xbdc63fc6, 0x3fe4381d, 0xb50833f7,
    0xbeb94023, 0xb0c63d4c, 0x38c5b9aa, 0x360ab65d, 0x3ad7bd31, 0x39bf3f3e, 0xb920bc24, 0xb7a33774,
    0xbcb9afd2, 0x387ab627, 0x38cbb7cc, 0xb37eaf26, 0xbad1b2c5, 0xb982b881, 0xbf5fbe79, 0x3cb6b58c,
    0xc084b76a, 0x305f3888, 0xbb1c4386, 0x2990318f, 0xb415bc45, 0x3b9abd45, 0xb6c7400b, 0x3f4940d8,
    0x39533b3e, 0x41a54121, 0x3a863e15, 0xb50cb745, 0xbc62394d, 0x36fd3ce8, 0xb7d3b3fe, 0x3663320d,
    0x39e838fd, 0x38c0b2dd, 0xb5163992, 0xbc1fb019, 0xa8c635e3, 0x39753fda, 0xc0dd3dba, 0x3863166a,
    0x3ab2a3cd, 0xbcb9332f, 0xb88ebce3, 0x41f6bd5e, 0xbd0aba72, 0xc170bac9, 0xb615bc26, 0xbca7bab0,
    0xbb6cc082, 0x415e3f5b, 0xbe9b3be8, 0x3f323c55, 0x410aa521, 0xbe1ebf27, 0x39dfbca8, 0xbbf5b924,
    0x39c7b2aa, 0x424b3af7, 0xaed0b4c6, 0x3532358e, 0x39243116, 0xc199bee5, 0x3849ba6e, 0xb9203c77,
    0xb4813c82, 0x3c4e33de, 0x3775ba98, 0x96a7b594, 0x4000bd85, 0x3fd82fd0, 0x3bb73d71, 0x38083dc5,
    0x34a8bc1c, 0xb86c3d3d, 0x38413bf1, 0xb9d3399c, 0xb752bcee, 0xb00a3ab9, 0x3198b744, 0x398f2fcd,
    0x3d759a32, 0x3bbc360a, 0xbb5937e9, 0xb330b969, 0x3ae6c0a5, 0x40b03e0c, 0x343cbcda, 0xb86c34bc,
    0x35be3cb0, 0xac1bbd82, 0xa5a7bb08, 0xb0f534bf, 0xb4853087, 0xbb9ca123, 0xb45d3df9, 0x404c3c17,
    0x39a3347a, 0xb913b6e8, 0x38cebd20, 0x40b0c006, 0xbb4d402a, 0x3e493e7d, 0xb6c83b8b, 0x3b902d82,
    0xae29388d, 0x3ad7327b, 0xb9843d10, 0x34cc3933, 0xbbda3002, 0x3819bcf3, 0xbc312c89, 0x38e83dbd,
    0xbb87b4e4, 0x3886b172, 0xb82939b9, 0xb345bd54, 0xba7fb951, 0x36c13d9f, 0x3dd3b259, 0x36a43a49,
    0x3cf23b65, 0xbe2cacae, 0x3ba3bdae, 0x3603356a, 0x33703d0e, 0x2d88b96b, 0x36e3b8d3, 0x3b24b804,
    0xbdd0c000, 0xb8c7c006, 0x33673dc9, 0x344bb663, 0x341f3406, 0xc0cfac64, 0x3d0234c6, 0x39f83676,
    0x2e04a4b0, 0x3b5fb4ee, 0x37d4bb6d, 0xb491b0cc, 0xacc8b1aa, 0x3c6cbea5, 0xbc46b99e, 0x3b6cbe17,
    0xbf93bd5d, 0x38fab462, 0x331fc0e4, 0x31643419, 0x4066bb25, 0xbe28bd61, 0xbc663891, 0xbc0c32be,
    0x3dee3055, 0xb3ab20f7, 0x3a5031ff, 0xbad6be6a, 0x3cb5bb8f, 0x3e97be0a, 0xbc8b3110, 0x3c1b36cb,
    0xbe5fbb83, 0x310baecd, 0x2985b4b8, 0xb9d93c4a, 0xbc1d39c5, 0x3d9e390b, 0xbca43675, 0xb49c3ce1,
    0x3fffbb08, 0x35e33927, 0xbc60bd50, 0xbd1339b6, 0xc2a24123, 0x37d9beab, 0x3425b8ed, 0x3a36bace,
    0x4057b0c7, 0x3e0c3c69, 0x3a03b5b0, 0xbaa0ba05, 0x9c4c19c4, 0x3c37b72f, 0xbd023467, 0xb537b522,
    0x2c4d3673, 0xbdefb17b, 0x3daab95a, 0xb4e3bbe8, 0xbb3a3034, 0xbd9cc116, 0xc1d53891, 0xb874bd9f,
    0x357fbc4b, 0x3f79b09a, 0xc07e3f0c, 0xbca2bbfe, 0x2af5bb50, 0xbaf7b02f, 0xb509b629, 0x36fc39c8,
    0x395ca825, 0x3553b901, 0xbbebbd9d, 0xb45139f2, 0x3f3c37bc, 0x386635a3, 0x35fb3046, 0xbcb33999,
    0x3d1da417, 0x3becbc37, 0x3bfa3c7c, 0xc1d5bcb8, 0x3c403c1a, 0xb73bb23c, 0x3a093546, 0xbb7fbbf5,
    0xbe3f30e8, 0x3a8b39fe, 0x9b02341a, 0x2b91bc35, 0x3a514256, 0xba433fd1, 0x393ab55c, 0xae6a3c6f,
    0xb2663da8, 0xba78bd17, 0x3e3a35cf, 0x3e4db5dd, 0x32ff345d, 0x35a2bca2, 0x3c82b9f1, 0x38f1bbc5,
    0xbe7b3f36, 0xb44abe77, 0x3501bd3b, 0xbe42344c, 0x3dc3bc81, 0xbf152613, 0x3c71bb1c, 0xba6a345f,
    0x3c03b944, 0xbae1b5ea, 0x3e77ab67, 0xb9dbc04f, 0xb86ebe0c, 0xb9cdbc4f, 0xb07a3ca8, 0xb41f36be,
    0x3256ac27, 0xb3273870, 0x3b9db5f6, 0xb8de30ba, 0xb06a3c1b, 0x3d543ae1, 0x3d08ba28, 0xb93838a3,
    0xb96b3472, 0x3b803945, 0x3ab1adbe, 0x39b53ccf, 0x383634a8, 0x308cb49e, 0xb5ba3916, 0x3b1bbb8a,
    0xbe6fbc56, 0xb4d5bd9e, 0xa3433e83, 0xb2fc3adc, 0x3b55ba2f, 0xb6243cec, 0x37d7261f, 0x3eb1bf76,
    0xb9c838b1, 0x361ca60d, 0xc00cbca6, 0xbc6bbb55, 0x38ba34c5, 0x383cb802, 0x3e68bd59, 0xbbad3a2e,
    0x3b243c74, 0x3ebf3010, 0xbfa8b953, 0x3b7cb16b, 0x2c823952, 0x412cb8ef, 0x3cdbb7ec, 0x38e03bc9,
    0xbaa3bbfb, 0xb1df3f28, 0xbccf3a77, 0xb933b1a5, 0x37cb3018, 0x31bcbec0, 0x3997bd2e, 0x39063930,
    0x37fdb86f, 0x33e5323d, 0x38d7a60a, 0xb451b30a, 0x31233597, 0xbe8ac063, 0x37afbe42, 0x3b21aff0,
    0xb277365f, 0x2ad2b5b8, 0x3a45c11d, 0xb50031fd, 0xbc72bf3f, 0x24a33038, 0xb5952890, 0xc1c7c08f,
    0xbcaab86e, 0x38a63189, 0x325db408, 0x3b052ca7, 0xbf353c55, 0x3b7e387c, 0x3a0b3c20, 0x3db83741,
    0x3c4d3a12, 0x3d11b87c, 0xbed7bdc9, 0xbdbb2444, 0x3cc9b9c6, 0xba10ae72, 0x395338b7, 0xbbe2b1e6,
    0x30b1a918, 0x283dbc8f, 0xb160aeb3, 0x33363794, 0x401d3327, 0x3d15bf99, 0x3a32ad02, 0x3c573df6,
    0x34edbef4, 0x368cbc1f, 0xb95b3aed, 0x3bf338dd, 0x3c2f4185, 0x37b5bbfc, 0x355e37f3, 0x3c463559,
    0x38bcc003, 0xbda03b53, 0x2acbac8d, 0x396cb8ac, 0xbb1f2be0, 0x3b903ce8, 0x3ef8ae9f, 0x3cf4b33e,
    0xb8982c97, 0x3f56baa5, 0xbb163e58, 0xc004b33c, 0xbc68400e, 0xa93938bd, 0x3b353a55, 0x3db540d6,
    0xb8694176, 0xbd7f30ba, 0xb379ba99, 0xb9153ecf, 0x30f2b19f, 0xb6323e11, 0x3be5be94, 0xac5d3d28,
    0xbb64b6e7, 0x4380b282, 0x36b5bde1, 0x386736c2, 0x2e1c3cca, 0x2e2f31cc, 0xbcffb0d4, 0xbd67bb1c,
    0x3d2c309a, 0xbd03ab7d, 0xbefcbbe1, 0x3b98bfbe, 0xbc72b53c, 0x3cca2f15, 0xbc36c106, 0x3b57b702,
    0xb4da391a, 0x3afe3861, 0x335b3ea8, 0xbd6234e4, 0xbc863f1f, 0x35bcbcf4, 0xbf53b34a, 0xbd103834,
    0x39a1273d, 0xb4e4b839, 0x34da39b8, 0x3a76bed6, 0xc0fa3c50, 0xb061bd35, 0x34f0b1b6, 0xbc183ab1,
    0xb62cb91d, 0xb829bb36, 0xafabb940, 0x3e813d46, 0xb822b20b, 0x35fdbe39, 0x396632f1, 0xb5432e17,
    0x3a89bbed, 0xbcc5b2a3, 0xb42b3369, 0x1f58bbd4, 0x3ef4bc48, 0xac2b3bbe, 0xb2e4bdf5, 0x3f94bcae,
    0xc068a11a, 0xb82e35bb, 0xbbb93f77, 0x3b46b386, 0x22f3b675, 0xbcb23e8b, 0xba29b8ba, 0xbd084235,
    0x3f9f1038, 0x399fade8, 0xb9da3363, 0x37c83909, 0xbc4c3c82, 0xb788bc3a, 0x39daa897, 0x3beeb6d2,
};
// END tq_lut
// the trellis codebook point i (two fp16 values in one word): from the constant copy, or the threadgroup one
static inline uint g_tq(threadgroup const uint* tab, uint i) { return ctab ? tq_lut_u[i] : tab[i]; }

template <> inline void load_tab<T_Q2_0>(threadgroup uint*, ushort, ushort) {}
template <int F> static inline void load_tab_tq(threadgroup uint* tab, ushort tid, ushort nth) {
    if (ctab) return;
    for (int i = tid; i < 2048; i += nth) tab[i] = tq_lut_u[i];
}
template <> inline void load_tab<T_TQ2T>(threadgroup uint* tab, ushort tid, ushort nth) { load_tab_tq<T_TQ2T>(tab, tid, nth); }
template <> inline void load_tab<T_TQK6>(threadgroup uint* tab, ushort tid, ushort nth) { load_tab_tq<T_TQK6>(tab, tid, nth); }
template <> inline void load_tab<T_TQK7>(threadgroup uint* tab, ushort tid, ushort nth) { load_tab_tq<T_TQK7>(tab, tid, nth); }
template <> inline void load_tab<T_Q2_0R>(threadgroup uint*, ushort, ushort) {}
template <> inline void load_tab<T_Q8_0>(threadgroup uint*, ushort, ushort) {}
template <> inline void load_tab<T_Q4_0>(threadgroup uint*, ushort, ushort) {}
template <> inline void load_tab<T_Q5_0>(threadgroup uint*, ushort, ushort) {}
template <> inline void load_tab<T_Q5_1>(threadgroup uint*, ushort, ushort) {}
template <> inline void load_tab<T_Q3_K>(threadgroup uint*, ushort, ushort) {}
template <> inline void load_tab<T_Q4_K>(threadgroup uint*, ushort, ushort) {}
template <> inline void load_tab<T_Q5_K>(threadgroup uint*, ushort, ushort) {}
template <> inline void load_tab<T_Q6_K>(threadgroup uint*, ushort, ushort) {}
template <> inline void load_tab<T_BF16>(threadgroup uint*, ushort, ushort) {}
template <> inline void load_tab<T_F32>(threadgroup uint*, ushort, ushort) {}

static inline void load_signs(threadgroup uint* tab, int at, ushort tid, ushort nth) {
    threadgroup uchar* s = (threadgroup uchar*)(tab + at);
    for (int i = tid; i < 128; i += nth) s[i] = ksigns_iq2xs[i];
}
template <> inline void load_tab<T_IQ3_XXS>(threadgroup uint* tab, ushort tid, ushort nth) {
    if (ctab) return;
    for (int i = tid; i < 256; i += nth) tab[i] = iq3xxs_grid[i];
    load_signs(tab, 256, tid, nth);
}
template <> inline void load_tab<T_IQ3_S>(threadgroup uint* tab, ushort tid, ushort nth) {
    if (ctab) return;
    for (int i = tid; i < 512; i += nth) tab[i] = iq3s_grid[i];
    load_signs(tab, 512, tid, nth);
}
template <> inline void load_tab<T_IQ2_XXS>(threadgroup uint* tab, ushort tid, ushort nth) {
    if (ctab) return;
    for (int i = tid; i < 256; i += nth) { tab[2 * i] = uint(iq2xxs_grid[i]); tab[2 * i + 1] = uint(iq2xxs_grid[i] >> 32); }
    load_signs(tab, 512, tid, nth);
}
template <> inline void load_tab<T_IQ2_XS>(threadgroup uint* tab, ushort tid, ushort nth) {
    if (ctab) return;
    for (int i = tid; i < 512; i += nth) { tab[2 * i] = uint(iq2xs_grid[i]); tab[2 * i + 1] = uint(iq2xs_grid[i] >> 32); }
    load_signs(tab, 1024, tid, nth);
}
// IQ1_M: no threadgroup table - the tile reads iq1s_grid from the constants (the w4 helper in moe_mm.metal)
template <> inline void load_tab<T_IQ1_M>(threadgroup uint*, ushort, ushort) {}
template <> inline void load_tab<T_IQ1_S>(threadgroup uint*, ushort, ushort) {}   // (same: iq1s_grid from the constants)
template <> inline void load_tab<T_IQ2_S>(threadgroup uint* tab, ushort tid, ushort nth) {
    if (ctab) return;
    for (int i = tid; i < 1024; i += nth) { tab[2 * i] = uint(iq2s_grid[i]); tab[2 * i + 1] = uint(iq2s_grid[i] >> 32); }
    load_signs(tab, 2048, tid, nth);
}
// IQ4: a byte -> (codebook[low nibble], codebook[high nibble]) as half2, so one lookup gives two weights
template <> inline void load_tab<T_IQ4_NL>(threadgroup uint* tab, ushort tid, ushort nth) {
    for (int i = tid; i < 256; i += nth)
        tab[i] = as_type<uint>(half2(half(kvalues_iq4nl[i & 15]), half(kvalues_iq4nl[i >> 4])));
}
template <> inline void load_tab<T_IQ4_XS>(threadgroup uint* tab, ushort tid, ushort nth) {
    load_tab<T_IQ4_NL>(tab, tid, nth);
}

// ---- unit decoders: return s, fill q[UNIT] ------------------------------------------------------------------------
// ui = unit index inside the block (0 .. QK/UNIT - 1).
template <int F> static inline float deq(device const uchar* b, int ui, thread float* q, threadgroup const uint* tab);

// Q2_0: 64 = 4 units of 16; code c (2 bits, LSB first) -> c - 1; scale d.
template <> inline float deq<T_Q2_0>(device const uchar* b, int ui, thread float* q, threadgroup const uint*) {
    device const ushort* p = (device const ushort*)(b + 2 + 4 * ui);
    const ushort a = p[0], c = p[1];
    for (int j = 0; j < 8; ++j) {
        q[j] = float((a >> (2 * j)) & 3) - 1.0f;
        q[8 + j] = float((c >> (2 * j)) & 3) - 1.0f;
    }
    return rdh(b);
}

// 8 signed grid bytes from a 4-byte grid word pair helper
static inline void put_signed4(thread float* q, uint g, uint s) {
    const uchar4 v = as_type<uchar4>(g);
    q[0] = (s & 1) ? -float(v[0]) : float(v[0]);
    q[1] = (s & 2) ? -float(v[1]) : float(v[1]);
    q[2] = (s & 4) ? -float(v[2]) : float(v[2]);
    q[3] = (s & 8) ? -float(v[3]) : float(v[3]);
}

// IQ3_XXS: 256 = 8 x 32; unit = half of a 32 (two grid pairs).
template <> inline float deq<T_IQ3_XXS>(device const uchar* b, int ui, thread float* q, threadgroup const uint* tab) {
    const int ib = ui >> 1, h = ui & 1;
    const uint aux = rd32(b + 2 + 64 + 4 * ib);
    device const uchar* qs = b + 2 + 8 * ib + 4 * h;
    for (int l2 = 0; l2 < 2; ++l2) {
        const int l = 2 * h + l2;
        const uint s = ksg(tab, 256, (aux >> (7 * l)) & 127);
        put_signed4(q + 8 * l2, g_iq3xxs(tab, qs[2 * l2]), s);
        put_signed4(q + 8 * l2 + 4, g_iq3xxs(tab, qs[2 * l2 + 1]), s >> 4);
    }
    return rdh(b) * (0.5f + float(aux >> 28)) * 0.5f;
}

// IQ2_XXS: per 32: two u32 (4 grid indices, then 4 x 7 sign bits + 4-bit scale).
template <> inline float deq<T_IQ2_XXS>(device const uchar* b, int ui, thread float* q, threadgroup const uint* tab) {
    const int ib = ui >> 1, h = ui & 1;
    const uint a0 = rd32(b + 2 + 8 * ib), a1 = rd32(b + 2 + 8 * ib + 4);
    for (int l2 = 0; l2 < 2; ++l2) {
        const int l = 2 * h + l2;
        const uint idx = (a0 >> (8 * l)) & 255;
        const uint s = ksg(tab, 512, (a1 >> (7 * l)) & 127);
        put_signed4(q + 8 * l2, g_iq2xxs(tab, idx, 0), s);
        put_signed4(q + 8 * l2 + 4, g_iq2xxs(tab, idx, 1), s >> 4);
    }
    return rdh(b) * (0.5f + float(a1 >> 28)) * 0.25f;
}

// IQ2_XS: uint16 qs[32] (9-bit grid index + 7-bit sign index), scales[8] nibbles.
template <> inline float deq<T_IQ2_XS>(device const uchar* b, int ui, thread float* q, threadgroup const uint* tab) {
    const int ib = ui >> 1, h = ui & 1;
    device const ushort* qs = (device const ushort*)(b + 2);
    for (int l2 = 0; l2 < 2; ++l2) {
        const uint v = qs[4 * ib + 2 * h + l2];
        const uint idx = v & 511, s = ksg(tab, 1024, v >> 9);
        put_signed4(q + 8 * l2, g_iq2xs(tab, idx, 0), s);
        put_signed4(q + 8 * l2 + 4, g_iq2xs(tab, idx, 1), s >> 4);
    }
    const uint sc = b[66 + ib];
    return rdh(b) * (0.5f + float((sc >> (4 * h)) & 15)) * 0.25f;
}

// IQ2_S: qs[32] low 8 bits of the 10-bit index, signs[32], qh[8] (2 high bits per group of 8), scales[8].
template <> inline float deq<T_IQ2_S>(device const uchar* b, int ui, thread float* q, threadgroup const uint* tab) {
    const int ib = ui >> 1, h = ui & 1;
    const uint qh = b[66 + ib];
    for (int l2 = 0; l2 < 2; ++l2) {
        const int l = 2 * h + l2;
        const uint idx = uint(b[2 + 4 * ib + l]) | ((qh << (8 - 2 * l)) & 0x300);
        const uint s = b[2 + 32 + 4 * ib + l];
        put_signed4(q + 8 * l2, g_iq2s(tab, idx, 0), s);
        put_signed4(q + 8 * l2 + 4, g_iq2s(tab, idx, 1), s >> 4);
    }
    const uint sc = b[74 + ib];
    return rdh(b) * (0.5f + float((sc >> (4 * h)) & 15)) * 0.25f;
}

// IQ3_S: qs[64] low 8 bits of 9-bit indices, qh[8], signs[32], scales[4] (nibble per 32).
template <> inline float deq<T_IQ3_S>(device const uchar* b, int ui, thread float* q, threadgroup const uint* tab) {
    const int ib = ui >> 1, h = ui & 1;
    const uint qh = b[66 + ib];
    for (int l2 = 0; l2 < 2; ++l2) {
        const int l = 2 * h + l2;
        const uint i1 = uint(b[2 + 8 * ib + 2 * l]) | ((qh << (8 - 2 * l)) & 256);
        const uint i2 = uint(b[2 + 8 * ib + 2 * l + 1]) | ((qh << (7 - 2 * l)) & 256);
        const uint s = b[74 + 4 * ib + l];
        put_signed4(q + 8 * l2, g_iq3s(tab, i1), s);
        put_signed4(q + 8 * l2 + 4, g_iq3s(tab, i2), s >> 4);
    }
    const uint sc = b[106 + (ib >> 1)];
    return rdh(b) * float(1 + 2 * ((sc >> (4 * (ib & 1))) & 15));
}

// IQ4_NL: 32 = 1 unit; low nibbles are elements 0..15, high 16..31 (one half2 lookup per byte).
template <> inline float deq<T_IQ4_NL>(device const uchar* b, int, thread float* q, threadgroup const uint* tab) {
    device const ushort* p = (device const ushort*)(b + 2);
    for (int j = 0; j < 8; ++j) {
        const ushort v = p[j];
        const half2 w0 = as_type<half2>(tab[v & 255]), w1 = as_type<half2>(tab[v >> 8]);
        q[2 * j] = float(w0.x);
        q[2 * j + 16] = float(w0.y);
        q[2 * j + 1] = float(w1.x);
        q[2 * j + 17] = float(w1.y);
    }
    return rdh(b);
}

// IQ4_XS: 256 = 8 units of 32; 6-bit scale from scales_l nibble + 2 bits of scales_h, minus 32.
template <> inline float deq<T_IQ4_XS>(device const uchar* b, int ui, thread float* q, threadgroup const uint* tab) {
    device const ushort* p = (device const ushort*)(b + 8 + 16 * ui);
    for (int j = 0; j < 8; ++j) {
        const ushort v = p[j];
        const half2 w0 = as_type<half2>(tab[v & 255]), w1 = as_type<half2>(tab[v >> 8]);
        q[2 * j] = float(w0.x);
        q[2 * j + 16] = float(w0.y);
        q[2 * j + 1] = float(w1.x);
        q[2 * j + 17] = float(w1.y);
    }
    const uint sh = *(device const ushort*)(b + 2);
    const int ls = int((b[4 + (ui >> 1)] >> (4 * (ui & 1))) & 15) | int(((sh >> (2 * ui)) & 3) << 4);
    return rdh(b) * float(ls - 32);
}

template <> inline float deq<T_Q8_0>(device const uchar* b, int, thread float* q, threadgroup const uint*) {
    device const ushort* p = (device const ushort*)(b + 2);
    for (int j = 0; j < 16; ++j) {
        const ushort v = p[j];
        q[2 * j] = float(as_type<char>(uchar(v & 255)));
        q[2 * j + 1] = float(as_type<char>(uchar(v >> 8)));
    }
    return rdh(b);
}

template <> inline float deq<T_Q4_0>(device const uchar* b, int, thread float* q, threadgroup const uint*) {
    device const ushort* p = (device const ushort*)(b + 2);
    for (int j = 0; j < 8; ++j) {
        const ushort v = p[j];
        q[2 * j] = float(v & 15) - 8.0f;
        q[2 * j + 16] = float((v >> 4) & 15) - 8.0f;
        q[2 * j + 1] = float((v >> 8) & 15) - 8.0f;
        q[2 * j + 17] = float(v >> 12) - 8.0f;
    }
    return rdh(b);
}

template <> inline float deq<T_Q5_0>(device const uchar* b, int, thread float* q, threadgroup const uint*) {
    const uint hb = rd32(b + 2);
    device const ushort* p = (device const ushort*)(b + 6);
    for (int j = 0; j < 8; ++j) {
        const ushort v = p[j];
        const int j0 = 2 * j, j1 = 2 * j + 1;
        q[j0] = float(int(v & 15) | int(((hb >> j0) & 1) << 4)) - 16.0f;
        q[j0 + 16] = float(int((v >> 4) & 15) | int(((hb >> (j0 + 16)) & 1) << 4)) - 16.0f;
        q[j1] = float(int((v >> 8) & 15) | int(((hb >> j1) & 1) << 4)) - 16.0f;
        q[j1 + 16] = float(int(v >> 12) | int(((hb >> (j1 + 16)) & 1) << 4)) - 16.0f;
    }
    return rdh(b);
}

// Q5_1: 32 = 1 unit; q = the 5-bit code (no offset; the min m is applied through the x sums), returns d
template <> inline float deq<T_Q5_1>(device const uchar* b, int, thread float* q, threadgroup const uint*) {
    const uint hb = rd32(b + 4);
    device const ushort* p = (device const ushort*)(b + 8);
    for (int j = 0; j < 8; ++j) {
        const ushort v = p[j];
        const int j0 = 2 * j, j1 = 2 * j + 1;
        q[j0] = float(int(v & 15) | int(((hb >> j0) & 1) << 4));
        q[j0 + 16] = float(int((v >> 4) & 15) | int(((hb >> (j0 + 16)) & 1) << 4));
        q[j1] = float(int((v >> 8) & 15) | int(((hb >> j1) & 1) << 4));
        q[j1 + 16] = float(int(v >> 12) | int(((hb >> (j1 + 16)) & 1) << 4));
    }
    return rdh(b);
}

// Trellis types (ggml-common-tq.h): a 128-value block is {fp16 d; qs}, 32 steps of K bits (K = 8 / 6 / 7).  A unit is
// lane ui = steps 4ui..4ui+3, 16 weights.  Step t's 15-bit state: TQ2_T qs[(t + 31) % 32] << 8 | qs[t]; TQK the 16
// stream bits from bit (31 - t) * K, circular over 32 K bits.  x = state * 0x9e3779b1 selects two codebook points,
// lut[x >> 21] and lut[(x >> 10) & 2047] (fp16 pairs); the four weights of the step are those four values times d.
// Same arithmetic as src/kernels/cuda/iq_kernels.cu (tq_states4 / tq_step) and ref_quant.cpp (deq_tq).
template <int KB>
static inline float deq_tq(device const uchar* b, int ui, thread float* q, threadgroup const uint* tab) {
    device const ushort* qs16 = (device const ushort*)(b + 2);
    uint s[4];
    if (KB == 8) {
        const uint w01 = qs16[2 * ui], w23 = qs16[2 * ui + 1];
        const uint prev = b[2 + ((4 * ui + 31) & 31)];
        const uint b0 = w01 & 0xFFu, b1 = w01 >> 8, b2 = w23 & 0xFFu, b3 = w23 >> 8;
        s[0] = (prev << 8) | b0;
        s[1] = (b0 << 8) | b1;
        s[2] = (b1 << 8) | b2;
        s[3] = (b2 << 8) | b3;
    } else {
        constexpr int NW = 2 * KB;
        const int off3 = (28 - 4 * ui) * KB, wb = off3 >> 4, r = off3 & 15;
        const int wi1 = wb + 1 >= NW ? wb + 1 - NW : wb + 1;
        const int wi2 = wb + 2 >= NW ? wb + 2 - NW : wb + 2;
        const int wi3 = wb + 3 >= NW ? wb + 3 - NW : wb + 3;
        const ulong v = ulong(qs16[wb]) | (ulong(qs16[wi1]) << 16) | (ulong(qs16[wi2]) << 32) | (ulong(qs16[wi3]) << 48);
        for (int i = 0; i < 4; ++i) s[i] = uint(v >> (r + (3 - i) * KB)) & 0xFFFFu;
    }
    for (int i = 0; i < 4; ++i) {
        const uint x = (s[i] & 0x7FFFu) * 0x9e3779b1u;
        const float2 a = float2(as_type<half2>(g_tq(tab, x >> 21)));
        const float2 c = float2(as_type<half2>(g_tq(tab, (x >> 10) & 2047u)));
        q[4 * i] = a.x;
        q[4 * i + 1] = a.y;
        q[4 * i + 2] = c.x;
        q[4 * i + 3] = c.y;
    }
    return rdh(b);
}
template <> inline float deq<T_TQ2T>(device const uchar* b, int ui, thread float* q, threadgroup const uint* tab) { return deq_tq<8>(b, ui, q, tab); }
template <> inline float deq<T_TQK6>(device const uchar* b, int ui, thread float* q, threadgroup const uint* tab) { return deq_tq<6>(b, ui, q, tab); }
template <> inline float deq<T_TQK7>(device const uchar* b, int ui, thread float* q, threadgroup const uint* tab) { return deq_tq<7>(b, ui, q, tab); }

// sum over groups of LPR consecutive lanes (LPR = 1, 2, 4, 8, 16 or 32); the result is valid in every lane
template <int LPR>
static inline float lane_sum(float v) {
    if (LPR == 32) return simd_sum(v);
    for (ushort o = LPR / 2; o > 0; o >>= 1) v += simd_shuffle_xor(v, o);
    return v;
}

// ---- Gyro: the block Hadamard rotation of the activations ------------------------------------------------------------
// Hadamard-folded experts (docs/GYRO.md, include/strata/kernels/cpu/hadamard.hpp) take y = H (s * x) per block of 128
// values (the normalized Sylvester-Walsh-Hadamard matrix, s = a +/-1 sign vector of the width) in place of x: the gate/up
// input and the down input.  One thread per value, 128 threads a block, grid (width / 128, rows).  The butterflies run in
// the CPU's order (len = 1, 2, 4, ...; lower element a + c, upper a - c), so the rotated floats are the CPU's and the
// CUDA kernel's: len 1..16 by SIMD shuffles, 32 and 64 through threadgroup memory.
struct HadArgs {
    uint width;      // values per row (a multiple of 128)
    uint x_stride;   // row strides in elements
    uint y_stride;
    float scale;     // 1 / sqrt(128), as the CPU computes it
    uint has_signs;
    uint pad;
};
template <typename T>
static inline float had_ld(device const T* p);
template <> inline float had_ld<float>(device const float* p) { return *p; }
template <> inline float had_ld<half>(device const half* p) { return float(*p); }
template <typename T>
static inline void had_body(device const T* x, device T* y, device const float* signs, constant HadArgs& a, uint3 tg,
                            ushort tid, threadgroup float* buf) {
    const uint col = tg.x * 128u + tid;
    float v = had_ld<T>(x + ulong(tg.y) * a.x_stride + col);
    if (a.has_signs) v *= signs[col];
    for (ushort len = 1; len < 32; len <<= 1) {
        const float o = simd_shuffle_xor(v, len);
        v = (tid & len) ? o - v : v + o;
    }
    buf[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float o = buf[tid ^ 32];
    v = (tid & 32) ? o - v : v + o;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o = buf[tid ^ 64];
    v = (tid & 64) ? o - v : v + o;
    y[ulong(tg.y) * a.y_stride + col] = T(v * a.scale);
}
kernel void had128_f32(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                       device const float* signs [[buffer(2)]], constant HadArgs& a [[buffer(3)]],
                       uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float buf[128];
    had_body<float>(x, y, signs, a, tg, tid, buf);
}
kernel void had128_f16(device const half* x [[buffer(0)]], device half* y [[buffer(1)]],
                       device const float* signs [[buffer(2)]], constant HadArgs& a [[buffer(3)]],
                       uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float buf[128];
    had_body<half>(x, y, signs, a, tg, tid, buf);
}
// half rows in place, one threadgroup a row (grid.x = the row, so an indirect grid of the row count works): the prompt
// path's h rows (the padded rows of the expert tiles), width values each
kernel void had128_f16_rows(device half* x [[buffer(0)]], device half* y [[buffer(1)]],
                            device const float* signs [[buffer(2)]], constant HadArgs& a [[buffer(3)]],
                            uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float buf[128];
    for (uint b = 0; b < a.width / 128u; ++b) {
        had_body<half>(x, y, signs, a, uint3(b, tg.x, 0), tid, buf);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
