// metal/include/metal_compat/vector_types.h - the CUDA vector types the dsv41 sources name, on Metal.
//
// The compat directory had none, so the first half-precision header to use float2 failed.  Only the shapes the engine
// touches are here (float2/float4/int2/uint2 and their make_ helpers); the Metal runtime's own float4 is not involved.
#pragma once

struct float2 { float x, y; };
struct float4 { float x, y, z, w; };
struct int2 { int x, y; };
struct uint2 { unsigned x, y; };

static inline float2 make_float2(float x, float y) { return float2{x, y}; }
static inline float4 make_float4(float x, float y, float z, float w) { return float4{x, y, z, w}; }
static inline int2 make_int2(int x, int y) { return int2{x, y}; }
static inline uint2 make_uint2(unsigned x, unsigned y) { return uint2{x, y}; }

struct int4 { int x, y, z, w; };
struct uint4 { unsigned x, y, z, w; };
struct char4 { signed char x, y, z, w; };
struct uchar4 { unsigned char x, y, z, w; };
struct short2 { short x, y; };
struct int3 { int x, y, z; };

static inline int4 make_int4(int x, int y, int z, int w) { return int4{x, y, z, w}; }
static inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w) { return uint4{x, y, z, w}; }
