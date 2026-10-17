/* SPDX-License-Identifier: Apache-2.0 */
/*
 * k3_metal.m - Apple Silicon Metal GPU compute engine for Kimi K3.
 *
 * Implements GPU-accelerated matrix multiplication for:
 *   - BF16 trunk matmuls (k3_matmul_bf16)
 *   - MXFP4 expert matmuls (k3_matmul_mxfp4)
 *   - FP32 matmuls (k3_matmul)
 */

#if defined(__APPLE__)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <os/lock.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "k3_metal.h"

static const char *K3_METAL_SHADERS = 
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"// ---------------------------------------------------------------------------\n"
"// BF16 Matrix-Vector Multiplication: y[out] = W[out][in] * x[in]\n"
"// Coalesced 128-bit reads using uint4 (8 BF16s per vector)\n"
"// ---------------------------------------------------------------------------\n"
"kernel void k3_metal_kernel_matmul_bf16(\n"
"    device float *y [[buffer(0)]],\n"
"    device const float *x [[buffer(1)]],\n"
"    device const ushort *W [[buffer(2)]],\n"
"    constant int &in_dim [[buffer(3)]],\n"
"    constant int &out_dim [[buffer(4)]],\n"
"    uint2 tg_id [[threadgroup_position_in_grid]],\n"
"    uint simd_id [[simdgroup_index_in_threadgroup]],\n"
"    uint lane_id [[thread_index_in_simdgroup]])\n"
"{\n"
"    uint row_idx = tg_id.x * 4 + simd_id;\n"
"    if (row_idx >= (uint)out_dim) return;\n"
"\n"
"    device const uint4 *row_u4 = (device const uint4 *)(W + (ulong)row_idx * in_dim);\n"
"    device const float4 *x4 = (device const float4 *)x;\n"
"\n"
"    float my_acc = 0.0f;\n"
"    int num_u4 = in_dim / 8;\n"
"\n"
"    for (int v = lane_id; v < num_u4; v += 32) {\n"
"        uint4 u = row_u4[v];\n"
"        float4 x_lo = x4[v * 2];\n"
"        float4 x_hi = x4[v * 2 + 1];\n"
"\n"
"        float w0 = as_type<float>((u.x & 0xFFFFu) << 16);\n"
"        float w1 = as_type<float>(u.x & 0xFFFF0000u);\n"
"        float w2 = as_type<float>((u.y & 0xFFFFu) << 16);\n"
"        float w3 = as_type<float>(u.y & 0xFFFF0000u);\n"
"        my_acc += w0 * x_lo.x + w1 * x_lo.y + w2 * x_lo.z + w3 * x_lo.w;\n"
"\n"
"        float w4 = as_type<float>((u.z & 0xFFFFu) << 16);\n"
"        float w5 = as_type<float>(u.z & 0xFFFF0000u);\n"
"        float w6 = as_type<float>((u.w & 0xFFFFu) << 16);\n"
"        float w7 = as_type<float>(u.w & 0xFFFF0000u);\n"
"        my_acc += w4 * x_hi.x + w5 * x_hi.y + w6 * x_hi.z + w7 * x_hi.w;\n"
"    }\n"
"\n"
"    int tail_start = num_u4 * 8 + lane_id;\n"
"    if (tail_start < in_dim) {\n"
"        device const ushort *row = W + (ulong)row_idx * in_dim;\n"
"        uint u = (uint)row[tail_start] << 16;\n"
"        my_acc += as_type<float>(u) * x[tail_start];\n"
"    }\n"
"\n"
"    float row_sum = simd_sum(my_acc);\n"
"    if (lane_id == 0) {\n"
"        y[row_idx] = row_sum;\n"
"    }\n"
"}\n"
"\n"
"// ---------------------------------------------------------------------------\n"
"// FP32 Matrix-Vector Multiplication: y[out] = W[out][in] * x[in]\n"
"// Coalesced 128-bit reads using float4\n"
"// ---------------------------------------------------------------------------\n"
"kernel void k3_metal_kernel_matmul_f32(\n"
"    device float *y [[buffer(0)]],\n"
"    device const float *x [[buffer(1)]],\n"
"    device const float *W [[buffer(2)]],\n"
"    constant int &in_dim [[buffer(3)]],\n"
"    constant int &out_dim [[buffer(4)]],\n"
"    uint2 tg_id [[threadgroup_position_in_grid]],\n"
"    uint simd_id [[simdgroup_index_in_threadgroup]],\n"
"    uint lane_id [[thread_index_in_simdgroup]])\n"
"{\n"
"    uint row_idx = tg_id.x * 4 + simd_id;\n"
"    if (row_idx >= (uint)out_dim) return;\n"
"\n"
"    device const float4 *row4 = (device const float4 *)(W + (ulong)row_idx * in_dim);\n"
"    device const float4 *x4   = (device const float4 *)x;\n"
"\n"
"    float my_acc = 0.0f;\n"
"    int num_f4 = in_dim / 4;\n"
"\n"
"    for (int v = lane_id; v < num_f4; v += 32) {\n"
"        my_acc += dot(row4[v], x4[v]);\n"
"    }\n"
"\n"
"    int tail_start = num_f4 * 4 + lane_id;\n"
"    if (tail_start < in_dim) {\n"
"        device const float *row = W + (ulong)row_idx * in_dim;\n"
"        my_acc += row[tail_start] * x[tail_start];\n"
"    }\n"
"\n"
"    float row_sum = simd_sum(my_acc);\n"
"    if (lane_id == 0) {\n"
"        y[row_idx] = row_sum;\n"
"    }\n"
"}\n"
"\n"
"// ---------------------------------------------------------------------------\n"
"// MXFP4 Matrix-Vector Multiplication: y[rows] = W[rows][in] * x[in]\n"
"// ---------------------------------------------------------------------------\n"
"constant float K3_LUT_E2M1[16] = {\n"
"    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,\n"
"   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f\n"
"};\n"
"\n"
"inline float k3_e8m0_scale(uchar sb) {\n"
"    if (sb == 255 || sb == 0) return 0.0f;\n"
"    return as_type<float>((uint)sb << 23);\n"
"}\n"
"\n"
"kernel void k3_metal_kernel_matmul_mxfp4(\n"
"    device float *y [[buffer(0)]],\n"
"    device const float *x [[buffer(1)]],\n"
"    device const uchar *packed [[buffer(2)]],\n"
"    device const uchar *scales [[buffer(3)]],\n"
"    constant int &in_dim [[buffer(4)]],\n"
"    constant int &rows_dim [[buffer(5)]],\n"
"    constant int &group [[buffer(6)]],\n"
"    uint2 tg_id [[threadgroup_position_in_grid]],\n"
"    uint simd_id [[simdgroup_index_in_threadgroup]],\n"
"    uint lane_id [[thread_index_in_simdgroup]])\n"
"{\n"
"    uint row_idx = tg_id.x * 4 + simd_id;\n"
"    if (row_idx >= (uint)rows_dim) return;\n"
"\n"
"    int pcols = in_dim / 2;\n"
"    int ngrp = in_dim / group;\n"
"    device const uchar *pr = packed + (ulong)row_idx * pcols;\n"
"    device const uchar *sr = scales + (ulong)row_idx * ngrp;\n"
"\n"
"    float my_acc = 0.0f;\n"
"\n"
"    if (group == 32) {\n"
"        device const uint4 *row_pk4 = (device const uint4 *)pr;\n"
"        for (int g = lane_id; g < ngrp; g += 32) {\n"
"            uchar sb = sr[g];\n"
"            if (sb == 255 || sb == 0) continue;\n"
"            float sc = as_type<float>((uint)sb << 23);\n"
"            uint4 p4 = row_pk4[g];\n"
"            int xb0 = g * 32;\n"
"            float group_acc = 0.0f;\n"
"\n"
"            for (int w_idx = 0; w_idx < 4; w_idx++) {\n"
"                uint u = (w_idx == 0) ? p4.x : ((w_idx == 1) ? p4.y : ((w_idx == 2) ? p4.z : p4.w));\n"
"                int xb = xb0 + w_idx * 8;\n"
"                for (int k = 0; k < 4; k++) {\n"
"                    uchar b = (u >> (k * 8)) & 0xFF;\n"
"                    group_acc += K3_LUT_E2M1[b & 0x0F] * x[xb + k * 2] + K3_LUT_E2M1[b >> 4] * x[xb + k * 2 + 1];\n"
"                }\n"
"            }\n"
"            my_acc += group_acc * sc;\n"
"        }\n"
"    } else {\n"
"        int byte_off = lane_id >> 1;\n"
"        int is_hi = lane_id & 1;\n"
"        for (int g = 0; g < ngrp; g++) {\n"
"            uchar sb = sr[g];\n"
"            if (sb == 255 || sb == 0) continue;\n"
"            float sc = as_type<float>((uint)sb << 23);\n"
"            int g_start = g * group;\n"
"            device const uchar *pb = pr + (ulong)(g_start / 2);\n"
"            for (int e = lane_id; e < group; e += 32) {\n"
"                if (g_start + e < in_dim) {\n"
"                    uchar byte_val = pb[e >> 1];\n"
"                    uchar nib = (e & 1) ? (byte_val >> 4) : (byte_val & 0x0F);\n"
"                    float weight = K3_LUT_E2M1[nib] * sc;\n"
"                    my_acc += weight * x[g_start + e];\n"
"                }\n"
"            }\n"
"        }\n"
"    }\n"
"\n"
"    float row_sum = simd_sum(my_acc);\n"
"    if (lane_id == 0) {\n"
"        y[row_idx] = row_sum;\n"
"    }\n"
"}\n";

static id<MTLDevice> s_device = nil;
static id<MTLCommandQueue> s_queue = nil;
static id<MTLComputePipelineState> s_pipe_bf16 = nil;
static id<MTLComputePipelineState> s_pipe_f32 = nil;
static id<MTLComputePipelineState> s_pipe_mxfp4 = nil;
static char s_device_name[128] = "None";
static int s_initialized = 0;
static int s_enabled = 1;
static size_t s_page_size = 0;

int k3_metal_init(void)
{
    if (s_initialized) return 1;

    @autoreleasepool {
        s_device = MTLCreateSystemDefaultDevice();
        if (!s_device) {
            return 0;
        }

        const char *name = [[s_device name] UTF8String];
        if (name) {
            strncpy(s_device_name, name, sizeof(s_device_name) - 1);
        }

        s_page_size = (size_t)getpagesize();
        if (s_page_size == 0) s_page_size = 16384;

        s_queue = [s_device newCommandQueue];
        if (!s_queue) {
            s_device = nil;
            return 0;
        }

        NSString *src = [NSString stringWithUTF8String:K3_METAL_SHADERS];
        NSError *err = nil;
        id<MTLLibrary> lib = [s_device newLibraryWithSource:src options:nil error:&err];
        if (!lib) {
            fprintf(stderr, "k3_metal: shader compilation failed: %s\n",
                    [[err localizedDescription] UTF8String]);
            s_queue = nil;
            s_device = nil;
            return 0;
        }

        id<MTLFunction> fn_bf16 = [lib newFunctionWithName:@"k3_metal_kernel_matmul_bf16"];
        id<MTLFunction> fn_f32 = [lib newFunctionWithName:@"k3_metal_kernel_matmul_f32"];
        id<MTLFunction> fn_mxfp4 = [lib newFunctionWithName:@"k3_metal_kernel_matmul_mxfp4"];

        if (!fn_bf16 || !fn_f32 || !fn_mxfp4) {
            fprintf(stderr, "k3_metal: missing kernel functions in shader library\n");
            return 0;
        }

        s_pipe_bf16 = [s_device newComputePipelineStateWithFunction:fn_bf16 error:&err];
        s_pipe_f32 = [s_device newComputePipelineStateWithFunction:fn_f32 error:&err];
        s_pipe_mxfp4 = [s_device newComputePipelineStateWithFunction:fn_mxfp4 error:&err];

        if (!s_pipe_bf16 || !s_pipe_f32 || !s_pipe_mxfp4) {
            fprintf(stderr, "k3_metal: pipeline creation failed: %s\n",
                    [[err localizedDescription] UTF8String]);
            return 0;
        }

        s_initialized = 1;
        return 1;
    }
}

int k3_metal_is_available(void)
{
    if (!s_initialized) {
        return k3_metal_init();
    }
    return (s_device != nil);
}

int k3_metal_is_enabled(void)
{
    return s_enabled && s_initialized && (s_device != nil);
}

void k3_metal_set_enabled(int enabled)
{
    s_enabled = enabled;
    if (enabled && !s_initialized) {
        k3_metal_init();
    }
}

const char *k3_metal_device_name(void)
{
    if (!s_initialized) k3_metal_init();
    return s_device_name;
}

#define K3_METAL_CACHE_MAX 128
typedef struct {
    void *base_page;
    size_t length;
    id<MTLBuffer> buffer;
    uint64_t lru;
} MetalCacheEntry;

static MetalCacheEntry s_buf_cache[K3_METAL_CACHE_MAX];
static uint64_t s_lru_counter = 0;
static os_unfair_lock s_cache_lock = OS_UNFAIR_LOCK_INIT;

void k3_metal_shutdown(void)
{
    if (!s_initialized) return;
    @autoreleasepool {
        os_unfair_lock_lock(&s_cache_lock);
        for (int i = 0; i < K3_METAL_CACHE_MAX; i++) {
            s_buf_cache[i].buffer = nil;
            s_buf_cache[i].base_page = NULL;
            s_buf_cache[i].length = 0;
            s_buf_cache[i].lru = 0;
        }
        os_unfair_lock_unlock(&s_cache_lock);

        s_pipe_bf16 = nil;
        s_pipe_f32 = nil;
        s_pipe_mxfp4 = nil;
        s_queue = nil;
        s_device = nil;
        s_initialized = 0;
        strcpy(s_device_name, "None");
    }
}

/* Helper to acquire a zero-copy buffer or fallback copy buffer for a memory pointer */
static inline id<MTLBuffer> get_metal_buffer(const void *ptr, size_t bytes, NSUInteger *out_offset)
{
    if (!ptr || bytes == 0) {
        *out_offset = 0;
        return nil;
    }

    uintptr_t uptr = (uintptr_t)ptr;
    uintptr_t page_offset = uptr % s_page_size;
    void *base_page = (void *)(uptr - page_offset);
    size_t total_len = bytes + page_offset;
    size_t aligned_len = (total_len + s_page_size - 1) & ~(s_page_size - 1);

    os_unfair_lock_lock(&s_cache_lock);
    s_lru_counter++;
    for (int i = 0; i < K3_METAL_CACHE_MAX; i++) {
        if (s_buf_cache[i].base_page == base_page &&
            s_buf_cache[i].length >= aligned_len &&
            s_buf_cache[i].buffer != nil) {
            s_buf_cache[i].lru = s_lru_counter;
            id<MTLBuffer> found = s_buf_cache[i].buffer;
            os_unfair_lock_unlock(&s_cache_lock);
            *out_offset = page_offset;
            return found;
        }
    }

    id<MTLBuffer> buf = [s_device newBufferWithBytesNoCopy:base_page
                                                    length:aligned_len
                                                   options:MTLResourceStorageModeShared
                                               deallocator:nil];
    if (buf) {
        int victim = 0;
        uint64_t oldest = UINT64_MAX;
        for (int i = 0; i < K3_METAL_CACHE_MAX; i++) {
            if (s_buf_cache[i].buffer == nil) {
                victim = i;
                break;
            }
            if (s_buf_cache[i].lru < oldest) {
                oldest = s_buf_cache[i].lru;
                victim = i;
            }
        }
        s_buf_cache[victim].base_page = base_page;
        s_buf_cache[victim].length = aligned_len;
        s_buf_cache[victim].buffer = buf;
        s_buf_cache[victim].lru = s_lru_counter;
        os_unfair_lock_unlock(&s_cache_lock);

        *out_offset = page_offset;
        return buf;
    }
    os_unfair_lock_unlock(&s_cache_lock);

    // Fallback if zero-copy is not permitted for this range
    *out_offset = 0;
    return [s_device newBufferWithBytes:ptr length:bytes options:MTLResourceStorageModeShared];
}

int k3_metal_matmul_bf16(float *y, const float *x, const uint16_t *W, int in, int out)
{
    if (!k3_metal_is_enabled()) return -1;
    if (in <= 0 || out <= 0 || !y || !x || !W) return -1;

    @autoreleasepool {
        size_t W_bytes = (size_t)in * out * sizeof(uint16_t);
        size_t x_bytes = (size_t)in * sizeof(float);
        size_t y_bytes = (size_t)out * sizeof(float);

        NSUInteger W_offset = 0, x_offset = 0, y_offset = 0;
        id<MTLBuffer> buf_W = get_metal_buffer(W, W_bytes, &W_offset);
        id<MTLBuffer> buf_x = get_metal_buffer(x, x_bytes, &x_offset);
        id<MTLBuffer> buf_y = get_metal_buffer(y, y_bytes, &y_offset);

        if (!buf_W || !buf_x || !buf_y) return -1;

        id<MTLCommandBuffer> cmd = [s_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:s_pipe_bf16];

        [enc setBuffer:buf_y offset:y_offset atIndex:0];
        [enc setBuffer:buf_x offset:x_offset atIndex:1];
        [enc setBuffer:buf_W offset:W_offset atIndex:2];
        [enc setBytes:&in length:sizeof(int) atIndex:3];
        [enc setBytes:&out length:sizeof(int) atIndex:4];

        // 4 simdgroups (128 threads) per threadgroup = 4 rows per threadgroup
        MTLSize tgSize = MTLSizeMake(128, 1, 1);
        MTLSize numTgs = MTLSizeMake((out + 3) / 4, 1, 1);

        [enc dispatchThreadgroups:numTgs threadsPerThreadgroup:tgSize];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        // If buf_y was copied rather than zero-copy mapped, sync back
        if (y_offset == 0 && [buf_y contents] != y) {
            memcpy(y, [buf_y contents], y_bytes);
        }
        return 0;
    }
}

int k3_metal_matmul_f32(float *y, const float *x, const float *W, int in, int out)
{
    if (!k3_metal_is_enabled()) return -1;
    if (in <= 0 || out <= 0 || !y || !x || !W) return -1;

    @autoreleasepool {
        size_t W_bytes = (size_t)in * out * sizeof(float);
        size_t x_bytes = (size_t)in * sizeof(float);
        size_t y_bytes = (size_t)out * sizeof(float);

        NSUInteger W_offset = 0, x_offset = 0, y_offset = 0;
        id<MTLBuffer> buf_W = get_metal_buffer(W, W_bytes, &W_offset);
        id<MTLBuffer> buf_x = get_metal_buffer(x, x_bytes, &x_offset);
        id<MTLBuffer> buf_y = get_metal_buffer(y, y_bytes, &y_offset);

        if (!buf_W || !buf_x || !buf_y) return -1;

        id<MTLCommandBuffer> cmd = [s_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:s_pipe_f32];

        [enc setBuffer:buf_y offset:y_offset atIndex:0];
        [enc setBuffer:buf_x offset:x_offset atIndex:1];
        [enc setBuffer:buf_W offset:W_offset atIndex:2];
        [enc setBytes:&in length:sizeof(int) atIndex:3];
        [enc setBytes:&out length:sizeof(int) atIndex:4];

        MTLSize tgSize = MTLSizeMake(128, 1, 1);
        MTLSize numTgs = MTLSizeMake((out + 3) / 4, 1, 1);

        [enc dispatchThreadgroups:numTgs threadsPerThreadgroup:tgSize];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if (y_offset == 0 && [buf_y contents] != y) {
            memcpy(y, [buf_y contents], y_bytes);
        }
        return 0;
    }
}

int k3_metal_matmul_mxfp4(float *y, const float *x, const unsigned char *packed,
                          const unsigned char *scales, int in, int rows, int group)
{
    if (!k3_metal_is_enabled()) return -1;
    if (in <= 0 || rows <= 0 || group <= 0 || !y || !x || !packed || !scales) return -1;

    @autoreleasepool {
        int pcols = in / 2;
        int ngrp = in / group;
        size_t pk_bytes = (size_t)rows * pcols;
        size_t sc_bytes = (size_t)rows * ngrp;
        size_t x_bytes = (size_t)in * sizeof(float);
        size_t y_bytes = (size_t)rows * sizeof(float);

        NSUInteger pk_offset = 0, sc_offset = 0, x_offset = 0, y_offset = 0;
        id<MTLBuffer> buf_pk = get_metal_buffer(packed, pk_bytes, &pk_offset);
        id<MTLBuffer> buf_sc = get_metal_buffer(scales, sc_bytes, &sc_offset);
        id<MTLBuffer> buf_x = get_metal_buffer(x, x_bytes, &x_offset);
        id<MTLBuffer> buf_y = get_metal_buffer(y, y_bytes, &y_offset);

        if (!buf_pk || !buf_sc || !buf_x || !buf_y) return -1;

        id<MTLCommandBuffer> cmd = [s_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:s_pipe_mxfp4];

        [enc setBuffer:buf_y offset:y_offset atIndex:0];
        [enc setBuffer:buf_x offset:x_offset atIndex:1];
        [enc setBuffer:buf_pk offset:pk_offset atIndex:2];
        [enc setBuffer:buf_sc offset:sc_offset atIndex:3];
        [enc setBytes:&in length:sizeof(int) atIndex:4];
        [enc setBytes:&rows length:sizeof(int) atIndex:5];
        [enc setBytes:&group length:sizeof(int) atIndex:6];

        MTLSize tgSize = MTLSizeMake(128, 1, 1);
        MTLSize numTgs = MTLSizeMake((rows + 3) / 4, 1, 1);

        [enc dispatchThreadgroups:numTgs threadsPerThreadgroup:tgSize];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if (y_offset == 0 && [buf_y contents] != y) {
            memcpy(y, [buf_y contents], y_bytes);
        }
        return 0;
    }
}

#else

/* Non-Apple fallback stubs */
#include "k3_metal.h"

int  k3_metal_init(void) { return 0; }
int  k3_metal_is_available(void) { return 0; }
int  k3_metal_is_enabled(void) { return 0; }
void k3_metal_set_enabled(int enabled) { (void)enabled; }
const char *k3_metal_device_name(void) { return "None"; }
void k3_metal_shutdown(void) {}
int  k3_metal_matmul_bf16(float *y, const float *x, const uint16_t *W, int in, int out)
{
    (void)y; (void)x; (void)W; (void)in; (void)out; return -1;
}
int  k3_metal_matmul_mxfp4(float *y, const float *x, const unsigned char *packed,
                          const unsigned char *scales, int in, int rows, int group)
{
    (void)y; (void)x; (void)packed; (void)scales; (void)in; (void)rows; (void)group; return -1;
}
int  k3_metal_matmul_f32(float *y, const float *x, const float *W, int in, int out)
{
    (void)y; (void)x; (void)W; (void)in; (void)out; return -1;
}

#endif /* __APPLE__ */
