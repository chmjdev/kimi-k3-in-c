#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// BF16 Matrix-Vector Multiplication: y[out] = W[out][in] * x[in]
// Coalesced 128-bit reads using uint4 (8 BF16s per vector)
// ---------------------------------------------------------------------------
kernel void k3_metal_kernel_matmul_bf16(
    device float *y [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device const ushort *W [[buffer(2)]],
    constant int &in_dim [[buffer(3)]],
    constant int &out_dim [[buffer(4)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]])
{
    // 4 simdgroups per threadgroup = 4 rows per threadgroup
    uint row_idx = tg_id.x * 4 + simd_id;
    if (row_idx >= (uint)out_dim) return;

    device const uint4 *row_u4 = (device const uint4 *)(W + (ulong)row_idx * in_dim);
    device const float4 *x4 = (device const float4 *)x;

    float my_acc = 0.0f;
    int num_u4 = in_dim / 8;

    for (int v = lane_id; v < num_u4; v += 32) {
        uint4 u = row_u4[v];
        float4 x_lo = x4[v * 2];
        float4 x_hi = x4[v * 2 + 1];

        float w0 = as_type<float>((u.x & 0xFFFFu) << 16);
        float w1 = as_type<float>(u.x & 0xFFFF0000u);
        float w2 = as_type<float>((u.y & 0xFFFFu) << 16);
        float w3 = as_type<float>(u.y & 0xFFFF0000u);
        my_acc += w0 * x_lo.x + w1 * x_lo.y + w2 * x_lo.z + w3 * x_lo.w;

        float w4 = as_type<float>((u.z & 0xFFFFu) << 16);
        float w5 = as_type<float>(u.z & 0xFFFF0000u);
        float w6 = as_type<float>((u.w & 0xFFFFu) << 16);
        float w7 = as_type<float>(u.w & 0xFFFF0000u);
        my_acc += w4 * x_hi.x + w5 * x_hi.y + w6 * x_hi.z + w7 * x_hi.w;
    }

    // Remainder if in_dim is not a multiple of 8
    int tail_start = num_u4 * 8 + lane_id;
    if (tail_start < in_dim) {
        device const ushort *row = W + (ulong)row_idx * in_dim;
        uint u = (uint)row[tail_start] << 16;
        my_acc += as_type<float>(u) * x[tail_start];
    }

    float row_sum = simd_sum(my_acc);
    if (lane_id == 0) {
        y[row_idx] = row_sum;
    }
}

// ---------------------------------------------------------------------------
// FP32 Matrix-Vector Multiplication: y[out] = W[out][in] * x[in]
// Coalesced 128-bit reads using float4
// ---------------------------------------------------------------------------
kernel void k3_metal_kernel_matmul_f32(
    device float *y [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device const float *W [[buffer(2)]],
    constant int &in_dim [[buffer(3)]],
    constant int &out_dim [[buffer(4)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]])
{
    uint row_idx = tg_id.x * 4 + simd_id;
    if (row_idx >= (uint)out_dim) return;

    device const float4 *row4 = (device const float4 *)(W + (ulong)row_idx * in_dim);
    device const float4 *x4   = (device const float4 *)x;

    float my_acc = 0.0f;
    int num_f4 = in_dim / 4;

    for (int v = lane_id; v < num_f4; v += 32) {
        my_acc += dot(row4[v], x4[v]);
    }

    int tail_start = num_f4 * 4 + lane_id;
    if (tail_start < in_dim) {
        device const float *row = W + (ulong)row_idx * in_dim;
        my_acc += row[tail_start] * x[tail_start];
    }

    float row_sum = simd_sum(my_acc);
    if (lane_id == 0) {
        y[row_idx] = row_sum;
    }
}

// ---------------------------------------------------------------------------
// MXFP4 Matrix-Vector Multiplication: y[rows] = W[rows][in] * x[in]
// OCP Microscaling FP4 (E2M1) with E8M0 group scales
// ---------------------------------------------------------------------------
constant float K3_LUT_E2M1[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

inline float k3_e8m0_scale(uchar sb) {
    if (sb == 255 || sb == 0) return 0.0f;
    return as_type<float>((uint)sb << 23);
}

kernel void k3_metal_kernel_matmul_mxfp4(
    device float *y [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device const uchar *packed [[buffer(2)]],
    device const uchar *scales [[buffer(3)]],
    constant int &in_dim [[buffer(4)]],
    constant int &rows_dim [[buffer(5)]],
    constant int &group [[buffer(6)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]])
{
    uint row_idx = tg_id.x * 4 + simd_id;
    if (row_idx >= (uint)rows_dim) return;

    int pcols = in_dim / 2;
    int ngrp = in_dim / group;
    device const uchar *pr = packed + (ulong)row_idx * pcols;
    device const uchar *sr = scales + (ulong)row_idx * ngrp;

    float my_acc = 0.0f;

    if (group == 32) {
        device const uint4 *row_pk4 = (device const uint4 *)pr;
        for (int g = lane_id; g < ngrp; g += 32) {
            uchar sb = sr[g];
            if (sb == 255 || sb == 0) continue;
            float sc = as_type<float>((uint)sb << 23);
            uint4 p4 = row_pk4[g];
            int xb0 = g * 32;
            float group_acc = 0.0f;

            for (int w_idx = 0; w_idx < 4; w_idx++) {
                uint u = (w_idx == 0) ? p4.x : ((w_idx == 1) ? p4.y : ((w_idx == 2) ? p4.z : p4.w));
                int xb = xb0 + w_idx * 8;
                for (int k = 0; k < 4; k++) {
                    uchar b = (u >> (k * 8)) & 0xFF;
                    group_acc += K3_LUT_E2M1[b & 0x0F] * x[xb + k * 2] + K3_LUT_E2M1[b >> 4] * x[xb + k * 2 + 1];
                }
            }
            my_acc += group_acc * sc;
        }
    } else {
        int byte_off = lane_id >> 1;
        int is_hi = lane_id & 1;
        for (int g = 0; g < ngrp; g++) {
            uchar sb = sr[g];
            if (sb == 255 || sb == 0) continue;
            float sc = as_type<float>((uint)sb << 23);
            int g_start = g * group;
            device const uchar *pb = pr + (ulong)(g_start / 2);
            for (int e = lane_id; e < group; e += 32) {
                if (g_start + e < in_dim) {
                    uchar byte_val = pb[e >> 1];
                    uchar nib = (e & 1) ? (byte_val >> 4) : (byte_val & 0x0F);
                    float weight = K3_LUT_E2M1[nib] * sc;
                    my_acc += weight * x[g_start + e];
                }
            }
        }
    }

    float row_sum = simd_sum(my_acc);
    if (lane_id == 0) {
        y[row_idx] = row_sum;
    }
}
