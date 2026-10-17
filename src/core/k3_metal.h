/* SPDX-License-Identifier: Apache-2.0 */
/*
 * k3_metal.h - Apple Silicon Metal GPU compute backend for Kimi K3.
 *
 * Provides accelerated matrix multiplication kernels for:
 *   - BF16 trunk matmuls (widen-on-read directly on GPU)
 *   - MXFP4 expert matmuls (OCP E2M1 nibble decode + E8M0 scale in register on GPU)
 *   - FP32 matmuls
 *
 * Leverages Apple Silicon unified memory (MTLResourceStorageModeShared) for
 * zero-copy compute without PCIe transfer overhead.
 */
#ifndef K3_METAL_H
#define K3_METAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize the Metal subsystem (device, command queue, compiled compute pipelines).
 * Returns 1 if a Metal device is initialized, 0 if unavailable or disabled. */
int  k3_metal_init(void);

/* Check whether Metal is available on the current machine. */
int  k3_metal_is_available(void);

/* Check whether Metal execution is currently enabled. */
int  k3_metal_is_enabled(void);

/* Enable or disable Metal GPU dispatch at runtime. */
void k3_metal_set_enabled(int enabled);

/* Name of the active Metal GPU device (e.g. "Apple M4"), or "None". */
const char *k3_metal_device_name(void);

/* Cleanly shutdown and release Metal resources. */
void k3_metal_shutdown(void);

/* GPU-accelerated BF16 matmul: y[out] = W[out][in] . x[in]
 * W is uint16_t bf16. Returns 0 on success, -1 if fallback to CPU is needed. */
int  k3_metal_matmul_bf16(float *y, const float *x, const uint16_t *W, int in, int out);

/* GPU-accelerated MXFP4 matmul: y[rows] = W[rows][in] . x[in]
 * packed is E2M1 pairs, scales is E8M0 exponent bytes.
 * Returns 0 on success, -1 if fallback to CPU is needed. */
int  k3_metal_matmul_mxfp4(float *y, const float *x, const unsigned char *packed,
                          const unsigned char *scales, int in, int rows, int group);

/* GPU-accelerated FP32 matmul: y[out] = W[out][in] . x[in]
 * Returns 0 on success, -1 if fallback to CPU is needed. */
int  k3_metal_matmul_f32(float *y, const float *x, const float *W, int in, int out);

#ifdef __cplusplus
}
#endif

#endif /* K3_METAL_H */
