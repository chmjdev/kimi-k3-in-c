/* SPDX-License-Identifier: Apache-2.0 */
/*
 * test_metal.c - Unit test verifying Apple Silicon Metal GPU parity against CPU.
 *
 * Tests:
 *   1. BF16 matmul parity
 *   2. FP32 matmul parity
 *   3. MXFP4 matmul parity (< 1e-4 tolerance against CPU reference)
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "k3.h"
#include "k3_metal.h"

static void fillf(float *p, size_t n, unsigned s)
{
    for (size_t i = 0; i < n; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        p[i] = ((float)(s >> 8) / 8388608.0f - 1.0f) * 0.05f;
    }
}

static void fillbf16(uint16_t *p, size_t n, unsigned s)
{
    for (size_t i = 0; i < n; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        const float f = ((float)(s >> 8) / 8388608.0f - 1.0f) * 0.05f;
        unsigned u; memcpy(&u, &f, sizeof u);
        p[i] = (uint16_t)(u >> 16);
    }
}

static void fillb(unsigned char *p, size_t n, unsigned s)
{
    for (size_t i = 0; i < n; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        p[i] = (unsigned char)(s >> 13);
    }
}

int main(void)
{
    printf("== Apple Silicon Metal GPU Verification ==\n");

    if (!k3_metal_is_available()) {
        printf("  SKIPPED: Metal device is not available on this platform\n");
        return 0;
    }

    printf("  Found Metal GPU: %s\n", k3_metal_device_name());

    /* Test 1: BF16 Matmul Parity */
    {
        const int in = 7168, out = 1024;
        uint16_t *W = (uint16_t *)malloc((size_t)in * out * sizeof(uint16_t));
        float *x = (float *)malloc((size_t)in * sizeof(float));
        float *y_cpu = (float *)malloc((size_t)out * sizeof(float));
        float *y_gpu = (float *)malloc((size_t)out * sizeof(float));

        fillbf16(W, (size_t)in * out, 12345u);
        fillf(x, in, 999u);

        /* Run CPU reference (Metal disabled) */
        k3_metal_set_enabled(0);
        k3_matmul_bf16(y_cpu, x, W, in, out);

        /* Run Metal GPU (Metal enabled) */
        k3_metal_set_enabled(1);
        int rc = k3_metal_matmul_bf16(y_gpu, x, W, in, out);
        if (rc != 0) {
            fprintf(stderr, "  FAIL: k3_metal_matmul_bf16 returned %d\n", rc);
            return 1;
        }

        float max_diff = 0.0f;
        for (int o = 0; o < out; o++) {
            float diff = fabsf(y_cpu[o] - y_gpu[o]);
            if (diff > max_diff) max_diff = diff;
        }
        printf("  BF16 matmul: max |CPU - GPU| = %e\n", max_diff);
        if (max_diff > 1e-4f) {
            fprintf(stderr, "  FAIL: BF16 diff %e exceeds tolerance\n", max_diff);
            return 1;
        }
        printf("  PASS: BF16 matmul matches CPU reference\n");

        free(W); free(x); free(y_cpu); free(y_gpu);
    }

    /* Test 2: FP32 Matmul Parity */
    {
        const int in = 3584, out = 1024;
        float *W = (float *)malloc((size_t)in * out * sizeof(float));
        float *x = (float *)malloc((size_t)in * sizeof(float));
        float *y_cpu = (float *)malloc((size_t)out * sizeof(float));
        float *y_gpu = (float *)malloc((size_t)out * sizeof(float));

        fillf(W, (size_t)in * out, 54321u);
        fillf(x, in, 111u);

        k3_metal_set_enabled(0);
        k3_matmul(y_cpu, x, W, in, out);

        k3_metal_set_enabled(1);
        int rc = k3_metal_matmul_f32(y_gpu, x, W, in, out);
        if (rc != 0) {
            fprintf(stderr, "  FAIL: k3_metal_matmul_f32 returned %d\n", rc);
            return 1;
        }

        float max_diff = 0.0f;
        for (int o = 0; o < out; o++) {
            float diff = fabsf(y_cpu[o] - y_gpu[o]);
            if (diff > max_diff) max_diff = diff;
        }
        printf("  FP32 matmul: max |CPU - GPU| = %e\n", max_diff);
        if (max_diff > 1e-4f) {
            fprintf(stderr, "  FAIL: FP32 diff %e exceeds tolerance\n", max_diff);
            return 1;
        }
        printf("  PASS: FP32 matmul matches CPU reference\n");

        free(W); free(x); free(y_cpu); free(y_gpu);
    }

    /* Test 3: MXFP4 Matmul Parity */
    {
        const int in = 3584, rows = 1024, group = K3_MXFP4_GROUP;
        const int pcols = in / 2, ngrp = in / group;
        unsigned char *pk = (unsigned char *)malloc((size_t)rows * pcols);
        unsigned char *sc = (unsigned char *)malloc((size_t)rows * ngrp);
        float *x = (float *)malloc((size_t)in * sizeof(float));
        float *y_cpu = (float *)malloc((size_t)rows * sizeof(float));
        float *y_gpu = (float *)malloc((size_t)rows * sizeof(float));

        fillb(pk, (size_t)rows * pcols, 777u);
        memset(sc, 127, (size_t)rows * ngrp);
        fillf(x, in, 4242u);

        k3_metal_set_enabled(0);
        k3_matmul_mxfp4(y_cpu, x, pk, sc, in, rows, group);

        k3_metal_set_enabled(1);
        int rc = k3_metal_matmul_mxfp4(y_gpu, x, pk, sc, in, rows, group);
        if (rc != 0) {
            fprintf(stderr, "  FAIL: k3_metal_matmul_mxfp4 returned %d\n", rc);
            return 1;
        }

        float max_diff = 0.0f;
        for (int r = 0; r < rows; r++) {
            float diff = fabsf(y_cpu[r] - y_gpu[r]);
            if (diff > max_diff) max_diff = diff;
        }
        printf("  MXFP4 matmul: max |CPU - GPU| = %e\n", max_diff);
        if (max_diff > 1e-4f) {
            fprintf(stderr, "  FAIL: MXFP4 diff %e exceeds tolerance\n", max_diff);
            return 1;
        }
        printf("  PASS: MXFP4 matmul matches CPU reference\n");

        free(pk); free(sc); free(x); free(y_cpu); free(y_gpu);
    }

    k3_metal_shutdown();
    printf("ALL METAL TESTS PASSED\n");
    return 0;
}
