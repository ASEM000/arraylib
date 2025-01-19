#ifndef ARRAYLIB_SIMD_H
#define ARRAYLIB_SIMD_H

#include "arraylib.h"

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#define SIMD_ID 1
#include <arm_neon.h>
#define SIMD_WIDTH 4
typedef float32x4_t simd_t;

#define SIMD_SET1(x) vdupq_n_f32(x)
#define SIMD_FMADDL(acc, x, y, z) vfmaq_laneq_f32(acc, x, y, z)  // acc += x * y[lane]
#define SIMD_LOAD(x) vld1q_f32(x)
#define SIMD_STORE(x, y) vst1q_f32(x, y)

static inline NDArray* array_array_matmul_contiguous_4x4_neon(
        const NDArray* lhs,
        const NDArray* rhs) {
    assert(lhs->lay->ndim == 2 && rhs->lay->ndim == 2);
    assert(lhs->lay->ndim == 2 && rhs->lay->ndim == 2);
    assert(lhs->lay->shape[1] == rhs->lay->shape[0]);

    const size_t M = lhs->lay->shape[0];
    const size_t K = lhs->lay->shape[1];
    const size_t N = rhs->lay->shape[1];

    NDArray* out = array_zeros((size_t[]){M, N}, 2);
    // process 4x4 blocks
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < M; i += SIMD_WIDTH) {
        for (size_t j = 0; j < N; j += SIMD_WIDTH) {
            // accumulators for SIMD_WIDTH rows
            simd_t c0 = SIMD_SET1(0.0f);
            simd_t c1 = SIMD_SET1(0.0f);
            simd_t c2 = SIMD_SET1(0.0f);
            simd_t c3 = SIMD_SET1(0.0f);
            for (size_t k = 0; k < K; k++) {
                simd_t a = SIMD_LOAD(&lhs->ptr[i * K + k]);  // M x K
                simd_t b = SIMD_LOAD(&rhs->ptr[k * N + j]);  // K x N
                c0 = SIMD_FMADDL(c0, b, a, 0);               // a[0] * b
                c1 = SIMD_FMADDL(c0, b, a, 1);
                c2 = SIMD_FMADDL(c0, b, a, 2);
                c3 = SIMD_FMADDL(c0, b, a, 3);
            }
            if (i + 0 < M && j + 4 <= N) SIMD_STORE(&out->ptr[(i + 0) * N + j], c0);
            if (i + 1 < M && j + 4 <= N) SIMD_STORE(&out->ptr[(i + 1) * N + j], c1);
            if (i + 2 < M && j + 4 <= N) SIMD_STORE(&out->ptr[(i + 2) * N + j], c2);
            if (i + 3 < M && j + 4 <= N) SIMD_STORE(&out->ptr[(i + 3) * N + j], c3);
        }
    }
    return out;
}
#define SIMD_MATMUL array_array_matmul_contiguous_4x4_neon

#else
#define SIMD_ID 0

#endif

#endif