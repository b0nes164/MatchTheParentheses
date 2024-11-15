/******************************************************************************
 * 
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 11/14/2024
 * 
 ******************************************************************************/
#include <cstdint>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define LANE_COUNT 32
#define LANE_MASK 31
#define LANE_LOG 5
#define WARP_INDEX (threadIdx.x >> LANE_LOG)

struct Bic {
    uint32_t a;
    uint32_t b;
};

__device__ __forceinline__ Bic bic_combine(Bic x, Bic y){
    const uint32_t min = x.b < y.a ? x.b : y.a;
    return Bic{x.a + y.a - min, x.b + y.b - min}; 
}

__device__ __forceinline__ uint32_t getLaneId() {
    uint32_t laneId;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneId));
    return laneId;
}

__device__ __forceinline__ Bic InclusiveDescendingWarpScanBic(Bic x) {
    #pragma unroll
    for (uint32_t i = 1; i <= 16; i <<= 1) {  // 16 = LANE_COUNT >> 1
        const uint32_t a = __shfl_down_sync(0xffffffff, x.a, i, 32);
        const uint32_t b = __shfl_down_sync(0xffffffff, x.b, i, 32);
        if (getLaneId() + i < LANE_COUNT) {
            const uint32_t m = x.b < a ? x.b : a;
            x.a = x.a + a - m;
            x.b = x.b + b - m;
        }
    }

    return x;
}

__device__ __forceinline__ Bic IncDescCircShiftBic(Bic x) {
    #pragma unroll
    for (uint32_t i = 1; i <= 16; i <<= 1) {  // 16 = LANE_COUNT >> 1
        const uint32_t a = __shfl_down_sync(0xffffffff, x.a, i, 32);
        const uint32_t b = __shfl_down_sync(0xffffffff, x.b, i, 32);
        if (getLaneId() + i < LANE_COUNT) {
            const uint32_t m = x.b < a ? x.b : a;
            x.a = x.a + a - m;
            x.b = x.b + b - m;
        }
    }

    //Descending so we shift in the reverse direction
    x.a = __shfl_sync(0xffffffff, x.a, getLaneId() + 1 & LANE_MASK);   
    x.b = __shfl_sync(0xffffffff, x.b, getLaneId() + 1 & LANE_MASK);

    return x;
}