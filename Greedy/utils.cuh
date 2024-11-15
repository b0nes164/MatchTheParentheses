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
#define UNROLL #pragma unroll

__device__ __forceinline__ uint32_t getLaneId() {
    uint32_t laneId;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneId));
    return laneId;
}

__device__ __forceinline__ unsigned getLaneMaskGt() {
    unsigned mask;
    asm("mov.u32 %0, %%lanemask_gt;" : "=r"(mask));
    return mask;
}

__device__ __forceinline__ unsigned getLaneMaskLt() {
    unsigned mask;
    asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
    return mask;
}

template<typename T>
__device__ __forceinline__ T InclusiveWarpScan(T val) {
    #pragma unroll
    for (uint32_t i = 1; i <= 16; i <<= 1) {  // 16 = LANE_COUNT >> 1
        const T t = __shfl_up_sync(0xffffffff, val, i, 32);
        if (getLaneId() >= i)
            val += t;
    }

    return val;
}

template<typename T>
__device__ __forceinline__ T InclusiveWarpScanCircularShift(T val)
{
    #pragma unroll
    for (int i = 1; i <= 16; i <<= 1) {  // 16 = LANE_COUNT >> 1
        const T t = __shfl_up_sync(0xffffffff, val, i, 32);
        if (getLaneId() >= i) val += t;
    }

    return __shfl_sync(0xffffffff, val, getLaneId() + LANE_MASK & LANE_MASK);
}

template<typename T>
__device__ __forceinline__ T ExclusiveWarpScan(T val) {
    #pragma unroll
    for (uint32_t i = 1; i <= 16; i <<= 1) {  // 16 = LANE_COUNT >> 1
        const T t = __shfl_up_sync(0xffffffff, val, i, 32);
        if (getLaneId() >= i)
            val += t;
    }

    const T t = __shfl_up_sync(0xffffffff, val, 1, 32);
    return getLaneId() ? t : 0;
}

__device__ __forceinline__ int32_t WarpMin(int32_t val) {
    #pragma unroll
    for (uint32_t mask = 16; mask; mask >>= 1) {  // 16 = LANE_COUNT >> 1
        const int32_t t = __shfl_xor_sync(0xffffffff, val, mask, LANE_COUNT);
        if (t < val)
            val = t;
    }
    return val;
}