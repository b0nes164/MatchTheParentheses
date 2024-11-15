/******************************************************************************
 * Greedy GPU Parentheses Matcher 
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 11/14/2024
 * 
 ******************************************************************************/
#include <cstdio>
#include <stack>
#include <vector>

#include "utils.cuh"

#define checkCudaError()                                         \
    {                                                            \
        cudaError_t err = cudaGetLastError();                    \
        if (err != cudaSuccess) {                                \
            printf("CUDA Error: %s\n", cudaGetErrorString(err)); \
        }                                                        \
    }

#define FLAG_NOT_READY 0
#define FLAG_READY 1
#define FLAG_INCLUSIVE 2
#define FLAG_MASK 3

uint32_t LCGTausworth(uint4& t) {
    t.x = ((t.x & 4294967294U) << 12) ^ (((t.x << 13) ^ t.x) >> 19);
    t.y = ((t.y & 4294967288U) << 4) ^ (((t.y << 2) ^ t.y) >> 25);
    t.z = ((t.z & 4294967280U) << 17) ^ (((t.z << 3) ^ t.z) >> 11);
    t.w = t.w * 1664525 + 1013904223U;
    return t.x ^ t.y ^ t.z ^ t.w;
}

void InitHost(uint32_t* hostData, const uint32_t size, const uint32_t maxDepth, const uint32_t seed) {
    uint4 t =
        make_uint4(seed * 1000000007, seed * 2000000011, seed * 3000000019, seed * 4000000007);
    uint32_t depth = 0;
    for (uint32_t i = 0; i < size - depth; ++i) {
        const uint32_t rand = LCGTausworth(t);
        if (!depth) {
            hostData[i] = 1;
            depth++;
        } else if (depth >= maxDepth - 1) {
            hostData[i] = 0;
            depth--;
        } else {
            if (rand & 1) {
                hostData[i] = 0;
                depth--;
            } else {
                hostData[i] = 1;
                depth++;
            }
        }
    }

    for (uint32_t i = size - depth; i < size; ++i) {
        hostData[i] = 0;
    }
}

bool GetHostSolution(uint32_t* hostData, uint32_t size) {
    std::stack<uint32_t> t;
    for (int i = 0; i < size; ++i) {
        if (hostData[i]) {
            t.push(i);
        } else {
            uint32_t top = t.top();
            t.pop();
            hostData[i] = top;
        }
    }

    return t.empty();
}

__global__ void ValidateKernel(uint32_t* data, uint32_t* expected, uint32_t* err,
                               const uint32_t size) {
    for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < size;
         i += blockDim.x * gridDim.x) {
        if (data[i] != expected[i]) {
            // printf("Error at %u expected: %u, got: %u\n", i, expected[i], braces[i]);
            atomicAdd(&err[0], 1);
            atomicAdd(&err[1], i - expected[i]);
        }
    }
}

template <uint32_t START_LOG, uint32_t END_LOG>
__device__ __forceinline__ void MultiSplitMatch(unsigned& mask, const bool notComplete,
                                                const int32_t target, const int32_t depth) {
    #pragma unroll
    for (uint32_t k = START_LOG; k <= END_LOG; ++k) {
        const uint32_t bit = 1 << k;
        const bool depthPred = (depth & bit) == 0;
        const unsigned ballot = __ballot_sync(0xffffffff, depthPred);
        if (notComplete) {
            bool targetPred = (target & bit);
            if (!targetPred)
                mask &= ballot;
            else
                mask &= ~ballot;
        }
    }
}

template <uint32_t START_LOG, uint32_t END_LOG>
__device__ __forceinline__ void MultiSplitMatch2(uint64_t& mask0, uint32_t& mask1,
                                                 const bool incomplete0, const bool incomplete1,
                                                 const int32_t target0, const int32_t target1,
                                                 const int32_t depth0, const int32_t depth1) {
    #pragma unroll
    for (uint32_t k = START_LOG; k < END_LOG; ++k) {
        const uint32_t bit = 1 << k;
        const bool depthPred1 = (depth1 & bit) == 0;
        const uint32_t bal1 = __ballot_sync(0xffffffff, depthPred1);
        if (incomplete1) {
            const bool targetPred1 = (target1 & bit) == 0;
            if (targetPred1) {
                mask1 &= bal1;
            } else {
                mask1 &= ~bal1;
            }
        }
        const bool depthPred0 = (depth0 & bit) == 0;
        const uint32_t bal0 = __ballot_sync(0xffffffff, depthPred0);
        uint64_t combined = 0;
        asm("{\n"
            "    mov.b64 %0, {%1, %2};\n"
            "}\n"
            : "+l"(combined)
            : "r"(bal0), "r"(bal1));
        if (incomplete0) {
            const bool targetPred0 = (target0 & bit) == 0;
            if (targetPred0) {
                mask0 &= combined;
            } else {
                mask0 &= ~combined;
            }
        }
    }
}

__device__ __forceinline__ uint32_t PackageFlag(int32_t val, int32_t flag) {
    uint32_t t;
    if (val < 0) {
        t = val * -1 << 2 | 0x80000000 | flag;
    } else {
        t = val << 2 | flag;
    }
    return t;
}

__device__ __forceinline__ int32_t UnpackageFlag(uint32_t payload) {
    return (payload & 0x80000000 ? -1 : 1) * (payload >> 2 & 0x1fffffff);
}

template <uint32_t PART_SIZE, uint32_t WARPS, uint32_t WARP_LEAVES, uint32_t PER_THREAD>
__global__ void DeviceMatch(uint32_t* data, volatile uint32_t* bump, volatile uint32_t* reductions,
                            int32_t* minimums, int16_t* leafRed, int8_t* leafMin,
                            int32_t* devIncompletes, const uint32_t size) {
    __shared__ uint32_t s_bump;
    __shared__ uint32_t s_bump2;
    __shared__ uint32_t s_broadcast;
    __shared__ int8_t s_depth[PART_SIZE];
    __shared__ int8_t s_leafMin[WARP_LEAVES];
    __shared__ int16_t s_leafRed[WARP_LEAVES];
    __shared__ int32_t s_minRed[WARP_LEAVES / LANE_COUNT];
    __shared__ int32_t s_blockRed;

    __shared__ int8_t s_target[PART_SIZE];
    __shared__ uint8_t s_leafIndex[PART_SIZE];
    __shared__ uint16_t s_openIndex[PART_SIZE];

    if (threadIdx.x == 0) {
        s_bump = 0;
        s_bump2 = 0;
        s_broadcast = atomicAdd((uint32_t*)&bump[0], 1);
    }
    __syncthreads();
    const uint32_t partIndex = s_broadcast;
    const uint32_t devOffset = partIndex * PART_SIZE;

    int32_t depth[PER_THREAD];
    bool incomplete[PER_THREAD];
    {
        const uint32_t warpOffsetA = WARP_INDEX * PER_THREAD;
        const uint32_t warpOffsetB = warpOffsetA * LANE_COUNT;
        #pragma unroll
        for (uint32_t i = getLaneId() + warpOffsetB, k = 0; k < PER_THREAD; i += LANE_COUNT, ++k) {
            const uint32_t t = data[i + devOffset];
            incomplete[k] = t == 1;
            depth[k] = InclusiveWarpScan(incomplete[k] ? 1 : -1);
            s_depth[i] = depth[k];
            const int32_t min = WarpMin(depth[k]);
            if (getLaneId() == 0) {
                s_leafMin[k + warpOffsetA] = min;
            }
        }

        // Attempt to match every open parentheses to its
        // closing mate withn the partition of a warp.
        // Uses a multisplitting style technique that searches for the
        // next depth with this parenthesis' depth - 1
        #pragma unroll
        for (uint32_t k = 0; k < PER_THREAD; k += 2) {
            uint64_t mask0 = (uint64_t)0xffffffff << 32 | getLaneMaskGt();
            uint32_t mask1 = getLaneMaskGt();
            const int32_t adjDepth0 = depth[k] + 64;
            const int32_t adjDepth1 =
                depth[k + 1] + __shfl_sync(0xffffffff, depth[k], LANE_MASK) + 64;
            const int32_t target0 = adjDepth0 - 1;
            const int32_t target1 = adjDepth1 - 1;
            MultiSplitMatch2<0, LANE_LOG + 1>(mask0, mask1, incomplete[k], incomplete[k + 1],
                                              target0, target1, adjDepth0, adjDepth1);

            uint32_t totalOffset = warpOffsetB + k * LANE_COUNT + devOffset;
            if (incomplete[k] && mask0 != 0) {
                data[__ffsll(mask0) - 1 + totalOffset] = getLaneId() + totalOffset;
                incomplete[k] = false;
            }

            totalOffset += LANE_COUNT;
            if (incomplete[k + 1] && mask1 != 0) {
                data[__ffs(mask1) - 1 + totalOffset] = getLaneId() + totalOffset;
                incomplete[k + 1] = false;
            }

            // If closing pair was not found within the
            // warp partition, use atomic bumping to
            // add the parenthesis to a workgroup level
            // list of incompletes
            const uint32_t t0 = __shfl_sync(0xffffffff, depth[k], LANE_MASK);
            if (incomplete[k]) {
                const uint32_t index = atomicAdd(&s_bump, 1);
                s_target[index] = depth[k] - t0 - 1;
                s_leafIndex[index] = 1 + k + warpOffsetA;
                s_openIndex[index] = getLaneId() + warpOffsetB + k * LANE_COUNT;
            }

            const uint32_t t1 = __shfl_sync(0xffffffff, depth[k + 1], LANE_MASK);
            if (incomplete[k + 1]) {
                const uint32_t index = atomicAdd(&s_bump, 1);
                s_target[index] = depth[k + 1] - t1 - 1;
                s_leafIndex[index] = 2 + k + warpOffsetA;
                s_openIndex[index] = getLaneId() + warpOffsetB + (k + 1) * LANE_COUNT;
            }
        }
    }
    __syncthreads();

    // Post the leaf reductions and minimums into device memory,
    // so they can be traversed if necessary. Get the block reduction,
    // and post that also. Scan across the leaf reductions.
    if (threadIdx.x < WARP_LEAVES) {
        s_leafRed[threadIdx.x] =
            InclusiveWarpScanCircularShift((int32_t)s_depth[(threadIdx.x << LANE_LOG) + LANE_MASK]);
    }
    __syncthreads();

    if (threadIdx.x < LANE_COUNT) {
        const bool pred = threadIdx.x < WARP_LEAVES / LANE_COUNT;
        const int32_t t = ExclusiveWarpScan(pred ? s_leafRed[threadIdx.x << LANE_LOG] : 0);
        if (pred) {
            s_leafRed[threadIdx.x << LANE_LOG] = t;
        }
        if (threadIdx.x == LANE_MASK) {  // dumb hack
            atomicExch((uint32_t*)&reductions[partIndex],
                       PackageFlag(t, partIndex ? FLAG_READY : FLAG_INCLUSIVE));
            s_blockRed = t;
        }
    }
    __syncthreads();

    if (threadIdx.x < WARP_LEAVES) {
        int32_t val = s_leafRed[threadIdx.x];
        if (getLaneId()) {
            val += __shfl_sync(0xfffffffe, s_leafRed[threadIdx.x - 1], 1);
        }
        s_leafRed[threadIdx.x] = val;

        const int32_t min = WarpMin(val + (int32_t)s_leafMin[threadIdx.x]);
        if (getLaneId() == 0) {
            s_minRed[WARP_INDEX] = min;
        }
    }
    __syncthreads();

    if (threadIdx.x < LANE_COUNT) {
        bool pred = threadIdx.x < WARP_LEAVES / LANE_COUNT;
        int32_t blockMin = WarpMin(pred ? s_minRed[threadIdx.x] : INT32_MAX);
        if (getLaneId() == 0) {
            minimums[partIndex] = blockMin;
        }
    }

    if (threadIdx.x < WARP_LEAVES) {
        leafRed[partIndex * WARP_LEAVES + threadIdx.x] = s_leafRed[threadIdx.x];
        leafMin[partIndex * WARP_LEAVES + threadIdx.x] = s_leafMin[threadIdx.x];
    }

    if (threadIdx.x == LANE_MASK && partIndex != 0) {
        uint32_t lookbackIndex = partIndex - 1;
        int32_t prevReduction = 0;
        while (true) {
            const uint32_t flagPayload = reductions[lookbackIndex];
            if ((flagPayload & FLAG_MASK) == FLAG_INCLUSIVE) {
                prevReduction += UnpackageFlag(flagPayload);
                atomicExch((uint32_t*)&reductions[partIndex],
                           PackageFlag(prevReduction + s_blockRed, FLAG_INCLUSIVE));
                break;
            }

            if ((flagPayload & FLAG_MASK) == FLAG_READY) {
                prevReduction += UnpackageFlag(flagPayload);
                lookbackIndex--;
            }
        }
    }

    // Attempt to match any remaining incomplete parentheses
    // within the workgroup partition. Use bumping to assign
    // work to better balance workload
    const uint32_t incompleteCount = s_bump;

    // Single Thread, slightly faster than warp
    uint32_t i = atomicAdd(&s_bump2, 1);
    while (i < incompleteCount) {
        int32_t target = s_target[i];
        bool complete = false;
        for (uint32_t j = s_leafIndex[i]; j < WARP_LEAVES; ++j) {
            const uint32_t offset = j << LANE_LOG;
            if (s_leafMin[j] <= target) {
                #pragma unroll
                for (uint32_t k = 0; k < LANE_COUNT; ++k) {
                    if (s_depth[k + offset] == target) {
                        data[k + offset + devOffset] = s_openIndex[i] + devOffset;
                        complete = true;
                        break;
                    }
                }
            }
            if (complete) {
                break;
            } else {
                target -= s_depth[offset + LANE_MASK];
            }
        }

        // If a parenthesis still has not found its mate,
        // add it to a device level list of incompletes
        if (!complete) {
            const uint32_t index = atomicAdd((uint32_t*)&bump[1], 1);
            devIncompletes[index * 2] = target;
            devIncompletes[index * 2 + 1] = s_openIndex[i] + devOffset;
        }
        i = atomicAdd(&s_bump2, 1);
    }

    // Warp
    // uint32_t i;
    // if(getLaneId() == 0){
    //     i = atomicAdd(&s_bump2, 1);
    // }
    // i = __shfl_sync(0xffffffff, i, 0);

    // while(i < incompleteCount) {
    //     uint32_t j = s_leafIndex[i];
    //     if(j == WARP_LEAVES){
    //         if(getLaneId() == 0){
    //             const uint32_t index = atomicAdd((uint32_t*)&bump[1], 1);
    //             devIncompletes[index * 2] = s_target[i];
    //             devIncompletes[index * 2 + 1] = s_openIndex[i] + devOffset;
    //         }
    //     } else {
    //         const int32_t target = s_target[i] + s_leafRed[j];
    //         j += getLaneId();
    //         bool complete = false;
    //         while(j - getLaneId() < WARP_LEAVES){
    //             const bool inBounds = j < WARP_LEAVES;
    //             const int32_t leafSearch = target - (inBounds ? s_leafRed[j] : 0);
    //             const int32_t leafMin = (inBounds ? s_leafMin[j] : 0);
    //             const uint32_t leafBallot = __ballot_sync(0xffffffff, inBounds && leafMin <=
    //             leafSearch); if(leafBallot){
    //                 const uint32_t leafPeer = __ffs(leafBallot) - 1;
    //                 const uint32_t leafIndex = __shfl_sync(0xffffffff, j, leafPeer);
    //                 const uint32_t leafTarget = __shfl_sync(0xffffffff, leafSearch, leafPeer);
    //                 const uint32_t ii = getLaneId() + leafIndex * LANE_COUNT;
    //                 const uint32_t finBallot = __ballot_sync(0xffffffff, leafTarget ==
    //                 s_depth[ii]); if(getLaneId() == __ffs(finBallot) - 1){
    //                     data[ii + devOffset] = s_openIndex[i] + devOffset;
    //                 }
    //                 complete = true;
    //                 break;
    //             } else {
    //                 j += LANE_COUNT;
    //             }
    //         }
    //         if(getLaneId() == 0 && !complete){
    //             const uint32_t index = atomicAdd((uint32_t*)&bump[1], 1);
    //             devIncompletes[index * 2] = target - s_blockRed;
    //             devIncompletes[index * 2 + 1] = s_openIndex[i] + devOffset;
    //         }
    //     }
    //     if(getLaneId() == 0){
    //         i = atomicAdd(&s_bump2, 1);
    //     }
    //     i = __shfl_sync(0xffffffff, i, 0);
    // }
}

// Assign each remaining incomplete parenthesis
// a warp, then traverse the segment tree to find its mate
template <uint32_t PART_SIZE, uint32_t WARP_LEAVES>
__global__ void Cleanup(uint32_t* data, int32_t* devIncompletes, uint32_t* reductions,
                        int32_t* minimums, int16_t* leafRed, int8_t* leafMin, const uint32_t* bump,
                        const uint32_t threadBlocks) {
    const uint32_t cleanupSize = bump[1];
    for (uint32_t index = WARP_INDEX + blockIdx.x * 2; index < cleanupSize;
         index += gridDim.x * 2) {
        int32_t target = devIncompletes[index * 2];
        const uint32_t openIndex = devIncompletes[index * 2 + 1];
        target += UnpackageFlag(reductions[openIndex / PART_SIZE]);

        // Warp significantly faster than single
        uint32_t i = getLaneId() + 1 + openIndex / PART_SIZE;
        while (i - getLaneId() < threadBlocks) {
            const bool inBounds = i < threadBlocks;
            const int32_t min = inBounds ? minimums[i] : 0;
            const int32_t blockSearch = target - (inBounds ? UnpackageFlag(reductions[i - 1]) : 0);
            const uint32_t blockBallot = __ballot_sync(0xffffffff, min <= blockSearch && inBounds);
            if (blockBallot) {
                const uint32_t blockPeer = __ffs(blockBallot) - 1;
                const uint32_t blockIndex = __shfl_sync(0xffffffff, i, blockPeer);
                const int32_t blockTarget = __shfl_sync(0xffffffff, blockSearch, blockPeer);
                const uint32_t leafOffset = blockIndex * WARP_LEAVES;
                for (uint32_t j = getLaneId(); j < WARP_LEAVES; j += LANE_COUNT) {
                    const int32_t leafSearch = blockTarget - leafRed[j + leafOffset];
                    const uint32_t leafBallot =
                        __ballot_sync(0xffffffff, leafMin[j + leafOffset] <= leafSearch);
                    if (leafBallot) {
                        const uint32_t leafPeer = __ffs(leafBallot) - 1;
                        const uint32_t leafIndex = __shfl_sync(0xffffffff, j, leafPeer);
                        const int32_t leafTarget = __shfl_sync(0xffffffff, leafSearch, leafPeer);
                        const uint32_t ii =
                            getLaneId() + leafIndex * LANE_COUNT + blockIndex * PART_SIZE;
                        const uint32_t val = data[ii];
                        const int32_t depth = InclusiveWarpScan(val == 1 ? 1 : -1);
                        const uint32_t finBallot = __ballot_sync(0xffffffff, leafTarget == depth);
                        if (getLaneId() == __ffs(finBallot) - 1) {
                            data[ii] = openIndex;
                        }
                        break;
                    }
                }
                break;
            } else {
                i += LANE_COUNT;
            }
        }

        // Single thread
        //  if(getLaneId() == 0){
        //      uint32_t i =  openIndex / PART_SIZE + 1;
        //      while(i < threadBlocks){
        //          const int32_t min = minimums[i];
        //          const int32_t tarTarget = target - UnpackageFlag(reductions[i - 1]);
        //          if(min <= tarTarget){
        //              const uint32_t offset = i * WARP_LEAVES;
        //              #pragma unroll
        //              for(uint32_t j = 0; j < WARP_LEAVES; ++j){
        //                  const int32_t adj = tarTarget - leafRed[j + offset];
        //                  if(leafMin[j + offset] <= adj){
        //                      int32_t depth = 0;
        //                      #pragma unroll
        //                      for(uint32_t k = 0; k < LANE_COUNT; ++k){
        //                          const uint32_t ii = k + j * LANE_COUNT + i * PART_SIZE;
        //                          const uint32_t val = data[ii];
        //                          depth += (val == 1 ? 1 : -1);
        //                          if(adj == depth){
        //                              data[ii] = openIndex;
        //                              break;
        //                          }
        //                      }
        //                      break;
        //                  }
        //              }
        //              break;
        //          } else {
        //              ++i;
        //          }
        //      }
        //  }
    }
}

bool Validate(uint32_t* hostData, uint32_t* data, const uint32_t size) {
    uint32_t* err;
    uint32_t* expected;
    cudaMalloc(&err, 2 * sizeof(uint32_t));
    cudaMalloc(&expected, size * sizeof(uint32_t));
    cudaMemset(err, 0, 2 * sizeof(uint32_t));
    cudaMemcpy(expected, hostData, size * sizeof(uint32_t), cudaMemcpyHostToDevice);
    ValidateKernel<<<256, 256>>>(data, expected, err, size);
    uint32_t isValid[2];
    cudaMemcpy(&isValid, err, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(err);
    cudaFree(expected);
    if (isValid[0]) {
        printf("Test failed %u errors encountered!\n", isValid[0]);
        double avgDist = (double)isValid[1] / isValid[0];
        printf("Average distance between error parenthesis: %f\n", avgDist);
    }
    return !isValid;
}

int main() {
    constexpr uint32_t size = 1 << 18;
    constexpr uint32_t maxDepth = size / 2;
    constexpr uint32_t warps = 8;
    constexpr uint32_t perThread = 8;
    constexpr uint32_t blockSize = warps * LANE_COUNT;
    constexpr uint32_t partSize = blockSize * perThread;
    constexpr uint32_t threadBlocks = (size + partSize - 1) / partSize;
    constexpr uint32_t batchCount = 500;

    uint32_t* data;
    uint32_t* bump;
    int16_t* leafRed;
    int8_t* leafMin;
    uint32_t* reductions;
    int32_t* minimums;
    int32_t* devIncompletes;
    uint32_t* hostData = new uint32_t[size];

    cudaMalloc(&data, size * sizeof(uint32_t));
    cudaMalloc(&bump, sizeof(uint32_t) * 2);
    cudaMalloc(&leafRed, sizeof(int16_t) * threadBlocks * partSize / LANE_COUNT);
    cudaMalloc(&leafMin, sizeof(int8_t) * threadBlocks * partSize / LANE_COUNT);
    cudaMalloc(&devIncompletes, sizeof(int32_t) * size);  // Wellformed so max is 1 / 2
    cudaMalloc(&reductions, sizeof(uint32_t) * threadBlocks);
    cudaMalloc(&minimums, sizeof(int32_t) * threadBlocks);
    checkCudaError();

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float totalTime = 0.0f;

    for (uint32_t i = 0; i < batchCount; ++i) {
        InitHost(hostData, size, maxDepth, i + 10);
        cudaMemcpy(data, hostData, size * sizeof(uint32_t), cudaMemcpyHostToDevice);
        checkCudaError();

        if (!GetHostSolution(hostData, size)) {
            printf("Err malformed input\n");
            return EXIT_FAILURE;
        }

        cudaMemset(bump, 0, sizeof(uint32_t) * 2);
        cudaMemset(reductions, 0, sizeof(uint32_t) * threadBlocks);

        cudaDeviceSynchronize();
        cudaEventRecord(start);
        DeviceMatch<partSize, warps, warps * perThread, perThread><<<threadBlocks, blockSize>>>(
            data, bump, reductions, minimums, leafRed, leafMin, devIncompletes, size);
        Cleanup<partSize, warps * perThread><<<4096, 64>>>(
            data, devIncompletes, reductions, minimums, leafRed, leafMin, bump, threadBlocks);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        Validate(hostData, data, size);

        float millis;
        cudaEventElapsedTime(&millis, start, stop);
        totalTime += millis;
    }
    printf("\n");
    totalTime /= 1000.0f;
    printf("Total time elapsed: %f\n", totalTime);
    printf("Estimated speed at %u 32-bit elements: %E keys/sec\n\n", size,
           size / totalTime * batchCount);
    checkCudaError()

        cudaFree(data);
    cudaFree(bump);
    cudaFree(leafRed);
    cudaFree(leafMin);
    cudaFree(reductions);
    cudaFree(minimums);
    cudaFree(devIncompletes);
    delete[] hostData;
    return EXIT_SUCCESS;
}