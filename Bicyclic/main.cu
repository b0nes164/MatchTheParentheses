/******************************************************************************
 * Bicyclic Monoid Parentheses Matcher
 *
 * SPDX-License-Identifier: MIT
 * Author: Thomas Smith 11/14/2024
 * 
 * This is mostly translation of the work by Raph Levien:
 *          https://github.com/linebender/vello/pull/140
 *          https://raphlinus.github.io/gpu/2020/09/05/stack-monoid.html
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

#define EMPTY_SENT 0xffffffff

uint32_t LCGTausworth(uint4& t) {
    t.x = ((t.x & 4294967294U) << 12) ^ (((t.x << 13) ^ t.x) >> 19);
    t.y = ((t.y & 4294967288U) << 4) ^ (((t.y << 2) ^ t.y) >> 25);
    t.z = ((t.z & 4294967280U) << 17) ^ (((t.z << 3) ^ t.z) >> 11);
    t.w = t.w * 1664525 + 1013904223U;
    return t.x ^ t.y ^ t.z ^ t.w;
}

void InitHost(uint32_t* hostData, const uint32_t size, const uint32_t maxDepth,
              const uint32_t seed) {
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
        const uint32_t in = hostData[i];
        if (t.empty()) {
            hostData[i] = EMPTY_SENT;
        } else {
            hostData[i] = t.top();
        }

        if (in == 0) {
            t.pop();
        } else {
            t.push(i);
        }
    }

    return t.empty();
}

__global__ void ValidateKernel(uint32_t* data, uint32_t* expected, uint32_t* err,
                               const uint32_t size) {
    for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < size;
         i += blockDim.x * gridDim.x) {
        const uint32_t e = expected[i];
        if (e != EMPTY_SENT && e != data[i]) {
            // printf("Error at %u expected: %u, got: %u\n", i, expected[i], data[i]);
            atomicAdd(&err[0], 1);
            atomicAdd(&err[1], i - expected[i]);
        }
    }
}

template <uint32_t PART_SIZE, uint32_t WARPS>
__global__ void StackReduce(uint32_t* dataIn, uint32_t* devStack, Bic* reduce) {
    __shared__ Bic s_red[WARPS];
    const uint32_t val = dataIn[threadIdx.x + blockIdx.x * PART_SIZE];
    Bic bic = Bic{1 - val, val};
    bic = IncDescCircShiftBic(bic);
    if (getLaneId() == LANE_MASK) {
        s_red[WARP_INDEX] = bic;
    }
    __syncthreads();

    if (threadIdx.x < LANE_COUNT) {
        const bool pred = threadIdx.x < WARPS;
        const Bic t = InclusiveDescendingWarpScanBic(pred ? s_red[threadIdx.x] : Bic{0, 0});
        if (pred) {
            s_red[threadIdx.x] = t;
        }
    }
    __syncthreads();

    if (!threadIdx.x) {
        reduce[blockIdx.x] = s_red[0];
    }

    const uint32_t size = s_red[0].b;
    bic = bic_combine(getLaneId() == LANE_MASK ? Bic{0, 0} : bic,
                      threadIdx.x + LANE_COUNT < blockDim.x ? s_red[WARP_INDEX + 1] : Bic{0, 0});

    if (val == 1 && bic.a == 0) {
        devStack[blockIdx.x * PART_SIZE + size - bic.b - 1] = threadIdx.x + blockIdx.x * PART_SIZE;
    }
}

template <uint32_t PART_SIZE, uint32_t WARPS, uint32_t LG_SIZE>
__global__ void StackLeaf(uint32_t* data_in, uint32_t* dataOut, uint32_t* devStack, Bic* reduce) {
    __shared__ Bic s_red[WARPS];
    __shared__ Bic s_bic[PART_SIZE * 2 - 2];
    __shared__ uint32_t s_stack[PART_SIZE];

    // Reverse scan down the previous reductions
    Bic bicRed = threadIdx.x < blockIdx.x ? reduce[threadIdx.x] : Bic{0, 0};
    bicRed = InclusiveDescendingWarpScanBic(bicRed);
    if (!getLaneId()) {
        s_red[WARP_INDEX] = bicRed;
    }
    __syncthreads();

    if (threadIdx.x < LANE_COUNT) {
        const bool pred = threadIdx.x < WARPS;
        const Bic t = InclusiveDescendingWarpScanBic(pred ? s_red[threadIdx.x] : Bic{0, 0});
        if (pred) {
            s_red[threadIdx.x] = t;
        }
    }
    __syncthreads();

    bicRed = bic_combine(bicRed,
                         threadIdx.x + LANE_COUNT < blockDim.x ? s_red[WARP_INDEX + 1] : Bic{0, 0});
    s_bic[threadIdx.x] = bicRed;
    __syncthreads();

    // Scan is backwards so search target is also inverted
    const uint32_t sp = PART_SIZE - threadIdx.x - 1;
    uint32_t ix = 0;
    #pragma unroll
    for (uint32_t i = 0; i < LG_SIZE; ++i) {
        uint32_t m = ix + (PART_SIZE >> i + 1);
        if (sp < s_bic[m].b) {
            ix = m;
        }
    }

    uint32_t b = s_bic[ix].b;
    if (sp < b) {
        s_stack[threadIdx.x] = devStack[ix * PART_SIZE + b - sp - 1];
    }
    __syncthreads();

    uint32_t val = data_in[threadIdx.x + blockIdx.x * PART_SIZE];
    Bic bic = Bic{1 - val, val};
    s_bic[threadIdx.x] = bic;

    // reduce
    uint32_t inBase = 0;
    #pragma unroll
    for (uint32_t i = 0; i < LG_SIZE - 1; ++i) {
        const uint32_t outBase = 2 * blockDim.x - (1 << LG_SIZE - i);
        __syncthreads();
        if (threadIdx.x < (1 << LG_SIZE - i - 1)) {
            s_bic[threadIdx.x + outBase] =
                bic_combine(s_bic[inBase + threadIdx.x * 2], s_bic[inBase + threadIdx.x * 2 + 1]);
        }
        inBase = outBase;
    }
    __syncthreads();

    ix = threadIdx.x;
    uint32_t j = 0;
    bic = Bic{0, 0};
    while (j < LG_SIZE) {
        const uint32_t base = 2 * blockDim.x - (2 << LG_SIZE - j);
        if (((ix >> j) & 1) != 0) {
            Bic test = bic_combine(s_bic[base + (ix >> j) - 1], bic);
            if (test.b > 0) {
                break;
            }
            bic = test;
            ix -= 1 << j;
        }
        j++;
    }

    if (ix > 0) {
        while (j > 0) {
            j--;
            uint32_t base = 2 * blockDim.x - (2u << LG_SIZE - j);
            Bic test = bic_combine(s_bic[base + (ix >> j) - 1], bic);
            if (test.b == 0) {
                bic = test;
                ix -= 1 << j;
            }
        }
    }

    dataOut[threadIdx.x + blockIdx.x * PART_SIZE] =
        ix > 0 ? blockIdx.x * PART_SIZE + ix - 1 : s_stack[PART_SIZE - 1 - bic.a];
}

template <uint32_t PART_SIZE, uint32_t LG_PART_SIZE, uint32_t WARPS, uint32_t PER_THREAD,
          uint32_t MAX_THREAD_BLOCKS, uint32_t LG_MAX_THREAD_BLOCKS>
__global__ void SinglePass(uint32_t* dataIn, uint32_t* dataOut, volatile uint32_t* devStack,
                           volatile uint32_t* bump, volatile uint32_t* reduce) {
    __shared__ Bic s_red[WARPS];
    __shared__ Bic s_bic[PART_SIZE * 2];
    __shared__ uint32_t s_stack[PART_SIZE];
    __shared__ uint32_t s_broadcast;

    if (!threadIdx.x) {
        s_broadcast = atomicAdd((uint32_t*)&bump[0], 1);
    }
    __syncthreads();
    const uint32_t partIndex = s_broadcast;

    uint32_t val[PER_THREAD];
    #pragma unroll
    for (uint32_t i = getLaneId() + WARP_INDEX * LANE_COUNT * PER_THREAD + partIndex * PART_SIZE,
                  k = 0;
         k < PER_THREAD; i += LANE_COUNT, ++k) {
        val[k] = dataIn[i];
    }

    // Reverse scan
    {
        Bic bicRed[PER_THREAD];
        bicRed[PER_THREAD - 1] =
            IncDescCircShiftBic(Bic{1 - val[PER_THREAD - 1], val[PER_THREAD - 1]});
        #pragma unroll
        for (int32_t k = PER_THREAD - 2; k >= 0; --k) {
            bicRed[k] = IncDescCircShiftBic(Bic{1 - val[k], val[k]});
            const uint32_t a = __shfl_sync(0xffffffff, bicRed[k + 1].a, LANE_MASK);
            const uint32_t b = __shfl_sync(0xffffffff, bicRed[k + 1].b, LANE_MASK);
            bicRed[k] = bic_combine(bicRed[k], Bic{a, b});
        }

        if (getLaneId() == LANE_MASK) {
            s_red[WARP_INDEX] = bicRed[0];
        }
        __syncthreads();

        if (threadIdx.x < LANE_COUNT) {
            const bool pred = threadIdx.x < WARPS;
            const Bic t = InclusiveDescendingWarpScanBic(pred ? s_red[threadIdx.x] : Bic{0, 0});
            if (pred) {
                s_red[threadIdx.x] = t;
            }
        }
        __syncthreads();

        const uint32_t size = s_red[0].b;
        const bool pred = getLaneId() == LANE_MASK;
        const Bic prev = threadIdx.x + LANE_COUNT < blockDim.x ? s_red[WARP_INDEX + 1] : Bic{0, 0};
        #pragma unroll
        for (uint32_t
                 i = getLaneId() + WARP_INDEX * LANE_COUNT * PER_THREAD + partIndex * PART_SIZE,
                 k = 0;
             k < PER_THREAD; i += LANE_COUNT, ++k) {
            bicRed[k] = bic_combine(
                pred ? (k != PER_THREAD - 1 ? bicRed[k + 1] : Bic{0, 0}) : bicRed[k], prev);
            if (val[k] && !bicRed[k].a) {
                atomicExch((uint32_t*)&devStack[partIndex * PART_SIZE + size - bicRed[k].b - 1], i);
            }
        }
        __threadfence();

        if (!threadIdx.x) {
            atomicExch((uint32_t*)&reduce[partIndex * 2], s_red[0].a << 1 | FLAG_READY);
            atomicExch((uint32_t*)&reduce[partIndex * 2 + 1], s_red[0].b << 1 | FLAG_READY);
        }
    }

    // Tiered reduce followed by tracing backlinks
    Bic bic[PER_THREAD];
    uint32_t trace[PER_THREAD];
    #pragma unroll
    for (uint32_t i = getLaneId() + WARP_INDEX * LANE_COUNT * PER_THREAD, k = 0; k < PER_THREAD;
         i += LANE_COUNT, ++k) {
        bic[k] = Bic{1 - val[k], val[k]};
        trace[k] = i;
        s_bic[i] = bic[k];
    }

    {
        uint32_t inBase[PER_THREAD];
        #pragma unroll
        for (uint32_t k = 0; k < PER_THREAD; ++k) {
            inBase[k] = 0;
        }

        #pragma unroll
        for (uint32_t i = 0; i < LG_PART_SIZE - 1; ++i) {
            const uint32_t outBase = 2 * PART_SIZE - (1 << LG_PART_SIZE - i);
            const uint32_t t = 1 << LG_PART_SIZE - i - 1;
            __syncthreads();

            #pragma unroll
            for (uint32_t j = getLaneId() + WARP_INDEX * LANE_COUNT * PER_THREAD, k = 0;
                 k < PER_THREAD; j += LANE_COUNT, ++k) {
                if (j < t) {
                    s_bic[j + outBase] =
                        bic_combine(s_bic[inBase[k] + j * 2], s_bic[inBase[k] + j * 2 + 1]);
                }
                inBase[k] = outBase;
            }
        }
        __syncthreads();

        // This loop ordering is possibly suboptimal
        #pragma unroll
        for (uint32_t k = 0; k < PER_THREAD; ++k) {
            bic[k] = Bic{0, 0};
            uint32_t j = 0;
            while (j < LG_PART_SIZE) {
                const uint32_t base = 2 * PART_SIZE - (2 << LG_PART_SIZE - j);
                if (((trace[k] >> j) & 1) != 0) {
                    Bic test = bic_combine(s_bic[base + (trace[k] >> j) - 1], bic[k]);
                    if (test.b > 0) {
                        break;
                    }
                    bic[k] = test;
                    trace[k] -= 1 << j;
                }
                j++;
            }

            if (trace[k] > 0) {
                while (j > 0) {
                    j--;
                    uint32_t base = 2 * PART_SIZE - (2u << LG_PART_SIZE - j);
                    Bic test = bic_combine(s_bic[base + (trace[k] >> j) - 1], bic[k]);
                    if (test.b == 0) {
                        bic[k] = test;
                        trace[k] -= 1 << j;
                    }
                }
            }
        }
    }

    // Read in the reductions, then reverse scan.
    // Fence guarantees that ready reductions also signal ready stack.
    {
        uint32_t prevA;
        uint32_t prevB;
        if (threadIdx.x < partIndex) {
            uint32_t index = threadIdx.x * 2;
            while (true) {
                const uint32_t payload = reduce[index];
                if (payload & 1) {
                    prevA = payload >> 1;
                    break;
                }
            }

            index += 1;
            while (true) {
                const uint32_t payload = reduce[index];
                if (payload & 1) {
                    prevB = payload >> 1;
                    break;
                }
            }
        }

        Bic bicSpine =
            InclusiveDescendingWarpScanBic(threadIdx.x < partIndex ? Bic{prevA, prevB} : Bic{0, 0});
        if (!getLaneId()) {
            s_red[WARP_INDEX] = bicSpine;
        }
        __syncthreads();

        if (threadIdx.x < LANE_COUNT) {
            const bool pred = threadIdx.x < WARPS;
            const Bic t = InclusiveDescendingWarpScanBic(pred ? s_red[threadIdx.x] : Bic{0, 0});
            if (pred) {
                s_red[threadIdx.x] = t;
            }
        }
        __syncthreads();

        bicSpine = bic_combine(
            bicSpine, threadIdx.x + LANE_COUNT < blockDim.x ? s_red[WARP_INDEX + 1] : Bic{0, 0});

        if (threadIdx.x < MAX_THREAD_BLOCKS) {
            s_bic[threadIdx.x] = bicSpine;
        }
        __syncthreads();

        #pragma unroll
        for (int32_t sp = PART_SIZE - threadIdx.x - 1, k = 0; k < PER_THREAD;
             sp -= blockDim.x, ++k) {
            uint32_t ix = 0;
            #pragma unroll
            for (uint32_t i = 0; i < LG_MAX_THREAD_BLOCKS; ++i) {
                uint32_t m = ix + (MAX_THREAD_BLOCKS >> i + 1);
                if (sp < s_bic[m].b) {
                    ix = m;
                }
            }

            uint32_t b = s_bic[ix].b;
            if (sp < b) {
                s_stack[threadIdx.x + blockDim.x * k] = devStack[ix * PART_SIZE + b - sp - 1];
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (uint32_t i = getLaneId() + WARP_INDEX * LANE_COUNT * PER_THREAD + partIndex * PART_SIZE,
                  k = 0;
         k < PER_THREAD; i += LANE_COUNT, ++k) {
        dataOut[i] =
            trace[k] > 0 ? partIndex * PART_SIZE + trace[k] - 1 : s_stack[PART_SIZE - 1 - bic[k].a];
    }
}

bool Validate(uint32_t* hostBraces, uint32_t* devBraces, const uint32_t size) {
    uint32_t* err;
    uint32_t* expected;
    cudaMalloc(&err, 2 * sizeof(uint32_t));
    cudaMalloc(&expected, size * sizeof(uint32_t));
    cudaMemset(err, 0, 2 * sizeof(uint32_t));
    cudaMemcpy(expected, hostBraces, size * sizeof(uint32_t), cudaMemcpyHostToDevice);
    ValidateKernel<<<256, 256>>>(devBraces, expected, err, size);
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
    constexpr uint32_t size = 1 << 18;  //CANNOT EXCEED 1 << 18
    constexpr uint32_t maxDepth = size / 2;
    constexpr uint32_t warps = 16;
    constexpr uint32_t perThread =
        2;  // WARNING TWO PASS METHOD DOES NOT SUPPORT MORE THAN 1 PER THREAD
    constexpr uint32_t blockSize = warps * LANE_COUNT;

    constexpr uint32_t partSize = blockSize * perThread;
    constexpr uint32_t lgPartSize = 10;  // IF THIS IS NOT EXACTLY LOG2 OF PART SIZE WILL BREAK!!!

    constexpr uint32_t threadBlocks = (size + partSize - 1) / partSize;
    constexpr uint32_t maxTblocks = 512;  // MUST MATCH EXACTLY
    constexpr uint32_t lgMaxTblocks = 9;  // WITH THIS
    constexpr uint32_t batchCount = 500;

    uint32_t* hostData = new uint32_t[size];

    uint32_t* bump;
    uint32_t* dataIn;
    uint32_t* dataOut;
    Bic* reduce;
    uint32_t* devStack;

    cudaMalloc(&bump, sizeof(uint32_t));
    cudaMalloc(&dataIn, size * sizeof(uint32_t));
    cudaMalloc(&dataOut, size * sizeof(uint32_t));
    cudaMalloc(&reduce, threadBlocks * sizeof(uint32_t) * 2);
    cudaMalloc(&devStack, threadBlocks * partSize * sizeof(uint32_t));
    checkCudaError();

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float totalTime = 0.0f;

    for (uint32_t i = 0; i < batchCount; ++i) {
        InitHost(hostData, size, maxDepth, i + 10);
        cudaMemcpy(dataIn, hostData, size * sizeof(uint32_t), cudaMemcpyHostToDevice);
        checkCudaError();

        if (!GetHostSolution(hostData, size)) {
            printf("Err malformed input\n");
            return EXIT_FAILURE;
        }

        cudaMemset(devStack, 0, threadBlocks * partSize * sizeof(uint32_t));
        cudaMemset(reduce, 0, threadBlocks * sizeof(uint32_t) * 2);
        cudaMemset(bump, 0, sizeof(uint32_t));
        cudaDeviceSynchronize();
        cudaEventRecord(start);

        // StackReduce<partSize, warps>
        //     <<<threadBlocks, warps * LANE_COUNT>>>(dataIn, devStack, reduce);
        // StackLeaf<partSize, warps, 9>
        //     <<<threadBlocks, warps * LANE_COUNT>>>(dataIn, dataOut, devStack, reduce);

        SinglePass<partSize, lgPartSize, warps, perThread, maxTblocks, lgMaxTblocks>
            <<<threadBlocks, warps * LANE_COUNT>>>(dataIn, dataOut, devStack, bump,
                                                   (uint32_t*)reduce);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        Validate(hostData, dataOut, size);

        float millis;
        cudaEventElapsedTime(&millis, start, stop);
        totalTime += millis;
    }
    checkCudaError();
    printf("\n");
    totalTime /= 1000.0f;
    printf("Total time elapsed: %f\n", totalTime);
    printf("Estimated speed at %u 32-bit elements: %E keys/sec\n\n", size,
           size / totalTime * batchCount);

    cudaFree(bump);
    cudaFree(dataIn);
    cudaFree(dataOut);
    cudaFree(reduce);
    cudaFree(devStack);
    delete[] hostData;
    return EXIT_SUCCESS;
}