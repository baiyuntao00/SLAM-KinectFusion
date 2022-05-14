#pragma once
#include "cuda_runtime_api.h"
#include <iostream>
#define cudaSafeCall(expr) kf::___cudaSafeCall(expr, __FILE__, __LINE__)

namespace kf
{
    static inline void ___cudaSafeCall(cudaError_t err, const char *file, const int line, const char *func = "")
    {
        if (cudaSuccess != err)
        {
            std::cout << "cuda error: " << cudaGetErrorString(err) << "\t" << file << ":" << line << std::endl;
        }
    };

}
