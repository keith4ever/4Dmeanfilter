/*
 * Copyright (c) 2014 -     Keith Ha (keith4ever@gmail.com)
 * All content herein is protected by U.S. copyright and other applicable intellectual property laws
 * and may not be copied without the expressive permission of Keith Ha, who reserves all rights.
 * Reuse of any of the content for any purpose without the permission of Keith Ha
 * is strictly and expressively prohibited.
 */

#pragma once

#include <chrono>
#include <vector_types.h>
#include <memory.h>
#include <math.h>
#include <iostream>
#include <driver_types.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include "clionCudaParse.h"

#define __PerfTimerStart__	auto startT = chrono::steady_clock::now();
#define __PerfTimerEnd__	auto endT = chrono::steady_clock::now(); \
	double elapsedSeconds = ((endT - startT).count()) * chrono::steady_clock::period::num \
	/ static_cast<double>(chrono::steady_clock::period::den); \
	cout << "Elapsed time: " << (elapsedSeconds*1000) << " msecs.." << endl;

#define __cu(a) do { \
    cudaError_t  ret; \
    if ((ret = (a)) != cudaSuccess) { \
        fprintf(stderr, "%s has returned CUDA error %d\n", #a, ret); \
        return cudaErrorInvalidValue;\
    }} while(0)

#define MIN(a,b) (((a)<(b))?(a):(b))

extern void initVars_wrap(float* expBuffer);
extern void meanFilteredTensor_wrap(float* inputTensor, float* outputTensor,
                                    int d1, int d2, int d3, int d4);
class CalcMeanfilter4D{
private:
    int m_d1dim;	// 128
    int m_d2dim;	// 128
    int m_d3dim;	// 128
    int m_d4dim;	// 128

    float *m_pdInbuffer;
    float *m_pdOutbuffer;
    float *m_pdExpbuffer;

public:
    __host__ __device__ CalcMeanfilter4D(int d1, int d2, int d3, int d4);
    __host__ __device__ void setBuffers(float *pIn, float *pOut, float* pExp);
    __host__ __device__ void computeMean(int i, int j, int k);
    __host__ __device__ void memcpyToExpBuf(int idxX, int idxY, int idxZ);
    __host__ __device__ int  boundCheck(int idx, int dim);
};
