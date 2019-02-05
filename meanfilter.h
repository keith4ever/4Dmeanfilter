/*
 * Copyright (c) 2014 -     Keith Ha (keith4ever@gmail.com)
 * All content herein is protected by U.S. copyright and other applicable intellectual property laws
 * and may not be copied without the expressive permission of Keith Ha., which reserves all rights.
 * Reuse of any of the content for any purpose without the permission of Keith Ha
 * is strictly and expressively prohibited.
 */

#pragma once

#include <chrono>
#include <stdio.h>
#include <iostream>
#include <memory>
#include <string.h>
#include <unistd.h>
#include <cuda.h>

#define D1DIM   2
#define D2DIM   2
#define D3DIM   5
#define D4DIM   2

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

using namespace std;

class Config{
private:
    bool m_bCPUCompute;

public:
    Config();
    ~Config();
    void printHelp(bool bInitSuccess);
    bool parseArguments(int argc, char** argv);
    void checkFileAndRead(float* buf, int bufsize, char* file);
    bool isCPUCompute() { return m_bCPUCompute; }
};
