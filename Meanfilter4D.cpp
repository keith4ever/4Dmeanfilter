/*
 * Copyright (c) 2014 -     Keith Ha (keith4ever@gmail.com)
 * All content herein is protected by U.S. copyright and other applicable intellectual property laws
 * and may not be copied without the expressive permission of Keith Ha, who reserves all rights.
 * Reuse of any of the content for any purpose without the permission of Keith Ha
 * is strictly and expressively prohibited.
 */

#include "Meanfilter4D.h"

Meanfilter4D* gpInstance;

void* meanFilteredTensorC(void* arg)
{
    int i, j, k;
    i = *(int*)arg;

    CalcMeanfilter4D filter(gpInstance->m_d1dim, gpInstance->m_d2dim,
                            gpInstance->m_d3dim, gpInstance->m_d4dim);
    filter.setBuffers(gpInstance->getInBuffer(), gpInstance->getOutBuffer(),
                      gpInstance->getExpBuffer());

    for(; i < gpInstance->m_d1dim; i += gpInstance->getNumThreads()) {
        for (j = 0; j < gpInstance->m_d2dim; j++) {
            for (k = 0; k < gpInstance->m_d3dim; k++)
                filter.memcpyToExpBuf(i, j, k);
        }
    }

    gpInstance->synchonizeThreads();

    i = *(int*)arg;
    for(; i < gpInstance->m_d1dim; i += gpInstance->getNumThreads()){
        for(j = 0; j < gpInstance->m_d2dim; j++){
            for (k = 0; k < gpInstance->m_d3dim; k++)
                filter.computeMean(i, j, k);
        }
    }

    return NULL;
}

Meanfilter4D::Meanfilter4D()
{
    m_pdInbuffer = NULL;
    m_pdExpbuffer = NULL;
    m_pdOutbuffer = NULL;
}

Meanfilter4D::~Meanfilter4D()
{

}

cudaError_t Meanfilter4D::init(int d1, int d2, int d3, int d4, bool bCPU)
{
    if(d1 <= 0 || d2 <= 0|| d3 <= 0|| d4 <= 0)
        return cudaErrorInvalidValue;

    m_d1dim         = d1;
    m_d2dim         = d2;
    m_d3dim         = d3;
    m_d4dim         = d4;

    gpInstance      = this;
    cout << "Input data dimensions: [" << m_d1dim << ", " << m_d2dim
    << ", " << m_d3dim << ", " << m_d4dim << "]" << endl;

    m_bCPUCompute = bCPU;
    if(m_bCPUCompute) {
        m_pdExpbuffer = new float[(d1 + 2) * (d2 + 2) * (d3 + 2) * (d4 + 2)];
        memset(m_pdExpbuffer, 0, sizeof(float) * (d1 + 2) * (d2 + 2) * (d3 + 2) * (d4 + 2));
        for (int i = 0; i < MAX_THREAD_NUM; i++) {
            m_pThreadIdx[i] = new int;
            *m_pThreadIdx[i] = i;
        }
    } else {
        printCudaDevProp();
        __cu(cudaMalloc((void **) &m_pdInbuffer, sizeof(float) * d1 * d2 * d3 * d4));
        __cu(cudaMalloc((void **) &m_pdExpbuffer, sizeof(float) * (d1 + 2) * (d2 + 2) * (d3 + 2) * (d4 + 2)));
        __cu(cudaMalloc((void **) &m_pdOutbuffer, sizeof(float) * d1 * d2 * d3 * d4));

        __cu(cudaMemset(m_pdExpbuffer, 0, sizeof(float) * (d1 + 2) * (d2 + 2) * (d3 + 2) * (d4 + 2)));
        __cu(cudaMemset(m_pdOutbuffer, 0, sizeof(float) * d1 * d2 * d3 * d4));
        __cu(cudaMemset(m_pdInbuffer, 0, sizeof(float) * d1 * d2 * d3 * d4));

        initVars_wrap(m_pdExpbuffer);
    }
    return cudaSuccess;
}

cudaError_t Meanfilter4D::deinit() {

    if (m_bCPUCompute) {
        delete[] m_pdExpbuffer;
        m_pdInbuffer = NULL;
        m_pdExpbuffer = NULL;
        m_pdOutbuffer = NULL;
        for (int i = 0; i < MAX_THREAD_NUM; i++) {
            delete m_pThreadIdx[i];
        }
    } else {
        SAFECUDADELETE(m_pdInbuffer);
        SAFECUDADELETE(m_pdOutbuffer);
        SAFECUDADELETE(m_pdExpbuffer);
    }

    return cudaSuccess;
}

cudaError_t Meanfilter4D::execute(float *inbuf, float *outbuf, int streamUnit)
{
    __PerfTimerStart__

    if(m_bCPUCompute) {
        m_pdInbuffer = inbuf;
        m_pdOutbuffer = outbuf;

        m_numThreads = ((m_d1dim > MAX_THREAD_NUM)? MAX_THREAD_NUM : m_d1dim);
        ThreadManager calcMeanThr(m_numThreads);
        calcMeanThr.Init(meanFilteredTensorC, (void **) m_pThreadIdx);
        m_barrier.Init(m_numThreads);
        calcMeanThr.Run();
        calcMeanThr.Join();
    } else {
        meanFilteredTensor_wrap(inbuf, m_pdInbuffer, outbuf, m_pdOutbuffer,
                                streamUnit, m_d1dim, m_d2dim, m_d3dim, m_d4dim);
    }
    __PerfTimerEnd__

    return cudaSuccess;
}

void Meanfilter4D::printOut(float *pOut) {
    for(int i=0; i < m_d1dim; i++) {
        for (int j = 0; j < m_d2dim; j++) {
            for (int k = 0; k < m_d3dim; k++) {
                cout << "[" << i << "," << j << "," << k << ",*] ";
                for (int l = 0; l < m_d4dim; l++) {
                    cout << pOut[i * m_d2dim * m_d3dim * m_d4dim + j * m_d3dim * m_d4dim + k * m_d4dim + l]
                    << ", ";
                }
                cout << endl;
            }
        }
    }
}

void Meanfilter4D::printCudaDevProp() {
    int nDevices;

    if (cudaGetDeviceCount(&nDevices) != cudaSuccess)
        return;

    printf("======================================\n");
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("CUDA Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  CUDA cores: %d\n", prop.multiProcessorCount * 128);
        printf("  Capable of Concurrent memcpy & execution: %d\n", prop.asyncEngineCount);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Memory Capacity: %ldMB\n", (prop.totalGlobalMem) >> 20);
    }
    printf("======================================\n");
}