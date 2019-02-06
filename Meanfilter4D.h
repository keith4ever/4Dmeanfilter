/*
 * Copyright (c) 2014 -     Keith Ha (keith4ever@gmail.com)
 * All content herein is protected by U.S. copyright and other applicable intellectual property laws
 * and may not be copied without the expressive permission of Keith Ha, who reserves all rights.
 * Reuse of any of the content for any purpose without the permission of Keith Ha
 * is strictly and expressively prohibited.
 */

#ifndef MEANFILTER_MEANFILTER4D_H
#define MEANFILTER_MEANFILTER4D_H

#include "kernel.h"
#include "ThreadManager.h"

#define SAFECUDADELETE(x)	if(x){ __cu(cudaFree(x)); x = NULL; }

class Meanfilter4D
{
private:

    float   *m_pdInbuffer;
    float   *m_pdOutbuffer;
    float   *m_pdExpbuffer;

    int*    m_pThreadIdx[MAX_THREAD_NUM];
    bool    m_bCPUCompute;
    int     m_numThreads;
    Barrier m_barrier;

public:
    int m_d1dim;	// 128
    int m_d2dim;	// 128
    int m_d3dim;	// 128
    int m_d4dim;	// 128

    Meanfilter4D();
    ~Meanfilter4D();
    cudaError_t init(int d1, int d2, int d3, int d4, bool bCPU);
    cudaError_t deinit();
    float*  getInBuffer()    { return m_pdInbuffer; }
    float*  getOutBuffer()   { return m_pdOutbuffer; }
    float*  getExpBuffer()   { return m_pdExpbuffer; }
    int     getNumThreads()  { return m_numThreads; }

    cudaError_t execute(float *inbuf, float *outbuf);
    void printCudaDevProp();
    void printOut(float *pOut);
    void synchonizeThreads()    { return m_barrier.Wait(); }

    template<typename T>
    void compare(T* dst, T* src, int size,
                 const char* title = "[Comparison Result] "){
        int mismatch = 0, temp = 0;
        for (int i = 0; i < size; i++) {
            if (dst[i] != src[i])
                mismatch++;
        }
        std::cout << title << mismatch << std::endl;
    }
};


#endif //MEANFILTER_MEANFILTER4D_H
