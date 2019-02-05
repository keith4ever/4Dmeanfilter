/*
 * Copyright (c) 2014 -     Keith Ha (keith4ever@gmail.com)
 * All content herein is protected by U.S. copyright and other applicable intellectual property laws
 * and may not be copied without the expressive permission of Keith Ha., which reserves all rights.
 * Reuse of any of the content for any purpose without the permission of Keith Ha
 * is strictly and expressively prohibited.
 */

#include "meanfilter.h"
#include "kernel.h"
#include <cuda_runtime.h>

using namespace std;

Meanfilter4D* gpInstance;

__device__ float* d_expTensor = NULL;

__device__ inline int boundCheckCUDA(int idx, int dim){
    int ret = 3;
    if(idx < 0 || idx >= dim) return 0;

    if(idx <= 0) ret--;
    if(idx >= dim-1) ret--;
    return ret;
}

__global__ void initVars(float* expBuffer){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i > 0) return;

    d_expTensor = expBuffer;
}

__global__ void calcMeanfilterCUDA(float* inputTensor, float* outputTensor,
                                   int d1, int d2, int d3, int d4) {
    int idxX = threadIdx.x + blockIdx.x * blockDim.x;
    int idxY = threadIdx.y + blockIdx.y * blockDim.y;
    if (idxX >= d1|| idxY >= d2 || d_expTensor == NULL) return;

    int k, l;
    int wi, ed2, ed3, ed4;
    float tempsum = 0.0f;

    // setting expanded dimension values
    ed2 = d2+2; ed3 = d3+2; ed4 = d4+2;

    int div;

    // first copy into expanded memory
    float *pIn, *pExp, *pBase;
    for (k = 0; k < d3; k++) {
        // copying raw source data into expanded (+2 for each dimension, to handle boundary data)
        pIn = &inputTensor[idxX * d2 * d3 * d4 + idxY * d3 * d4 + k * d4];
        pExp = &d_expTensor[(idxX + 1) * (d2+2) * (d3+2) * (d4+2)
                             + (idxY + 1) * (d3+2) * (d4+2) + (k + 1) * (d4+2) + 1];
        memcpy(pExp, pIn, sizeof(float) * d4);
    }

    // wait until all mem copy is done
    __syncthreads();

    // now compute 4D Tensor mean filter
    for(k = 0; k < d3; k++){
        for(l = 0; l < d4; l++) {
            div = boundCheckCUDA(idxX, d1) * boundCheckCUDA(idxY, d2)
                  * boundCheckCUDA(k, d3) * boundCheckCUDA(l, d4); // 1 to 81
            for (wi = 0; wi < 3; wi++){
                pBase   =  &d_expTensor[(idxX+wi)*ed2*ed3*ed4 + idxY*ed3*ed4];
                tempsum += pBase[k*ed4     + l];
                tempsum += pBase[k*ed4     + l+1];
                tempsum += pBase[k*ed4     + l+2];
                tempsum += pBase[(k+1)*ed4 + l];
                tempsum += pBase[(k+1)*ed4 + l+1];
                tempsum += pBase[(k+1)*ed4 + l+2];
                tempsum += pBase[(k+2)*ed4 + l];
                tempsum += pBase[(k+2)*ed4 + l+1];
                tempsum += pBase[(k+2)*ed4 + l+2];

                pBase   += ed3*ed4;
                tempsum += pBase[k*ed4     + l];
                tempsum += pBase[k*ed4     + l+1];
                tempsum += pBase[k*ed4     + l+2];
                tempsum += pBase[(k+1)*ed4 + l];
                tempsum += pBase[(k+1)*ed4 + l+1];
                tempsum += pBase[(k+1)*ed4 + l+2];
                tempsum += pBase[(k+2)*ed4 + l];
                tempsum += pBase[(k+2)*ed4 + l+1];
                tempsum += pBase[(k+2)*ed4 + l+2];

                pBase   += ed3*ed4;
                tempsum += pBase[k*ed4     + l];
                tempsum += pBase[k*ed4     + l+1];
                tempsum += pBase[k*ed4     + l+2];
                tempsum += pBase[(k+1)*ed4 + l];
                tempsum += pBase[(k+1)*ed4 + l+1];
                tempsum += pBase[(k+1)*ed4 + l+2];
                tempsum += pBase[(k+2)*ed4 + l];
                tempsum += pBase[(k+2)*ed4 + l+1];
                tempsum += pBase[(k+2)*ed4 + l+2];
            }
            outputTensor[idxX*d2*d3*d4 + idxY*d3*d4 + k*d4 + l] = ((div <= 0)? 0 : (tempsum / div));
            //printf("[%d,%d] sum: %f, div: %d\n", idxX, idxY, tempsum, div);
            tempsum = 0;
        }
    }
}

inline int boundCheck(int idx, int dim){
    int ret = 3;
    if(idx < 0 || idx >= dim) return 0;

    if(idx <= 0) ret--;
    if(idx >= dim-1) ret--;
    return ret;
}

void* calcMeanfilterC(void* arg)
{
    int i, j, k, l;
    i = *(int*)arg;

    for(; i < gpInstance->m_d1dim; i += MAX_THREAD_NUM){
        for(j = 0; j < gpInstance->m_d2dim; j++){
            for(k = 0; k < gpInstance->m_d3dim; k++){
                for(l = 0; l < gpInstance->m_d4dim; l++) {
                    gpInstance->calcMidCells(i, j, k, l); // passing expanded buffer index
                }
            }
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

        initVars <<< 1, 1 >>> (m_pdExpbuffer);
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

#define MIN(a,b) (((a)<(b))?(a):(b))

cudaError_t Meanfilter4D::execute(float *inbuf, float *outbuf)
{
    __PerfTimerStart__
    int i, j, k;
    float *pExp, *pIn;

    if(m_bCPUCompute) {
        m_pdInbuffer = inbuf;
        m_pdOutbuffer = outbuf;

        ThreadManager calcMeanThr(MAX_THREAD_NUM);
        calcMeanThr.Init(calcMeanfilterC, (void **) m_pThreadIdx);

        for (i = 0; i < m_d1dim; i++) {
            for (j = 0; j < m_d2dim; j++) {
                for (k = 0; k < m_d3dim; k++) {
                    pIn = &m_pdInbuffer[i * m_d2dim * m_d3dim * m_d4dim + j * m_d3dim * m_d4dim + k * m_d4dim];
                    pExp = &m_pdExpbuffer[(i + 1) * (m_d2dim + 2) * (m_d3dim + 2) * (m_d4dim + 2)
                                          + (j + 1) * (m_d3dim + 2) * (m_d4dim + 2) + (k + 1) * (m_d4dim + 2) + 1];
                    memcpy(pExp, pIn, sizeof(float) * m_d4dim);
                }
            }
        }
        calcMeanThr.Run();
        calcMeanThr.Join();
    } else {
        __cu(cudaMemcpy(m_pdInbuffer, inbuf, sizeof(float) * m_d1dim * m_d2dim * m_d3dim * m_d4dim,
                        cudaMemcpyHostToDevice));

        dim3 threads(MIN(m_d1dim, 32), MIN(m_d2dim, 16));
        dim3 grid((int) (m_d1dim + (threads.x - 1)) / threads.x,
                  (int) (m_d2dim + (threads.y - 1)) / threads.y);

        calcMeanfilterCUDA <<< grid, threads >>> (m_pdInbuffer, m_pdOutbuffer, m_d1dim, m_d2dim, m_d3dim, m_d4dim);
        cudaDeviceSynchronize();

        __cu(cudaMemcpy(outbuf, m_pdOutbuffer, sizeof(float) * m_d1dim * m_d2dim * m_d3dim * m_d4dim,
                        cudaMemcpyDeviceToHost));
    }
    __PerfTimerEnd__

    return cudaSuccess;
}

void Meanfilter4D::calcMidCells(int idxX, int idxY, int k, int l) {

    int wi;
    float tempsum = 0.0f;

    int ed2, ed3, ed4, div;
    float* pBase;

    ed2 = m_d2dim + 2;
    ed3 = m_d3dim + 2;
    ed4 = m_d4dim + 2;

    div = boundCheck(idxX, m_d1dim) * boundCheck(idxY, m_d2dim)
          * boundCheck(k, m_d3dim) * boundCheck(l, m_d4dim); // 0 to 4
    for (wi = 0; wi < 3; wi++) {
        //for (wj = -1; wj < 2; wj++) {
        pBase   =  &m_pdExpbuffer[(idxX+wi)*ed2*ed3*ed4 + idxY*ed3*ed4];
        tempsum += pBase[k*ed4   + l];
        tempsum += pBase[k*ed4   + l+1];
        tempsum += pBase[k*ed4   + l+2];
        tempsum += pBase[(k+1)*ed4 + l];
        tempsum += pBase[(k+1)*ed4 + l+1];
        tempsum += pBase[(k+1)*ed4 + l+2];
        tempsum += pBase[(k+2)*ed4 + l];
        tempsum += pBase[(k+2)*ed4 + l+1];
        tempsum += pBase[(k+2)*ed4 + l+2];

        pBase   += ed3*ed4;
        tempsum += pBase[k * ed4 + l];
        tempsum += pBase[k * ed4 + l + 1];
        tempsum += pBase[k * ed4 + l + 2];
        tempsum += pBase[(k + 1) * ed4 + l];
        tempsum += pBase[(k + 1) * ed4 + l + 1];
        tempsum += pBase[(k + 1) * ed4 + l + 2];
        tempsum += pBase[(k + 2) * ed4 + l];
        tempsum += pBase[(k + 2) * ed4 + l + 1];
        tempsum += pBase[(k + 2) * ed4 + l + 2];

        pBase   += ed3*ed4;
        tempsum += pBase[k*ed4   + l];
        tempsum += pBase[k*ed4   + l+1];
        tempsum += pBase[k*ed4   + l+2];
        tempsum += pBase[(k+1)*ed4 + l];
        tempsum += pBase[(k+1)*ed4 + l+1];
        tempsum += pBase[(k+1)*ed4 + l+2];
        tempsum += pBase[(k+2)*ed4 + l];
        tempsum += pBase[(k+2)*ed4 + l+1];
        tempsum += pBase[(k+2)*ed4 + l+2];
    }
    m_pdOutbuffer[idxX*m_d2dim*m_d3dim*m_d4dim + idxY*m_d3dim*m_d4dim + k*m_d4dim + l]
            = tempsum / div;
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

void Meanfilter4D::printCudaDevProp()
{
    int nDevices;

    if(cudaGetDeviceCount(&nDevices) != cudaSuccess)
        return;

    printf("======================================\n");
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("CUDA Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Memory Capacity: %ldMB\n", (prop.totalGlobalMem)>>20);
    }
    printf("======================================\n");
}