/*
 * Copyright (c) 2014 -     Keith Ha (keith4ever@gmail.com)
 * All content herein is protected by U.S. copyright and other applicable intellectual property laws
 * and may not be copied without the expressive permission of Keith Ha, who reserves all rights.
 * Reuse of any of the content for any purpose without the permission of Keith Ha
 * is strictly and expressively prohibited.
 */

#include "meanfilter.h"
#include "kernel.h"

using namespace std;

__device__ float* d_expTensor = NULL;

__global__ void initVars(float* expBuffer){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i > 0) return;

    d_expTensor = expBuffer;
}

__global__ void meanFilteredTensor(float* inputTensor, float* outputTensor,
                                   int d1, int d2, int d3, int d4) {
    int idxX = threadIdx.x + blockIdx.x * blockDim.x;
    int idxY = threadIdx.y + blockIdx.y * blockDim.y;
    if (idxX >= d1|| idxY >= d2 || d_expTensor == NULL) return;

    CalcMeanfilter4D filter(d1, d2, d3, d4);
    filter.setBuffers(inputTensor, outputTensor, d_expTensor);
    filter.memcpyToExpBuf(idxX, idxY);
    // wait until all mem copy is done
    __syncthreads();

#if     1
    int wi;
    int ed2, ed3, ed4;
    int div, divxy, divk;
    int k, l;
    float tempsum = 0.0f;
    float* pBase;
    ed2 = d2 + 2;
    ed3 = d3 + 2;
    ed4 = d4 + 2;

    // now compute 4D Tensor mean filter
    divxy = filter.boundCheck(idxX, d1) * filter.boundCheck(idxY, d2);
    for(k = 0; k < d3; k++){
        divk = filter.boundCheck(k, d3);
        for(l = 0; l < d4; l++) {
            div = divxy * divk * filter.boundCheck(l, d4); // 1 to 81
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
            outputTensor[idxX*d2*d3*d4 + idxY*d3*d4 + k*d4 + l]
                    = ((div <= 0)? 0 : (tempsum / div));
            //printf("[%d,%d] sum: %f, div: %d\n", idxX, idxY, tempsum, div);
            tempsum = 0;
        }
    }
#else
    filter.computeMean(idxX, idxY);
#endif
}

void initVars_wrap(float* expBuffer){
    initVars <<< 1, 1 >>> (expBuffer);
}

void meanFilteredTensor_wrap(float* inputTensor, float* outputTensor,
                                   int d1, int d2, int d3, int d4) {
    dim3 threads(MIN(d1, 32), MIN(d2, 32));
    dim3 grid((int) (d1 + (threads.x - 1)) / threads.x,
              (int) (d2 + (threads.y - 1)) / threads.y);

    meanFilteredTensor <<< grid, threads >>> (inputTensor, outputTensor, d1, d2, d3, d4);
    cudaDeviceSynchronize();
}

__host__ __device__ CalcMeanfilter4D::CalcMeanfilter4D(int d1, int d2, int d3, int d4)
{
    if(d1 <= 0 || d2 <= 0 || d3 <= 0 || d4 <= 0){
        printf("[%s] provided dimension value is <= 0..\n", __func__);
        return;
    }

    m_d1dim = d1;   m_d2dim = d2;
    m_d3dim = d3;   m_d4dim = d4;
}

__host__ __device__ int CalcMeanfilter4D::boundCheck(int idx, int dim){
    int ret = 3;
    if(idx < 0 || idx >= dim) return 0;

    if(idx <= 0) ret--;
    if(idx >= dim-1) ret--;
    return ret;
}

__host__ __device__ void CalcMeanfilter4D::setBuffers(float *pIn, float *pOut, float *pExp)
{
    if(pIn == NULL || pOut == NULL || pExp == NULL){
        printf("[%s] provided buffer pointer is null..\n", __func__);
        return;
    }
    m_pdInbuffer = pIn;
    m_pdOutbuffer = pOut;
    m_pdExpbuffer = pExp;
}


__host__ __device__ void CalcMeanfilter4D::memcpyToExpBuf(int idxX, int idxY)
{
    float *pIn, *pExp, *pdst, *psrc;
    int ed2, ed3, ed4;
    size_t i = 0, k;
    ed2 = m_d2dim + 2;
    ed3 = m_d3dim + 2;
    ed4 = m_d4dim + 2;

    // copying raw source data into expanded (+2 for each dimension, to handle boundary data)
    pIn = &m_pdInbuffer[idxX*m_d2dim*m_d3dim*m_d4dim + idxY*m_d3dim * m_d4dim];
    pExp = &m_pdExpbuffer[(idxX+1)*ed2*ed3*ed4 + (idxY+1)*ed3*ed4 + ed4 + 1];
    for (k = 0; k < m_d3dim; k++) {
        /* pIn = &inputTensor[idxX*d2*d3*d4 + idxY*d3*d4 + k*d4];
        pExp = &d_expTensor[(idxX+1)*ed2*ed3*ed4 + (idxY+1)*ed3*ed4 + (k+1)*ed4 + 1]; */
        i = 0;
        pdst = pExp;    psrc = pIn;
        while(i++ < m_d4dim){
            *pdst++ = *psrc++;
        }
        pIn += m_d4dim;
        pExp += ed4;
    }
}

__host__ __device__ void CalcMeanfilter4D::computeMean(int idxX, int idxY) {
    int wi, ed2, ed3, ed4;
    int div, divxy, divk;
    int k, l;
    float tempsum = 0.0f;
    float* pBase;

    ed2 = m_d2dim + 2;
    ed3 = m_d3dim + 2;
    ed4 = m_d4dim + 2;

    // now compute 4D Tensor mean filter
    divxy = boundCheck(idxX, m_d1dim) * boundCheck(idxY, m_d2dim);
    for(k = 0; k < m_d3dim; k++){
        divk = boundCheck(k, m_d3dim);
        for(l = 0; l < m_d4dim; l++) {
            div = divxy * divk * boundCheck(l, m_d4dim); // 1 to 81
            for (wi = 0; wi < 3; wi++){
                pBase   =  &m_pdExpbuffer[(idxX+wi)*ed2*ed3*ed4 + idxY*ed3*ed4];
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
            m_pdOutbuffer[idxX*m_d2dim*m_d3dim*m_d4dim + idxY*m_d3dim*m_d4dim + k*m_d4dim + l]
                    = ((div <= 0)? 0 : (tempsum / div));
            //printf("[%d,%d] sum: %f, div: %d\n", idxX, idxY, tempsum, div);
            tempsum = 0;
        }
    }
}
