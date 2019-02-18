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
    int idxZ = threadIdx.z + blockIdx.z * blockDim.z;
    if (idxX >= d1|| idxY >= d2 || idxZ >= d3|| d_expTensor == NULL) return;

    CalcMeanfilter4D filter(d1, d2, d3, d4);
    filter.setBuffers(inputTensor, outputTensor, d_expTensor);
    filter.memcpyToExpBuf(idxX, idxY, idxZ);
    // wait until all mem copy is done
    __syncthreads();

    filter.computeMean(idxX, idxY, idxZ);
}

void initVars_wrap(float* expBuffer){
    initVars <<< 1, 1 >>> (expBuffer);
}

void meanFilteredTensor_wrap(float* inputTensor, float* outputTensor,
                             int d1, int d2, int d3, int d4) {
    dim3 threads(MIN(d1, 32), MIN(d2, 32), MIN(d3, 32));
    dim3 grid((int) (d1 + (threads.x - 1)) / threads.x,
              (int) (d2 + (threads.y - 1)) / threads.y,
              (int) (d3 + (threads.z - 1)) / threads.z);

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


__host__ __device__ void CalcMeanfilter4D::memcpyToExpBuf(int idxX, int idxY, int idxZ)
{
    float *pIn, *pExp, *pdst, *psrc;
    int ed2, ed3, ed4;
    size_t i = 0;
    ed2 = m_d2dim + 2;
    ed3 = m_d3dim + 2;
    ed4 = m_d4dim + 2;

    // copying raw source data into expanded (+2 for each dimension, to handle boundary data)
    pIn = &m_pdInbuffer[idxX*m_d2dim*m_d3dim*m_d4dim + idxY*m_d3dim*m_d4dim + idxZ*m_d4dim];
    pExp = &m_pdExpbuffer[(idxX+1)*ed2*ed3*ed4 + (idxY+1)*ed3*ed4 + (idxZ+1)*ed4 + 1];
    pdst = pExp;    psrc = pIn;
    while(i++ < m_d4dim){
        *pdst++ = *psrc++;
    }
}

__host__ __device__ void CalcMeanfilter4D::computeMean(int idxX, int idxY, int idxZ) {
    int wi, ed2, ed3, ed4;
    int div, divxyz;
    int l;
    float sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    float* pBase;

    ed2 = m_d2dim + 2;
    ed3 = m_d3dim + 2;
    ed4 = m_d4dim + 2;

    // now compute 4D Tensor mean filter
    divxyz = boundCheck(idxX, m_d1dim) * boundCheck(idxY, m_d2dim) * boundCheck(idxZ, m_d3dim);
    for(l = 0; l < m_d4dim; l++) {
        div = divxyz * boundCheck(l, m_d4dim); // 1 to 81
        for (wi = 0; wi < 3; wi++){
            pBase   =  &m_pdExpbuffer[(idxX+wi)*ed2*ed3*ed4 + idxY*ed3*ed4 + idxZ*ed4];
            sum1 += pBase[l];
            sum2 += pBase[l+1];
            sum3 += pBase[l+2];
            sum1 += pBase[ed4 + l];
            sum2 += pBase[ed4 + l+1];
            sum3 += pBase[ed4 + l+2];
            sum1 += pBase[2*ed4 + l];
            sum2 += pBase[2*ed4 + l+1];
            sum3 += pBase[2*ed4 + l+2];

            pBase   += ed3*ed4;
            sum1 += pBase[l];
            sum2 += pBase[l+1];
            sum3 += pBase[l+2];
            sum1 += pBase[ed4 + l];
            sum2 += pBase[ed4 + l+1];
            sum3 += pBase[ed4 + l+2];
            sum1 += pBase[2*ed4 + l];
            sum2 += pBase[2*ed4 + l+1];
            sum3 += pBase[2*ed4 + l+2];

            pBase   += ed3*ed4;
            sum1 += pBase[l];
            sum2 += pBase[l+1];
            sum3 += pBase[l+2];
            sum1 += pBase[ed4 + l];
            sum2 += pBase[ed4 + l+1];
            sum3 += pBase[ed4 + l+2];
            sum1 += pBase[2*ed4 + l];
            sum2 += pBase[2*ed4 + l+1];
            sum3 += pBase[2*ed4 + l+2];
        }
        m_pdOutbuffer[idxX*m_d2dim*m_d3dim*m_d4dim + idxY*m_d3dim*m_d4dim + idxZ*m_d4dim + l]
                = ((div <= 0)? 0 : ((sum1 + sum2 + sum3) / div));
        //printf("[%d,%d] sum: %f, div: %d\n", idxX, idxY, tempsum, div);
        sum1 = sum2 = sum3 = 0.0f;
    }
}
