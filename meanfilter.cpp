/*
 * Copyright (c) 2014 -     Keith Ha (keith4ever@gmail.com)
 * All content herein is protected by U.S. copyright and other applicable intellectual property laws
 * and may not be copied without the expressive permission of Keith Ha, who reserves all rights.
 * Reuse of any of the content for any purpose without the permission of Keith Ha
 * is strictly and expressively prohibited.
 */

#include "meanfilter.h"
#include "Meanfilter4D.h"

const float inputTensor[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                             10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                             10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

Config::Config() {
    m_bCPUCompute   = false;
    m_bPrintOut     = false;
    m_bDefaultDim   = true;
    m_dim[0]        = D1DIM;
    m_dim[1]        = D2DIM;
    m_dim[2]        = D3DIM;
    m_dim[3]        = D4DIM;
}

Config::~Config() {

}

void Config::generateRandNum(int size, float* buf)
{
    float* ptr = buf;

//    __PerfTimerStart__
    for(int i=0; i < size; i++)
        ptr[i] = (rand() % (1<<10));
//    __PerfTimerEnd__
}

bool Config::parseArguments(int argc, char **argv) {
    for (int i = 1; i < argc; i++)
    {
/*        if (stricmp(argv[i], "-i") == 0)
        {
            if (++i >= argc)
            {
                fprintf(stderr, "invalid parameter for %s\n", argv[i - 1]);
                return false;
            }
            inputDir = argv[i];
        }
        else*/
        if (strcasecmp(argv[i], "-cpu") == 0)
        {
            m_bCPUCompute = true;
        }
        else if (strcasecmp(argv[i], "-p") == 0)
        {
            m_bPrintOut = true;
        }
        else if (strcasecmp(argv[i], "-d1") == 0)
        {
            int data = 0;
            if (++i >= argc || sscanf(argv[i], "%d", &data) != 1)
            {
                fprintf(stderr, "invalid parameter for %s\n", argv[i - 1]);
                return false;
            }
            m_dim[0] = data;    m_bDefaultDim = false;
        }
        else if (strcasecmp(argv[i], "-d2") == 0)
        {
            int data = 0;
            if (++i >= argc || sscanf(argv[i], "%d", &data) != 1)
            {
                fprintf(stderr, "invalid parameter for %s\n", argv[i - 1]);
                return false;
            }
            m_dim[1] = data;    m_bDefaultDim = false;
        }
        else if (strcasecmp(argv[i], "-d3") == 0)
        {
            int data = 0;
            if (++i >= argc || sscanf(argv[i], "%d", &data) != 1)
            {
                fprintf(stderr, "invalid parameter for %s\n", argv[i - 1]);
                return false;
            }
            m_dim[2] = data;    m_bDefaultDim = false;
        }
        else if (strcasecmp(argv[i], "-d4") == 0)
        {
            int data = 0;
            if (++i >= argc || sscanf(argv[i], "%d", &data) != 1)
            {
                fprintf(stderr, "invalid parameter for %s\n", argv[i - 1]);
                return false;
            }
            m_dim[3] = data;    m_bDefaultDim = false;
        }
        else
        {
            fprintf(stderr, "invalid parameter  %s\n", argv[i++]);
            return false;
        }
    }

    return true;
}

void Config::checkFileAndRead(float* buf, int bufsize, char* file)
{
    FILE* fp;
    if (buf == NULL || file == NULL) return;

    string filename(file);
    fp = fopen(filename.c_str(), "r");
    if (fp == NULL) {
        cerr << file << " doesn't exist !" << endl;
        exit(-1);
    }
    fseek(fp, 0L, SEEK_END);
    size_t size = ftell(fp);
    if(size < bufsize){
        cerr << file << " size is less than " << bufsize << " !" << endl;
        exit(-1);
    }
    fread(buf, 1, bufsize, fp);
    fclose(fp);
}

int main(int argc, char* argv[]) {
    auto pFilter = make_shared<Meanfilter4D>();
    auto pConfig = make_shared<Config>();

    pConfig->parseArguments(argc, argv);

    int inBufferSize = pConfig->getDataSize();
    float *pInputBuffer = new float[inBufferSize];
    float *pOutputBuffer = new float[inBufferSize];
    int streamUnit;

    if (pConfig->m_bDefaultDim) {
        streamUnit = 2;
        memcpy(pInputBuffer, inputTensor, sizeof(float) * inBufferSize);
    } else {
        streamUnit = 8;
        pConfig->generateRandNum(inBufferSize, pInputBuffer);
    }

    pFilter->init(pConfig->m_dim[0], pConfig->m_dim[1], pConfig->m_dim[2], pConfig->m_dim[3],
                  pConfig->isCPUCompute());
    pFilter->execute(pInputBuffer, pOutputBuffer, streamUnit);
    if(pConfig->m_bPrintOut)
        pFilter->printOut(pOutputBuffer);
    pFilter->deinit();

    delete [] pInputBuffer;
    delete [] pOutputBuffer;
    
    return 0;
}

