/*
 * Copyright (c) 2014 -     Keith Ha (keith4ever@gmail.com)
 * All content herein is protected by U.S. copyright and other applicable intellectual property laws
 * and may not be copied without the expressive permission of Keith Ha., which reserves all rights.
 * Reuse of any of the content for any purpose without the permission of Keith Ha
 * is strictly and expressively prohibited.
 */

#include "meanfilter.h"
#include "kernel.h"

const float inputTensor[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                             10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                             10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

Config::Config() {
    m_bCPUCompute = false;
}

Config::~Config() {

}

void Config::printHelp(bool bInitSuccess) {
    printf("=================================================================\n");
    if (!bInitSuccess){
        printf("Usage: meanfilter \n");
        printf("	-cpu [=> Use CPU] \n");
    }
    else{
        printf(" [Execution Parameters]\n");
        printf(" computing device: %s\n", m_bCPUCompute ? "CPU" : "GPU");
    }
    printf("=================================================================\n");
    if (!bInitSuccess)
        exit(-1);
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

int main(int argc, char* argv[])
{
    /*if(argc < 2) {
        print_usage(argv);
        return -1;
    }*/

    auto pFilter = make_shared<Meanfilter4D>();
    auto pConfig = make_shared<Config>();

    int inBufferSize = D1DIM * D2DIM * D3DIM * D4DIM;
    float* pInputBuffer = new float[inBufferSize];
    float* pOutputBuffer = new float[inBufferSize];
    memcpy(pInputBuffer, inputTensor, sizeof(float)*D1DIM * D2DIM * D3DIM * D4DIM);

    pConfig->parseArguments(argc, argv);
    pFilter->init(D1DIM, D2DIM, D3DIM, D4DIM, pConfig->isCPUCompute());
    pFilter->execute(pInputBuffer, pOutputBuffer);
    pFilter->printOut(pOutputBuffer);
    pFilter->deinit();

    delete [] pInputBuffer;
    return 0;
}
