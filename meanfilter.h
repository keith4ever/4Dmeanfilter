/*
 * Copyright (c) 2014 -     Keith Ha (keith4ever@gmail.com)
 * All content herein is protected by U.S. copyright and other applicable intellectual property laws
 * and may not be copied without the expressive permission of Keith Ha, who reserves all rights.
 * Reuse of any of the content for any purpose without the permission of Keith Ha
 * is strictly and expressively prohibited.
 */

#pragma once

#include <stdio.h>
#include <iostream>
#include <memory>
#include <string.h>
#include <unistd.h>
#include <cuda.h>
#include <stdlib.h>

#define D1DIM   16
#define D2DIM   16
#define D3DIM   16
#define D4DIM   16

using namespace std;

class Config{
private:
    bool m_bCPUCompute;

public:
    bool m_bPrintOut;
    int  m_dim[4];

    Config();
    ~Config();
    void generateRandNum(int size, float* buf);
    bool parseArguments(int argc, char** argv);
    void checkFileAndRead(float* buf, int bufsize, char* file);
    bool isCPUCompute() { return m_bCPUCompute; }
    int  getDataSize()  { return m_dim[0] * m_dim[1] * m_dim[2] * m_dim[3]; }
};
