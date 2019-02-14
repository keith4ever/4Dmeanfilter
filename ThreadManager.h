/*
 * Copyright (c) 2014 -     Keith Ha (keith4ever@gmail.com)
 * All content herein is protected by U.S. copyright and other applicable intellectual property laws
 * and may not be copied without the expressive permission of Keith Ha, who reserves all rights.
 * Reuse of any of the content for any purpose without the permission of Keith Ha
 * is strictly and expressively prohibited.
 */

#ifndef _THREADMANAGER_H
#define _THREADMANAGER_H

#include <pthread.h>
#include <memory.h>
#include <semaphore.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <condition_variable>

#define MAX_THREAD_NUM      8

using namespace std;

class ThreadManager {
private:
    pthread_t   m_pid[MAX_THREAD_NUM];
    vector<int>    m_vRunningThrIdx;
    int         m_numThread = 0;
    void*       (*m_pThreadFunc)(void*);
    void**      m_arg;
    bool        m_bCond[MAX_THREAD_NUM];

public:
    ThreadManager(int numthread);
    ~ThreadManager();

    void Init(void*(*func)(void*), void* arg[]);
    void Deinit();
    void SetRunCond(int i, bool bCond);
    int  Run();
    bool  Run(int i);
    void Join();
    void Cancel();
    bool isRunning(int idx);
};

class Barrier {
public:
    Barrier();
    void Init(size_t iCount);
    void Wait();

private:
    mutex mMutex;
    condition_variable mCond;
    size_t mThreshold;
    size_t mCount;
    size_t mGeneration;
};

#endif //MULTIPIP_THREADMANAGER_H
