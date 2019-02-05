/*
 * Copyright (c) 2014 -     Keith Ha (keith4ever@gmail.com)
 * All content herein is protected by U.S. copyright and other applicable intellectual property laws
 * and may not be copied without the expressive permission of Keith Ha., which reserves all rights.
 * Reuse of any of the content for any purpose without the permission of Keith Ha
 * is strictly and expressively prohibited.
 */

#ifndef _THREADMANAGER_H
#define _THREADMANAGER_H

#include <pthread.h>
#include <memory.h>
#include <semaphore.h>
#include <algorithm>
#include <vector>
#define MAX_THREAD_NUM      16

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


#endif //MULTIPIP_THREADMANAGER_H
