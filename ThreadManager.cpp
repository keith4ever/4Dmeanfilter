/*
 * Copyright (c) 2014 -     Keith Ha (keith4ever@gmail.com)
 * All content herein is protected by U.S. copyright and other applicable intellectual property laws
 * and may not be copied without the expressive permission of Keith Ha, who reserves all rights.
 * Reuse of any of the content for any purpose without the permission of Keith Ha
 * is strictly and expressively prohibited.
 */

#include "ThreadManager.h"

ThreadManager::ThreadManager(int numthread) {
    memset(&m_pid[0], 0, sizeof(pthread_t) * MAX_THREAD_NUM);
    memset(&m_bCond[0], true, sizeof(bool) * MAX_THREAD_NUM);
    m_numThread = (numthread > MAX_THREAD_NUM) ? MAX_THREAD_NUM : numthread;
    m_pThreadFunc   = NULL;
    m_arg           = NULL;
    m_vRunningThrIdx.clear();
}

ThreadManager::~ThreadManager() {

}

void ThreadManager::Init(void *(*func)(void *), void **arg) {
    if(func == NULL) return;
    m_pThreadFunc   = func;
    m_arg           = arg;
}

int ThreadManager::Run() {
    if(m_pThreadFunc == NULL) return -1;

    int numCreatedThreads = 0;
    for(int i = 0; i < m_numThread; i++) {
        if(!m_bCond[i] || isRunning(i)) continue;
        if(m_arg == NULL)
            pthread_create(&m_pid[i], NULL, m_pThreadFunc, NULL);
        else
            pthread_create(&m_pid[i], NULL, m_pThreadFunc, m_arg[i]);
        numCreatedThreads++;
        m_vRunningThrIdx.push_back(i);
    }
    return numCreatedThreads;
}

bool ThreadManager::Run(int idx) {
    if(isRunning(idx) || m_pThreadFunc == NULL
       || !m_bCond[idx] || m_numThread <= idx) return false;

    if(m_arg == NULL)
        pthread_create(&m_pid[idx], NULL, m_pThreadFunc, NULL);
    else
        pthread_create(&m_pid[idx], NULL, m_pThreadFunc, m_arg[idx]);

    m_vRunningThrIdx.push_back(idx);
    return true;
}

void ThreadManager::Join() {
    if(m_vRunningThrIdx.size() <= 0) return;

    for(int i = 0; i < m_numThread; i++) {
        if(m_pid[i] == 0 || !m_bCond[i] || !isRunning(i)) continue;
        pthread_join(m_pid[i], NULL);
        m_pid[i] = 0;     m_bCond[i] = true;
    }
    m_vRunningThrIdx.clear();
}

void ThreadManager::Cancel() {
    if(m_vRunningThrIdx.size() <= 0) return;

    for(int i = 0; i < m_numThread; i++) {
        if(m_pid[i] == 0 || !isRunning(i)) continue;
        pthread_cancel(m_pid[i]);
        m_pid[i] = 0;     m_bCond[i] = true;
    }
    m_vRunningThrIdx.clear();
}

bool ThreadManager::isRunning(int idx){
    return (std::find(m_vRunningThrIdx.begin(), m_vRunningThrIdx.end(), idx) != m_vRunningThrIdx.end());
}

void ThreadManager::Deinit() {
    m_pThreadFunc   = NULL;
    m_arg           = NULL;
}

void ThreadManager::SetRunCond(int i, bool bCond) {
    if(i >= m_numThread) return;
    m_bCond[i] = bCond;
}
