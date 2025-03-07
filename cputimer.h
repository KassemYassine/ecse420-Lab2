
#ifndef __CPU_TIMER_H__
#define __CPU_TIMER_H__

#include <windows.h>

/**
 * @brief Timer for CPU.
 * 
 */
struct CpuTimer
{
    LARGE_INTEGER freq;
    LARGE_INTEGER start;
    LARGE_INTEGER stop;

    CpuTimer()
    {
        QueryPerformanceFrequency(&freq);
    }

    void Start()
    {
        QueryPerformanceCounter(&start);
    }

    void Stop()
    {
        QueryPerformanceCounter(&stop);
    }

    /**
     * @brief Computes the elapsed time in milliseconds.
     * 
     * @return the elapsed time in milliseconds 
     */
    double Elapsed() const
    {
        return (double)1000 * (stop.QuadPart - start.QuadPart) / freq.QuadPart;
    }
};

#endif