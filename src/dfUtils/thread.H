#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

inline int df_max_threads(){
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

inline int df_thread_size(){
#ifdef _OPENMP
    return omp_get_num_threads();
#else
    return 1;
#endif
}

inline int df_thread_rank(){
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}