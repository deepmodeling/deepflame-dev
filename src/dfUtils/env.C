#include "env.H"
#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h> 
#endif

bool env_get_bool(const char* name, bool default_value){
    char* tmp = getenv(name);
    if(tmp == NULL){
        return default_value;
    }
    if(std::strcmp(tmp, "true") == 0){
        return true;
    }else{
        return false;
    }
}

int env_get_int(const char* name, int default_value){
    char* tmp = getenv(name);
    if(tmp == NULL){
        return default_value;
    }
    return std::atoi(tmp);
}

int dnn_batch_size = env_get_int("DNN_BATCH_SIZE", 16384);
int row_block_bit = env_get_int("ROW_BLOCK_BIT", 5);

void env_show(){
    int mpirank;
    int flag_mpi_init;
    MPI_Initialized(&flag_mpi_init);
    if (flag_mpi_init) {
        MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    }
    
    if(mpirank != 0) return;
    
    std::cout << std::endl;
    std::cout << "env show --------------------------------" << std::endl;
    std::cout << "dnn_batch_size : " << dnn_batch_size << std::endl;
    std::cout << "row_block_bit : " << row_block_bit << std::endl;
    std::cout << "max_threads : " << omp_get_max_threads() << std::endl;
    #ifdef __ARM_FEATURE_SVE
    std::cout << "simd width : " << svcntd() * 64 << std::endl;
    #endif
    std::cout << "-----------------------------------------" << std::endl;
}

