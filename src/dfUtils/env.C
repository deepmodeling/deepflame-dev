#include "env.H"
#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h> 
#endif

namespace Foam{

const int env::REGION_DECOMPOSE_NBLOCKS = env::get_int_default("REGION_DECOMPOSE_NBLOCKS", 12);
const int env::DNN_BATCH_SIZE = env::get_int_default("DNN_BATCH_SIZE", 32768);
const std::string env::DFMATRIX_INNERMATRIX_TYPE = env::get_string_default("DFMATRIX_INNERMATRIX_TYPE", "LDU");


bool env::get_bool(const std::string& name){
    const char* tmp = std::getenv(name.c_str());
    if(tmp == nullptr){
        // print error
        SeriousError << "env " << name << " not set" << endl << flush;
        std::exit(-1);
    }
    if(std::strcmp(tmp, "true") == 0){
        return true;
    }else{
        return false;
    }
}

int env::get_int(const std::string& name){
    const char* tmp = std::getenv(name.c_str());
    if(tmp == nullptr){
        // print error
        SeriousError << "env " << name << " not set" << endl << flush;
        std::exit(-1);
    }
    return std::atoi(tmp);
}

const std::string& env::get_string(const std::string& name){
    const char* tmp = std::getenv(name.c_str());
    if(tmp == nullptr){
        // print error
        SeriousError << "env " << name << " not set" << endl << flush;
        std::exit(-1);
    }
    return std::string(tmp);
}

bool env::get_bool_default(const std::string& name, bool default_value){
    const char* tmp = std::getenv(name.c_str());
    if(tmp == nullptr){
        return default_value;
    }
    if(std::strcmp(tmp, "true") == 0){
        return true;
    }else{
        return false;
    }
}

int env::get_int_default(const std::string& name, int default_value){
    const char* tmp = std::getenv(name.c_str());
    if(tmp == nullptr){
        return default_value;
    }
    return std::atoi(tmp);
}

const std::string& env::get_string_default(const std::string& name, const std::string& default_value){
    const char* tmp = std::getenv(name.c_str());
    if(tmp == nullptr){
        return default_value;
    }
    return std::string(tmp);
}

void env::show(){
    Info << endl;
    Info << "env show --------------------------------" << endl;
    Info << "dnn_batch_size : " << DNN_BATCH_SIZE << endl;
    Info << "region_decompose_nblocks : " << REGION_DECOMPOSE_NBLOCKS << endl;
    Info << "dfmatrix_innermatrix_type : " << DFMATRIX_INNERMATRIX_TYPE << endl;
    Info << "max_threads : " << omp_get_max_threads() << endl;
    #ifdef __ARM_FEATURE_SVE
    Info << "simd width : " << svcntd() * 64 << endl;
    #endif
    Info << "-----------------------------------------" << endl;
}


}
