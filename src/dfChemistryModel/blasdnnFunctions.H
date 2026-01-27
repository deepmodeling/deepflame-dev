/*
#include <hbwmalloc.h>
#include <memkind.h>
#include <assert.h>
#include <numa.h>
#include <numaif.h>
#include <sys/mman.h>
#include <stdio.h>
#include <stdlib.h>

#define HBM_ALIGNED_SIZE (1 << 21)
#define ENABLE_HBM
template<typename DataType>
int alloc_on_hbm_hugetlb_with_check(DataType** addr, size_t length_bytes) {
        if(length_bytes > 0) {
            length_bytes = (length_bytes + HBM_ALIGNED_SIZE - 1) / HBM_ALIGNED_SIZE * HBM_ALIGNED_SIZE;
            void* mapAddress = NULL;
            mapAddress = mmap(NULL, length_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
            if (mapAddress == NULL) {
                printf("mmap failed ,length = %d \n", length_bytes);
                exit(-1);
            };
            *addr = (DataType*)mapAddress;
            printf("using mmap allocating memory success, length = %d \n", length_bytes);                
        } else {
            printf("illegal length_bytes \n");    
            exit(-1);
        }
}
*/

template <class ThermoType>
template <class DeltaTType>
Foam::scalar Foam::dfChemistryModel<ThermoType>::solve_DNN_blas
(
    const DeltaTType& deltaT
)
{
    double init_time = 0.;
    double prepare_time = 0.;
    double inference_time = 0.;
    double update_time = 0.;
    double destory_time = 0.;
    
    clockTime clock;

    scalar deltaTMin = great;
    if (!this->chemistry_)
    {
        return deltaTMin;
    }
    Info<<"=== begin solve_DNN with BLASDNN === "<<endl;

    label nCells = mesh_.nCells();
    std::vector<size_t> Cell0;
    Cell0.reserve(nCells);
    
    size_t nSpecies = mixture_.nSpecies();

    const scalar lambda = 0.1;
    const scalar lambdaR = 1 / lambda;
    const scalar CONST_101325 = 101325.;
    const scalar CONSTR_101325 = 1. / CONST_101325;

    label input_dim = nSpecies + 2;
    int output_dim = nSpecies -1;
/*
#if defined(ENABLE_HBM)
    int ret = -1;
#if defined(USE_BLASDNN_HALF)
    __fp16* NNInputs0;
    __fp16* results0;
    ret = alloc_on_hbm_hugetlb_with_check<__fp16>(&NNInputs0, (nCells * input_dim) * sizeof(__fp16));
    ret = alloc_on_hbm_hugetlb_with_check<__fp16>(&results0, (nCells * output_dim) * sizeof(__fp16));
//    memset(NNInputs0, '\0', ((nCells * input_dim) * sizeof(__fp16) + HBM_ALIGNED_SIZE - 1) / HBM_ALIGNED_SIZE * HBM_ALIGNED_SIZE   );
#elif defined(USE_BLASDNN_FLOAT)
    float* NNInputs0, * results0;
    ret = alloc_on_hbm_hugetlb_with_check<float>(&NNInputs0, (nCells * input_dim) * sizeof(float));
    ret = alloc_on_hbm_hugetlb_with_check<float>(&NNInputs0, (nCells * input_dim) * sizeof(float));
    memset(NNInputs0, '\0', (nCells * input_dim) * sizeof(float));
#elif defined(USE_BLASDNN_DOUBLE)
    double* NNInputs0, * results0;
    ret = alloc_on_hbm_hugetlb_with_check<double>(&NNInputs0, (nCells * input_dim) * sizeof(double));
    ret = alloc_on_hbm_hugetlb_with_check<double>(&NNInputs0, (nCells * input_dim) * sizeof(double));
    memset(NNInputs0, '\0', (nCells * input_dim) * sizeof(double));
#else
#error "must define one of USE_BLASDNN_HALF, USE_BLASDNN_FLOAT or USE_BLASDNN_DOUBLE"
#endif

#else
*/
#if defined(USE_BLASDNN_HALF)
/*
    __fp16* NNInputs0, * results0;
    int length_NNInputs0 = ((nCells * input_dim) * sizeof(__fp16) + HBM_ALIGNED_SIZE - 1) / HBM_ALIGNED_SIZE * HBM_ALIGNED_SIZE;
    void* mapAddress = NULL;
    mapAddress = mmap(NULL, length_NNInputs0, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (mapAddress == NULL) {
        printf("mmap NNInputs0 failed ,length = %d \n", length_NNInputs0);

    };
    NNInputs0 = (__fp16*)mapAddress;
    printf("allocate NNInputs0 success\n");
    int length_results0 = ((nCells * output_dim) * sizeof(__fp16) + HBM_ALIGNED_SIZE - 1) / HBM_ALIGNED_SIZE * HBM_ALIGNED_SIZE;
    mapAddress = NULL;
    mapAddress = mmap(NULL, length_results0, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (mapAddress == NULL) {
        printf("mmap results0 failed ,length = %d \n", length_results0);

    };
    results0 = (__fp16*)mapAddress;
    printf("allocate results0 success\n");
    memset(NNInputs0, '\0', length_NNInputs0 );
    printf("memset success\n");        
/*
    int ret = -1;
    __fp16* NNInputs0;
    __fp16* results0;
    ret = alloc_on_hbm_hugetlb_with_check<__fp16>(&NNInputs0, (nCells * input_dim) * sizeof(__fp16));
    printf("allocate NNInputs0 success\n");
    ret = alloc_on_hbm_hugetlb_with_check<__fp16>(&results0, (nCells * output_dim) * sizeof(__fp16));
    printf("allocate results0 success\n");    
    memset(NNInputs0, '\0', ((nCells * input_dim) * sizeof(__fp16) + HBM_ALIGNED_SIZE - 1) / HBM_ALIGNED_SIZE * HBM_ALIGNED_SIZE   );
    printf("memset success\n");
*/
    __fp16* NNInputs0 = (__fp16*)aligned_alloc(64, (nCells * input_dim) * sizeof(__fp16));
    __fp16* results0 = (__fp16*)aligned_alloc(64, (nCells * output_dim) * sizeof(__fp16));
    memset(NNInputs0, '\0', (nCells * input_dim) * sizeof(__fp16));

#elif defined(USE_BLASDNN_FLOAT)
    float* NNInputs0 = (float*)aligned_alloc(64, (nCells * input_dim) * sizeof(float));
    float* results0 = (float*)aligned_alloc(64, (nCells * output_dim) * sizeof(float));
    memset(NNInputs0, '\0', (nCells * input_dim) * sizeof(float));
#elif defined(USE_BLASDNN_DOUBLE)
    double* NNInputs0 = (double*)aligned_alloc(64, (nCells * input_dim) * sizeof(double));
    double* results0 = (double*)aligned_alloc(64, (nCells * output_dim) * sizeof(double));
    memset(NNInputs0, '\0', (nCells * input_dim) * sizeof(double));
#else
#error "must define one of USE_BLASDNN_HALF, USE_BLASDNN_FLOAT or USE_BLASDNN_DOUBLE"
#endif
//#endif

    init_time += clock.timeIncrement();

#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        scalarList yPre_(nSpecies);
        scalarList yBCT_(nSpecies);
        scalarList yTmp_(nSpecies);
        scalarList u_(nSpecies);
        // get problems
        #pragma omp for
        for(size_t celli = 0; celli<nCells; ++celli){
            // size_t celli = Cell0[celli];
            scalar Ti = T_[celli];
            scalar pi = p_[celli];
            size_t offset = celli * input_dim;
            // size_t offset = celli;

            NNInputs0[offset + 0] = (Ti - Xmu_[0])/Xstd_[0];
            NNInputs0[offset + 1] = (pi - Xmu_[1])/Xstd_[1];
            // NNInputs0[offset + 0 * input_dim] = (Ti - Xmu_[0])/Xstd_[0];
            // NNInputs0[offset + 1 * input_dim] = (pi - Xmu_[1])/Xstd_[1];
            for (size_t i=0; i<nSpecies; i++)
            {
                yPre_[i] = Y_[i][celli];
                yBCT_[i] = (pow(yPre_[i],lambda) - 1) / lambda; // function BCT
            }
            for (size_t i=0; i<nSpecies; i++)
            {
                NNInputs0[offset + 2 + i] = (yBCT_[i] - Xmu_[i+2]) / Xstd_[i+2];
                // NNInputs0[offset + (2 + i) * input_dim] = (yBCT_[i] - Xmu_[i+2]) / Xstd_[i+2];
            }
        }
    }
/*
    prepare_time += clock.timeIncrement();
    char* DNN_tmp = getenv("DNN_NUM_THREADS");
    if (DNN_tmp != NULL) {
        int DNN = std::atol(DNN_tmp);
        omp_set_num_threads(DNN);
        printf("use DNN_NUM_THREADS = %d\n", DNN);
    } else {
        printf("use OMP_NUM_THREADS\n");
    }
*/
    DNNInferencer_blas_.Inference_multiDNNs(NNInputs0, results0, nCells);
//    omp_set_num_threads(std::atol(getenv("OMP_NUM_THREADS")));

    inference_time += clock.timeIncrement();

//     // update Q & RR
// #ifdef _OPENMP
//     #pragma omp parallel
// #endif
//     {
//         scalarList yPre_(mixture_.nSpecies());
//         scalarList yBCT_(mixture_.nSpecies());
//         scalarList yTmp_(mixture_.nSpecies());
//         scalarList u_(mixture_.nSpecies());
//         // // - NN0
//         #pragma omp for
//         for(size_t celli = 0; celli<nCells; celli ++)
//         {
//             size_t offset = celli * (nSpecies-1);
//             scalar Yt = 0;
//             for (size_t i=0; i<nSpecies ; i++)
//             {
//                 yPre_[i] = Y_[i][celli];
//                 yTmp_[i] = Y_[i][celli];
//                 yBCT_[i] = (pow(yPre_[i],lambda) - 1) / lambda; 
//             }
//             for (size_t i=0; i<nSpecies -1; i++)
//             {

//                 u_[i] = results0[offset + i ]*Ystd_[i]+Ymu_[i];
//                 yTmp_[i] = pow((u_[i]+yBCT_[i])*lambda+1,1/lambda);
//                 Yt += yTmp_[i]; // normalization
//             }
//             u_[17] = 0;
//             yTmp_[17] = yPre_[17] ;
//             Qdot_[celli] = 0;
//             RR_[17][celli] = 0;
//             for (size_t i=0; i<nSpecies-1; i++) 
//             {
//                 // TODO: READIN
//                 if(T_[celli]<510){
//                     RR_[i][celli] = 0;
//                 }
//                 else{
//                     yTmp_[i] = yTmp_[i] / Yt *(1-yTmp_[17]);
//                     // TODO: READIN
//                     RR_[i][celli] = (yTmp_[i] - Y_[i][celli])*rho_[celli]/5e-8;
//                     // Qdot_[celli] -= hc_[i]*RR_[i][celli];
//                 }
//             }
//         }
//     }

    update_time = clock.timeIncrement();
//#if defined(ENABLE_HBM)
//    munmap(NNInputs0, length_NNInputs0);
//    munmap(results0, length_results0);
//#else
    free(NNInputs0);
    free(results0);
//#endif

    destory_time = clock.timeIncrement();
    double solve_DNN_blas_time = clock.elapsedTime();

    Info << "=== end solve_DNN with BLASDNN === " << endl;

    Info << "solve_DNN_blas profiling --------------------------------------------------" << endl;
    Info << "solve_DNN_blas time : " << solve_DNN_blas_time << endl;
    Info << "init time : " << init_time << endl;
    Info << "prepare time : " << prepare_time << endl;
    Info << "inference time : " << inference_time << endl;
    Info << "update time : " << update_time << endl;
    Info << "destory time : " << destory_time << endl;
    Info << "other time : " << solve_DNN_blas_time - init_time - prepare_time - inference_time - update_time - destory_time << endl;
    Info << "---------------------------------------------------------------------------" << endl;

    return deltaTMin;
}
