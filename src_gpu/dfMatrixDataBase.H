#pragma once

#include <stdio.h>
#include <unistd.h>
#include "cuda_profiler_api.h"
#include <cuda_runtime.h>
#include "nccl.h"
#include "mpi.h"
#include "nvtx3/nvToolsExt.h"
#include <vector>
#include <numeric>
#include <algorithm>
#include <map>
#include <iostream>
#include <ctime>
#include <cmath>
#include <unordered_map>

#include "AmgXSolver.H"
#include <amgx_c.h>

//#define DEBUG_
//#define DEBUG_CHECK_LDU

extern int myRank;

#define GPU_DEBUG_
#ifdef GPU_DEBUG_
   #define DEBUG_TRACE fprintf(stderr, "myRank[%d] %s %d\n", myRank, __FILE__, __LINE__);
#else
   #define DEBUG_TRACE
#endif

const double SMALL = std::numeric_limits<double>::epsilon();

static const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorName(error);
}

template <typename T>
void check(T result, char const *const func, const char *const file,
        int const line) {
  if (result) {
    fprintf(stderr, "cuda error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

inline void checkVectorEqual(int count, const double* basevec, double* vec, double max_relative_error, bool print = false) {
    for (int i = 0; i < count; ++i)
    {
        double abs_diff = fabs(basevec[i] - vec[i]);
        double rel_diff = fabs(basevec[i] - vec[i]) / fabs(basevec[i]);
        if (print && myRank == 0)
            fprintf(stderr, "myRank[%d] index %d, cpu data: %.24lf, gpu data: %.24lf, relative error: %.24lf\n", myRank, i, basevec[i
    ], vec[i], rel_diff);
        // if (abs_diff > 1e-12 && rel_diff > max_relative_error && !std::isinf(rel_diff))
        if (abs_diff > 1e-15 && rel_diff > max_relative_error && myRank == 0)
            fprintf(stderr, "myRank[%d] mismatch index %d, cpu data: %.30lf, gpu data: %.30lf, relative error: %.16lf\n", myRank, i, basevec[i], vec[i], rel_diff);
    }   
}

enum AMGXSetting {
    u_setting,
    p_setting
};

enum location {
    cpu,
    gpu
};

enum position {
    internal,
    boundary
};

enum boundaryConditions{
    zeroGradient,
    fixedValue,
    coupled,
    empty,
    gradientEnergy,
    calculated,
    cyclic,
    processor,
    extrapolated,
    fixedEnergy,
    processorCyclic,
    pressureInletOutletVelocity,
    totalPressure,
    wedge,
    noSlip,
    turbulentMixingLengthDissipationRateInlet,
    turbulentIntensityKineticEnergyInlet,
    nutkWallFunction,
    epsilonWallFunction,
    kqRWallFunction,
    alphatWallFunction
};

void constructBoundarySelectorPerPatch(int *patchTypeSelector, const std::string& patchTypeStr);

struct GAMGStruct
{
    int curLevel = -100; // current level && flag to save data 
    int nCell;
    int nFace;

    // === host data get from agglomeration
    std::vector<int> upperAddr;
    std::vector<int> lowerAddr;

    std::vector<int> restrictMap;
    std::vector<int> faceRestrictMap;
    std::vector<int> faceFlipMap;   // bool2int

    // === host data for multigpu parallel
    std::vector<int> nPatchFaces;

    // === d2h matrix data for verify
    double* h_lower = nullptr;
    double* h_upper = nullptr;
    double* h_diag = nullptr;

    // === d2h interface data for verify
    double** h_interfaceBouCoeffs = nullptr;
    double** h_interfaceIntCoeffs = nullptr;

    // === device map data
    int* d_restrictMap = nullptr;
    int* d_faceRestrictMap = nullptr;
    int* d_faceFlipMap = nullptr;   // bool2int

    // === device data for multigpu parallel
    int** d_faceCells = nullptr;
    int** d_patchFaceRestrictMap = nullptr;
    double** d_interfaceBouCoeffs = nullptr;
    double** d_interfaceIntCoeffs = nullptr;
  
    // === device matrix data
    double* d_lower = nullptr;
    double* d_upper = nullptr;
    double* d_diag = nullptr;
    int* d_lowerAddr = nullptr;
    int* d_upperAddr = nullptr;

    // === device csr matrix data
    double* d_off_diag_value = nullptr;
    int* d_csr_row_index_no_diag = nullptr;
    int* d_csr_col_index_no_diag = nullptr;

    // === device ell matrix data
    int ell_row_maxcount = 0;
    int *d_ell_cols = nullptr;
    double *d_ell_values = nullptr;

    // === device iteration data
    double* d_CorrFields = nullptr;
    double* d_Sources = nullptr;

    // === device temp data
    double* d_AcfField = nullptr;
    double* d_preSmoothField = nullptr;
    double* d_scalingFactorNum = nullptr;
    double* d_scalingFactorDenom = nullptr;
};


struct dfMatrixDataBase
{
    // cuda resource
    cudaStream_t stream;

    // communication info
    MPI_Comm mpi_comm;
    ncclComm_t nccl_comm;
    ncclUniqueId nccl_id;
    int nRanks;
    int myRank;
    int localRank;
    std::vector<int> neighbProcNo;

    // cyclic info
    std::vector<int> cyclicNeighbor;
    std::vector<int> patchSizeOffset;

    // constant values -- basic
    int num_cells = 0;
    int num_total_cells = 0;
    int num_surfaces = 0;
    int num_boundary_surfaces = 0;
    int num_proc_surfaces = 0;
    int num_Nz = 0;
    int num_patches = 0;
    int num_species = 0;
    int *interfaceFlag = nullptr;
    std::vector<int> patch_size;
    std::vector<int> patch_type_calculated;
    std::vector<int> patch_type_extropolated;
    std::vector<int> patch_type_T;
    double rdelta_t = 0;
    std::vector<int> lduCSRIndex;
    std::vector<int> lduCSRNoDiagIndex;

    // constant values -- ldu bytesize
    size_t cell_value_bytes = 0;
    size_t cell_value_vec_bytes = 0;
    size_t cell_value_tsr_bytes = 0;
    size_t cell_index_bytes = 0;
    size_t surface_value_bytes = 0;
    size_t surface_index_bytes = 0;
    size_t surface_value_vec_bytes = 0;
    size_t boundary_surface_value_bytes = 0;
    size_t boundary_surface_value_vec_bytes = 0;
    size_t boundary_surface_value_tsr_bytes = 0;
    size_t boundary_surface_index_bytes = 0;

    // constant values -- csr bytesize
    size_t csr_row_index_bytes = 0;
    size_t csr_col_index_bytes = 0;
    size_t csr_value_bytes = 0;
    size_t csr_value_vec_bytes = 0;
    size_t csr_col_index_bytes_no_diag = 0;

    // constant indexes
    int *d_owner = nullptr;
    int *d_neighbor = nullptr;
    int *d_diag_to_csr_index= nullptr;
    int *d_ldu_to_csr_index = nullptr;
    int *d_csr_row_index= nullptr;
    int *d_csr_col_index= nullptr;
    int *d_ldu_to_csr_no_diag = nullptr;
    int *d_csr_row_index_no_diag= nullptr;
    int *d_csr_col_index_no_diag= nullptr;

    int ell_row_maxcount = 0;
    int *d_ellCols = nullptr;
    int *d_ldu2ellIndex = nullptr;

    // GAMG constant indexes
    int **h_ldu_to_csr_no_diag = nullptr;
    int **h_csr_row_index_no_diag= nullptr;
    int **h_csr_col_index_no_diag= nullptr;

    int *h_ell_row_maxcount = nullptr;
    int **h_ellCols = nullptr;
    int **h_ldu2ellIndex = nullptr;

    // amgx solvers
	AmgXSolver *u_setting_solver = nullptr;
	AmgXSolver *p_setting_solver = nullptr;

    // constant fields - internal
    double *d_sf = nullptr;
    double *d_mesh_dis = nullptr;
    double *d_mag_sf = nullptr;
    double *d_weight = nullptr;
    double *d_phi_weight = nullptr; // weight for mvConvection->fvmDiv
    double *d_delta_coeffs = nullptr;
    double *d_volume = nullptr;
    
    double *h_sf = nullptr;
    double *h_mesh_dis = nullptr;

    // constant fields - boundary
    double *d_boundary_sf = nullptr;
    double *d_boundary_mag_sf = nullptr;
    double *d_boundary_weight = nullptr;
    double *d_boundary_delta_coeffs = nullptr;
    int *d_boundary_face_cell = nullptr;

    double *h_boundary_sf = nullptr;

    // non-constant fields - internal 
    // TODO: further estimate
    // fields solved by eqns - new
    double *d_rho = nullptr;
    double *d_u = nullptr;
    double *d_y = nullptr;
    double *d_he = nullptr;
    double *d_p = nullptr;
    double *d_k = nullptr;
    // fields solved by eqns - old 
    // TODO: not all fields need to store oldTime
    double *d_rho_old = nullptr;
    double *d_k_old = nullptr;
    double *d_u_old = nullptr;
    double *d_p_old = nullptr;
    double *d_u_old_host_order = nullptr; // tmp
    //double *d_y_old = nullptr;
    //double *d_he_old = nullptr;
    //double *d_p_old = nullptr;
    // other shared fields between eqns
    double *d_phi = nullptr;
    double *d_phi_old = nullptr;
    // other shared fields between eqns - thermophysical
    double *d_thermo_psi = nullptr;
    double *d_hDiff_corr_flux = nullptr;
    double *d_diff_alphaD = nullptr;
    double *d_dpdt = nullptr;

    double *d_T = nullptr;
    double *h_T = nullptr;
    double *d_mu = nullptr;
    double *d_thermo_alpha = nullptr;
    double *d_thermo_rhoD = nullptr;

    // computed on GPU, used on CPU, need memcpyd2h - host
    double *h_rho = nullptr;
    double *h_rho_old = nullptr;
    double *h_u= nullptr;
    double *h_u_old= nullptr;
    double *h_y= nullptr;
    double *h_he= nullptr;
    double *h_k= nullptr;
    double *h_k_old = nullptr;
    // computed on CPU, used on GPU, need memcpyh2d - host
    double *h_p= nullptr;
    double *h_p_old = nullptr;
    double *h_phi= nullptr;
    // internal fields used between eqns
    double *d_rAU = nullptr;
    double *d_HbyA = nullptr;
    // turbulence fields
    double *d_turbulence_k = nullptr;
    double *d_turbulence_epsilon = nullptr;

    // non-constant fields - boundary
    // TODO: further estimate
    // fields solved by eqns - new
    double *d_boundary_rho = nullptr;
    double *d_boundary_u = nullptr;
    double *d_boundary_y = nullptr;
    double *d_boundary_he = nullptr;
    double *d_boundary_p = nullptr;
    double *d_boundary_p_old = nullptr;
    double *d_boundary_k = nullptr;
    // fields solved by eqns - old
    double *d_boundary_rho_old = nullptr;
    double *d_boundary_u_old = nullptr;
    double *d_boundary_u_old_host_order = nullptr; // tmp
    //double *d_boundary_y_old = nullptr;
    //double *d_boundary_he_old = nullptr;
    //double *d_boundary_p_old = nullptr;
    // other shared fields between eqns
    double *d_boundary_phi = nullptr;
    double *d_boundary_phi_old = nullptr;
    // other shared fields between eqns - thermophysical
    double *d_boundary_thermo_psi = nullptr;
    double *d_boundary_hDiff_corr_flux = nullptr;
    double *d_boundary_diff_alphaD = nullptr;

    double *d_boundary_T = nullptr;
    double *d_boundary_mu = nullptr;
    double *d_boundary_thermo_alpha = nullptr;
    double *d_boundary_thermo_rhoD = nullptr;
    // boundary fields used between eqns
    double *d_boundary_rAU = nullptr;
    double *d_boundary_HbyA = nullptr;
    // computed on GPU, used on CPU, need memcpyd2h - host
    double *h_boundary_rho = nullptr;
    double *h_boundary_rho_old = nullptr;
    double *h_boundary_u= nullptr;
    double *h_boundary_u_old= nullptr;
    double *h_boundary_y= nullptr;
    double *h_boundary_he= nullptr;
    double *h_boundary_k = nullptr;
    // computed on CPU, used on GPU, need memcpyh2d - host
    double *h_boundary_p= nullptr;
    double *h_boundary_p_old= nullptr;
    double *h_boundary_phi= nullptr;
    double *h_boundary_phi_old= nullptr;

    std::unordered_map<std::string, double*> fieldPointerMap;

    // constructor
    dfMatrixDataBase();

    // deconstructor
    ~dfMatrixDataBase();

    // member function
    void prepareCudaResources();
    void cleanCudaResources();
    void setCommInfo(MPI_Comm mpi_comm, ncclComm_t nccl_comm, ncclUniqueId nccl_id,
            int nRanks, int myRank, int localRank, std::vector<int> &neighbProcNo);
    void setConstantValues(int num_cells, int num_total_cells, int num_surfaces, 
        int num_boundary_surfaces, int num_patches, int num_proc_surfaces, 
        std::vector<int> patch_size, int num_species, double rdelta_t);  
    void setConstantIndexes_GAMG(int nCells, int nFaces, int *owner, int *neighbor, int leveli, int type);           
    void setConstantIndexes(const int *owner, const int *neighbor, const int *procRows, 
            const int *procCols, int globalOffset);
    void setAmgxSolvers(const std::string &mode_string, const std::string &u_setting_path, const std::string &p_setting_path);
    void resetAmgxSolvers();
    void solve(int num_iteration, AMGXSetting setting, double *d_A, double *d_x, double *d_b);
    void setCyclicInfo(std::vector<int> &cyclicNeighbor); // when use cyclic boundary

    void createConstantFieldsInternal();
    void createConstantFieldsBoundary();
    void initConstantFieldsInternal(const double *sf, const double *mag_sf, 
            const double *weight, const double *delta_coeffs, const double *volume, const double *mesh_distance);
    void initConstantFieldsBoundary(const double *boundary_sf, const double *boundary_mag_sf, 
            const double *boundary_delta_coeffs, const double *boundary_weight, const int *boundary_face_cell, 
            std::vector<int> &patch_type_calculated, std::vector<int> &patch_type_extropolated);

    void createNonConstantFieldsInternal();
    void createNonConstantFieldsBoundary();
    void initNonConstantFieldsInternal();
    void initNonConstantFieldsBoundary();

    void preTimeStep();
    void postTimeStep();

    // getter
    double* getFieldPointer(const char* fieldAlias, location loc, position pos);
};

