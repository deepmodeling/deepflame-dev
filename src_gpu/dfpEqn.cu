#include "dfpEqn.H"
#include "dfSolverOpBase.H"
// #define AMGX_
#define CSR_
// #define ELL_

// #define PARALLEL_

// Start if write vector to file ...
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <string>
#include <nvtx3/nvToolsExt.h>

void readArrayFromBinaryFile(const std::string& filename, double* array, std::size_t size) {
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) {
        std::cerr << "Error opening file for reading: " << filename << std::endl;
        return;
    }

    inFile.read(reinterpret_cast<char*>(array), size * sizeof(double));
    inFile.close();
}


__global__ void fvc_interpolate_internal_multi_scalar_kernel(int num_surfaces, const int *lower_index, const int *upper_index,
        const double *vf1, const double *vf2, const double *weight, double *output, double sign)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_surfaces)
        return;
    
    double w = weight[index];

    int owner = lower_index[index];
    int neighbor = upper_index[index];

    double vf3_owner = vf1[owner] * vf2[owner];
    double vf3_neighbour = vf1[neighbor] * vf2[neighbor];

    output[index] = (w * (vf3_owner - vf3_neighbour) + vf3_neighbour);
}

__global__ void fvc_interpolate_boundary_multi_scalar_kernel_unCouple(int num, int offset,
        const double *boundary_vf1, const double *boundary_vf2, double *output, double sign)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num)
        return;
    
    int start_index = offset + index;
    double boundary_vf3 = boundary_vf1[start_index] * boundary_vf2[start_index];
    output[start_index] = boundary_vf3;
}

__global__ void fvc_interpolate_boundary_multi_scalar_kernel_processor(int num, int offset,
        const double *boundary_weight, const double *boundary_vf1, const double *boundary_vf2, double *output, double sign)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num)
        return;
    
    int neighbor_start_index = offset + index;
    int internal_start_index = offset + num + index;

    double bouWeight = boundary_weight[neighbor_start_index];

    double neighbor_boundary_vf3 = boundary_vf1[neighbor_start_index] * boundary_vf2[neighbor_start_index];
    double internal_boundary_vf3 = boundary_vf1[internal_start_index] * boundary_vf2[internal_start_index];
    
    double boundary_vf3 = (1 - bouWeight) * neighbor_boundary_vf3 + bouWeight * internal_boundary_vf3;
    
    output[neighbor_start_index] = boundary_vf3;
}

__global__ void get_phiCorr_internal_kernel(int num_cells, int num_surfaces, 
        const int *lower_index, const int *upper_index, const double *phi_old, 
        const double *field_vector, const double *field_scalar, const double *weight, const double *face_vector,
        double *output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_surfaces)
        return;
    
    double w = weight[index];
    double Sfx = face_vector[num_surfaces * 0 + index];
    double Sfy = face_vector[num_surfaces * 1 + index];
    double Sfz = face_vector[num_surfaces * 2 + index];

    int owner = lower_index[index];
    int neighbor = upper_index[index];

    double vf_own_x = field_vector[num_cells * 0 + owner] * field_scalar[owner];
    double vf_own_y = field_vector[num_cells * 1 + owner] * field_scalar[owner];
    double vf_own_z = field_vector[num_cells * 2 + owner] * field_scalar[owner];

    double vf_nei_x = field_vector[num_cells * 0 + neighbor] * field_scalar[neighbor];
    double vf_nei_y = field_vector[num_cells * 1 + neighbor] * field_scalar[neighbor];
    double vf_nei_z = field_vector[num_cells * 2 + neighbor] * field_scalar[neighbor];

    double ssfx = (w * (vf_own_x - vf_nei_x) + vf_nei_x);
    double ssfy = (w * (vf_own_y - vf_nei_y) + vf_nei_y);
    double ssfz = (w * (vf_own_z - vf_nei_z) + vf_nei_z);

    output[index] = phi_old[index] - (Sfx * ssfx + Sfy * ssfy + Sfz * ssfz);    
}

__global__ void get_phiCorr_boundary_kernel_zeroGradient(int num_boundary_surfaces, int num, int offset,
        const double *boundary_face_vector, const double *boundary_field_vector, 
        const double *boundary_field_scalar, const double *boundary_phi_old, double *output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num)
        return;
    
    int start_index = offset + index;

    double bouSfx = boundary_face_vector[num_boundary_surfaces * 0 + start_index];
    double bouSfy = boundary_face_vector[num_boundary_surfaces * 1 + start_index];
    double bouSfz = boundary_face_vector[num_boundary_surfaces * 2 + start_index];

    double boussfx = boundary_field_vector[num_boundary_surfaces * 0 + start_index] * boundary_field_scalar[start_index];
    double boussfy = boundary_field_vector[num_boundary_surfaces * 1 + start_index] * boundary_field_scalar[start_index];
    double boussfz = boundary_field_vector[num_boundary_surfaces * 2 + start_index] * boundary_field_scalar[start_index];

    output[start_index] = boundary_phi_old[start_index] - (bouSfx * boussfx + bouSfy * boussfy + bouSfz * boussfz);
}

__global__ void get_phiCorr_boundary_kernel_processor(int num_boundary_surfaces, int num, int offset,
        const double *boundary_face_vector, const double *boundary_field_vector, 
        const double *boundary_field_scalar, const double *boundary_phi_old, 
        const double *boundary_weight, double *output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num)
        return;
    
    int neighbor_start_index = offset + index;
    int internal_start_index = offset + num + index;

    double bouWeight = boundary_weight[neighbor_start_index];

    double bouSfx = boundary_face_vector[num_boundary_surfaces * 0 + neighbor_start_index];
    double bouSfy = boundary_face_vector[num_boundary_surfaces * 1 + neighbor_start_index];
    double bouSfz = boundary_face_vector[num_boundary_surfaces * 2 + neighbor_start_index];

    double boussfxNeighbor = boundary_field_vector[num_boundary_surfaces * 0 + neighbor_start_index] 
            * boundary_field_scalar[neighbor_start_index];
    double boussfyNeighbor = boundary_field_vector[num_boundary_surfaces * 1 + neighbor_start_index] 
            * boundary_field_scalar[neighbor_start_index];
    double boussfzNeighbor = boundary_field_vector[num_boundary_surfaces * 2 + neighbor_start_index] 
            * boundary_field_scalar[neighbor_start_index];
    
    double boussfxInternal = boundary_field_vector[num_boundary_surfaces * 0 + internal_start_index] 
            * boundary_field_scalar[internal_start_index];
    double boussfyInternal = boundary_field_vector[num_boundary_surfaces * 1 + internal_start_index] 
            * boundary_field_scalar[internal_start_index];
    double boussfzInternal = boundary_field_vector[num_boundary_surfaces * 2 + internal_start_index] 
            * boundary_field_scalar[internal_start_index];
    
    double boussfx = (1 - bouWeight) * boussfxNeighbor + bouWeight * boussfxInternal;
    double boussfy = (1 - bouWeight) * boussfyNeighbor + bouWeight * boussfyInternal;
    double boussfz = (1 - bouWeight) * boussfzNeighbor + bouWeight * boussfzInternal;

    output[neighbor_start_index] = boundary_phi_old[neighbor_start_index] - (bouSfx * boussfx + bouSfy * boussfy + bouSfz * boussfz);
}

__global__ void get_ddtCorr_internal_kernel(int num_cells, int num_surfaces, 
        const double *phiCorr, const double *phi, const double rDeltaT,
        double *output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_surfaces)
        return;
    
    double phiCorrVal = phiCorr[index];
    double phiVal = phi[index];

    double tddtCouplingCoeff = 1. - min(fabs(phiCorrVal)/fabs(phiVal) + SMALL, 1.);
    
    output[index] = tddtCouplingCoeff * rDeltaT * phiCorrVal;
}

__global__ void get_ddtCorr_boundary_nonZero_kernel(int num_boundary_surfaces, int num, int offset,
        const double *boundary_phiCorr, const double *boundary_phi, const double rDeltaT,
        double *output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num)
        return;
    
    int start_index = offset + index;

    double bouPhiCorrVal = boundary_phiCorr[start_index];
    double bouPhiVal = boundary_phi[start_index];

    double bou_tddtCouplingCoeff = 1. - min(fabs(bouPhiCorrVal)/fabs(bouPhiVal) + SMALL, 1.);
    output[start_index] = bou_tddtCouplingCoeff * rDeltaT * bouPhiCorrVal;
}

__global__ void multi_fvc_flux_fvc_intepolate_internal_kernel(int num_cells, int num_surfaces, 
        const int *lower_index, const int *upper_index,
        const double *field_vector, const double *vf, const double *weight, const double *face_vector,
        double *output, double sign)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_surfaces)
        return;
    
    double w = weight[index];
    int owner = lower_index[index];
    int neighbor = upper_index[index];

    // fvc_flux_HbyA
    double Sfx = face_vector[num_surfaces * 0 + index];
    double Sfy = face_vector[num_surfaces * 1 + index];
    double Sfz = face_vector[num_surfaces * 2 + index];

    double ssfx = (w * (field_vector[num_cells * 0 + owner] - field_vector[num_cells * 0 + neighbor]) + field_vector[num_cells * 0 + neighbor]);
    double ssfy = (w * (field_vector[num_cells * 1 + owner] - field_vector[num_cells * 1 + neighbor]) + field_vector[num_cells * 1 + neighbor]);
    double ssfz = (w * (field_vector[num_cells * 2 + owner] - field_vector[num_cells * 2 + neighbor]) + field_vector[num_cells * 2 + neighbor]);

    // fvc_interpolate_rho
    double vf_interp = (w * (vf[owner] - vf[neighbor]) + vf[neighbor]);

    output[index] += (Sfx * ssfx + Sfy * ssfy + Sfz * ssfz) * vf_interp;
}

__global__ void multi_fvc_flux_fvc_intepolate_boundary_kernel_zeroGradient(int num_boundary_surfaces, int num, int offset, 
        const double *boundary_face_vector, const double *boundary_field_vector, 
        const double *boundary_vf, double *output, double sign)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num)
        return;
    
    int start_index = offset + index;

    double bouSfx = boundary_face_vector[num_boundary_surfaces * 0 + start_index];
    double bouSfy = boundary_face_vector[num_boundary_surfaces * 1 + start_index];
    double bouSfz = boundary_face_vector[num_boundary_surfaces * 2 + start_index];

    double boussfx = boundary_field_vector[num_boundary_surfaces * 0 + start_index];
    double boussfy = boundary_field_vector[num_boundary_surfaces * 1 + start_index];
    double boussfz = boundary_field_vector[num_boundary_surfaces * 2 + start_index];

    output[start_index] += (bouSfx * boussfx + bouSfy * boussfy + bouSfz * boussfz) * boundary_vf[start_index];
}

__global__ void multi_fvc_flux_fvc_intepolate_boundary_kernel_processor(int num_boundary_surfaces, int num, int offset, 
        const double *boundary_face_vector, const double *boundary_field_vector, const double *boundary_weight,
        const double *boundary_vf, double *output, double sign)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num)
        return;
    
    int neighbor_start_index = offset + index;
    int internal_start_index = offset + num + index;

    double bouWeight = boundary_weight[neighbor_start_index];

    double bouSfx = boundary_face_vector[num_boundary_surfaces * 0 + neighbor_start_index];
    double bouSfy = boundary_face_vector[num_boundary_surfaces * 1 + neighbor_start_index];
    double bouSfz = boundary_face_vector[num_boundary_surfaces * 2 + neighbor_start_index];

    // interpolate boundary vector
    double boussfx = (1 - bouWeight) * boundary_field_vector[num_boundary_surfaces * 0 + neighbor_start_index] + 
            bouWeight * boundary_field_vector[num_boundary_surfaces * 0 + internal_start_index];
    double boussfy = (1 - bouWeight) * boundary_field_vector[num_boundary_surfaces * 1 + neighbor_start_index] + 
            bouWeight * boundary_field_vector[num_boundary_surfaces * 1 + internal_start_index];
    double boussfz = (1 - bouWeight) * boundary_field_vector[num_boundary_surfaces * 2 + neighbor_start_index] + 
            bouWeight * boundary_field_vector[num_boundary_surfaces * 2 + internal_start_index];
    
    // interpolate boundary scalar
    double bouvf = (1 - bouWeight) * boundary_vf[neighbor_start_index] + bouWeight * boundary_vf[internal_start_index];
    
    output[neighbor_start_index] += (bouSfx * boussfx + bouSfy * boussfy + bouSfz * boussfz) * bouvf;
}

__global__ void correct_diag_mtx_multi_tpsi_kernel(int num_cells, const double *psi, const double *thermo_psi, 
        double *source, double *diag)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    // correction: source += (-diag * psi + source)
    double srcVal = source[index];
    double APsi = - diag[index] * psi[index] + srcVal;
    source[index] -= APsi;

    // multi psi
    double tPsiVal = thermo_psi[index];
    source[index] *= tPsiVal;
    diag[index] *= tPsiVal;
}

double* dfpEqn::getFieldPointer(const char* fieldAlias, location loc, position pos) {
    char mergedName[256];
    if (pos == position::internal) {
        sprintf(mergedName, "%s_%s", (loc == location::cpu) ? "h" : "d", fieldAlias);
    } else if (pos == position::boundary) {
        sprintf(mergedName, "%s_boundary_%s", (loc == location::cpu) ? "h" : "d", fieldAlias);
    }

    double *pointer = nullptr;
    if (fieldPointerMap.find(std::string(mergedName)) != fieldPointerMap.end()) {
        pointer = fieldPointerMap[std::string(mergedName)];
    }
    if (pointer == nullptr) {
        fprintf(stderr, "Warning! getFieldPointer of %s returns nullptr!\n", mergedName);
    }
    //fprintf(stderr, "fieldAlias: %s, mergedName: %s, pointer: %p\n", fieldAlias, mergedName, pointer);

    return pointer;
}

void dfpEqn::setConstantValues(const std::string &mode_string, const std::string &setting_path,
                GAMGStruct *GAMGdata_, int agglomeration_level) {
    this->stream = dataBase_.stream;
    this->mode_string = mode_string;
    this->setting_path = setting_path;
#ifdef AMGX_
    pSolver = new AmgXSolver(mode_string, setting_path, dataBase_.localRank);
#endif

#ifdef CSR_
    bool useGAMG = true;
    pCSRSolver = new PCGCSRSolver();
    pCSRSolver->initSolvePerformance
    (     
        1e-20, //small_
        2.22507e-308, //vsmall_
        4, //maxIter_ 
        4, //minIter_ 
        1e-9, //tolerance_ 
        0.01 //relTol_
    );
    if (useGAMG)
    {
        pCSRSolver->initializeGAMG(dataBase_, dataBase_.num_cells, dataBase_.boundary_surface_value_bytes,
                            GAMGdata_, agglomeration_level);
    }
    else 
    {
        pCSRSolver->initialize(dataBase_.num_cells, dataBase_.boundary_surface_value_bytes);
    }
#endif

#ifdef ELL_
    bool useGAMG = true;
    pELLSolver = new PCGELLSolver();
    pELLSolver->initSolvePerformance
    (     
        1e-20, //small_
        2.22507e-308, //vsmall_
        4, //maxIter_ 
        4, //minIter_ 
        1e-9, //tolerance_ 
        0.01 //relTol_
    );
    if (useGAMG)
    {
        pELLSolver->initializeGAMG(dataBase_, dataBase_.num_cells, dataBase_.boundary_surface_value_bytes,
                            GAMGdata_, agglomeration_level);
    }
    else 
    {
        pELLSolver->initialize(dataBase_.num_cells, dataBase_.boundary_surface_value_bytes);
    }
#endif
}

void dfpEqn::setConstantFields(const std::vector<int> patch_type_U, const std::vector<int> patch_type_p) {
    this->patch_type_U = patch_type_U;
    this->patch_type_p = patch_type_p;
}

void dfpEqn::createNonConstantFieldsInternal() {
#ifndef STREAM_ALLOCATOR
    // intermediate fields
    checkCudaErrors(cudaMalloc((void**)&d_rhorAUf, dataBase_.surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_phiHbyA, dataBase_.surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_flux, dataBase_.surface_value_bytes));
#endif
}

void dfpEqn::createNonConstantFieldsBoundary() {
#ifndef STREAM_ALLOCATOR
    // boundary coeffs
    checkCudaErrors(cudaMalloc((void**)&d_value_internal_coeffs, dataBase_.boundary_surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_value_boundary_coeffs, dataBase_.boundary_surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_gradient_internal_coeffs, dataBase_.boundary_surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_gradient_boundary_coeffs, dataBase_.boundary_surface_value_bytes));
    // intermediate boundary fields
    checkCudaErrors(cudaMalloc((void**)&d_boundary_rhorAUf, dataBase_.boundary_surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_phiHbyA, dataBase_.boundary_surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_flux, dataBase_.boundary_surface_value_bytes));
#endif
}

void dfpEqn::createNonConstantLduAndCsrFields() {
    // ldu and csr
    checkCudaErrors(cudaMalloc((void**)&d_ldu, dataBase_.csr_value_bytes));
#ifdef AMGX_
    d_lower = d_ldu;
    d_diag = d_ldu + dataBase_.num_surfaces;
    d_upper = d_ldu + dataBase_.num_cells + dataBase_.num_surfaces;
#endif
#ifdef CSR_
    d_lower = d_ldu;
    d_upper = d_ldu + dataBase_.num_surfaces;
    d_diag = d_ldu + 2 * dataBase_.num_surfaces;
#endif
#ifdef ELL_
    d_lower = d_ldu;
    d_upper = d_ldu + dataBase_.num_surfaces;
    d_diag = d_ldu + 2 * dataBase_.num_surfaces;
#endif
    d_extern = d_ldu + dataBase_.num_cells + 2 * dataBase_.num_surfaces;
#ifndef STREAM_ALLOCATOR
    checkCudaErrors(cudaMalloc((void**)&d_source, dataBase_.cell_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_internal_coeffs, dataBase_.boundary_surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_coeffs, dataBase_.boundary_surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_A, dataBase_.csr_value_bytes));
#endif
}

void dfpEqn::GAMGpEqnldu(GAMGStruct *GAMGdata_, int agglomeration_level, int type){
    GAMGlduPtr.resize(agglomeration_level);
    //csr
    if(type ==1){
        GAMGldu2csrPtr.resize(agglomeration_level);
        for(int i = 0; i < agglomeration_level; i++){
            checkCudaErrors(cudaMalloc((void**)&GAMGlduPtr[i], GAMGdata_[i].nFace * 2 * sizeof(double)));
            checkCudaErrors(cudaMalloc((void**)&GAMGldu2csrPtr[i], GAMGdata_[i].nFace * 2 * sizeof(int)));
            checkCudaErrors(cudaMemcpy(GAMGldu2csrPtr[i], dataBase_.h_ldu_to_csr_no_diag[i], GAMGdata_[i].nFace * 2 * sizeof(int), cudaMemcpyHostToDevice));
        }
    }
    //ell
    if(type==2){
        GAMGldu2ellPtr.resize(agglomeration_level);
        for(int i = 0; i < agglomeration_level; i++){
            checkCudaErrors(cudaMalloc((void**)&GAMGlduPtr[i], GAMGdata_[i].nFace * 2 * sizeof(double)));
            checkCudaErrors(cudaMalloc((void**)&GAMGldu2ellPtr[i], GAMGdata_[i].nFace * 2 * sizeof(int)));
            checkCudaErrors(cudaMemcpy(GAMGldu2ellPtr[i], dataBase_.h_ldu2ellIndex[i], GAMGdata_[i].nFace * 2 * sizeof(int), cudaMemcpyHostToDevice));
        }
    }

}

void dfpEqn::initNonConstantFields(const double *p, const double *boundary_p){
    checkCudaErrors(cudaMemcpyAsync(dataBase_.d_p, dataBase_.h_p, dataBase_.cell_value_bytes, cudaMemcpyHostToDevice, dataBase_.stream));
    checkCudaErrors(cudaMemcpyAsync(dataBase_.d_boundary_p, dataBase_.h_boundary_p, dataBase_.boundary_surface_value_bytes, cudaMemcpyHostToDevice, dataBase_.stream));
}

void dfpEqn::cleanCudaResources() {
#ifdef USE_GRAPH
    if (pre_graph_created) {
        checkCudaErrors(cudaGraphExecDestroy(graph_instance_pre));
        checkCudaErrors(cudaGraphDestroy(graph_pre));
    }
    if (post_graph_created) {
        checkCudaErrors(cudaGraphExecDestroy(graph_instance_post));
        checkCudaErrors(cudaGraphDestroy(graph_post));
    }
#endif
}

// tmp
void dfpEqn::preProcess(double *h_phi, double *h_boundary_phi) {
    checkCudaErrors(cudaMemcpyAsync(dataBase_.d_phi, h_phi, dataBase_.surface_value_bytes, cudaMemcpyHostToDevice, dataBase_.stream));
    checkCudaErrors(cudaMemcpyAsync(dataBase_.d_boundary_phi, h_boundary_phi, dataBase_.boundary_surface_value_bytes, cudaMemcpyHostToDevice, dataBase_.stream));
};

void dfpEqn::correctPsi(const double *h_thermoPsi, double *h_boundary_thermoPsi) {
    checkCudaErrors(cudaMemcpyAsync(dataBase_.d_thermo_psi, h_thermoPsi, dataBase_.cell_value_bytes, cudaMemcpyHostToDevice, dataBase_.stream));
    checkCudaErrors(cudaMemcpyAsync(dataBase_.d_boundary_thermo_psi, h_boundary_thermoPsi, dataBase_.boundary_surface_value_bytes, cudaMemcpyHostToDevice, dataBase_.stream));
};
void dfpEqn::correctP(const double *h_p, double *h_boundary_p) {
    checkCudaErrors(cudaMemcpyAsync(dataBase_.d_p, h_p, dataBase_.cell_value_bytes, cudaMemcpyHostToDevice, dataBase_.stream));
    checkCudaErrors(cudaMemcpyAsync(dataBase_.d_boundary_p, h_boundary_p, dataBase_.boundary_surface_value_bytes, cudaMemcpyHostToDevice, dataBase_.stream));
};

void dfpEqn::process(GAMGStruct *GAMGdata, int agglomeration_level) {
    TICK_INIT_EVENT;
    TICK_START_EVENT;
#ifdef USE_GRAPH
    if(!pre_graph_created) {
        DEBUG_TRACE;
        checkCudaErrors(cudaStreamBeginCapture(dataBase_.stream, cudaStreamCaptureModeGlobal));
#endif

#ifdef STREAM_ALLOCATOR
    // intermediate fields
    checkCudaErrors(cudaMallocAsync((void**)&d_rhorAUf, dataBase_.surface_value_bytes, dataBase_.stream));
    checkCudaErrors(cudaMallocAsync((void**)&d_phiHbyA, dataBase_.surface_value_bytes, dataBase_.stream));
    checkCudaErrors(cudaMallocAsync((void**)&d_flux, dataBase_.surface_value_bytes, dataBase_.stream));

    // boundary coeffs
    checkCudaErrors(cudaMallocAsync((void**)&d_value_internal_coeffs, dataBase_.boundary_surface_value_bytes, dataBase_.stream));
    checkCudaErrors(cudaMallocAsync((void**)&d_value_boundary_coeffs, dataBase_.boundary_surface_value_bytes, dataBase_.stream));
    checkCudaErrors(cudaMallocAsync((void**)&d_gradient_internal_coeffs, dataBase_.boundary_surface_value_bytes, dataBase_.stream));
    checkCudaErrors(cudaMallocAsync((void**)&d_gradient_boundary_coeffs, dataBase_.boundary_surface_value_bytes, dataBase_.stream));
    // intermediate boundary fields
    checkCudaErrors(cudaMallocAsync((void**)&d_boundary_rhorAUf, dataBase_.boundary_surface_value_bytes, dataBase_.stream));
    checkCudaErrors(cudaMallocAsync((void**)&d_boundary_phiHbyA, dataBase_.boundary_surface_value_bytes, dataBase_.stream));
    checkCudaErrors(cudaMallocAsync((void**)&d_boundary_flux, dataBase_.boundary_surface_value_bytes, dataBase_.stream));

    // ldu and csr
    checkCudaErrors(cudaMallocAsync((void**)&d_source, dataBase_.cell_value_bytes, dataBase_.stream));
    checkCudaErrors(cudaMallocAsync((void**)&d_internal_coeffs, dataBase_.boundary_surface_value_bytes, dataBase_.stream));
    checkCudaErrors(cudaMallocAsync((void**)&d_boundary_coeffs, dataBase_.boundary_surface_value_bytes, dataBase_.stream));
    checkCudaErrors(cudaMallocAsync((void**)&d_A, dataBase_.csr_value_bytes, dataBase_.stream));
#endif

    checkCudaErrors(cudaMemsetAsync(d_ldu, 0, dataBase_.csr_value_bytes, dataBase_.stream)); // d_ldu contains d_lower, d_diag, and d_upper
    checkCudaErrors(cudaMemsetAsync(d_source, 0, dataBase_.cell_value_bytes, dataBase_.stream));
    checkCudaErrors(cudaMemsetAsync(d_internal_coeffs, 0, dataBase_.boundary_surface_value_bytes, dataBase_.stream));
    checkCudaErrors(cudaMemsetAsync(d_boundary_coeffs, 0, dataBase_.boundary_surface_value_bytes, dataBase_.stream));
    checkCudaErrors(cudaMemsetAsync(d_A, 0, dataBase_.csr_value_bytes, dataBase_.stream));

    // intermediate parameters
    checkCudaErrors(cudaMemsetAsync(d_rhorAUf, 0, dataBase_.surface_value_bytes, dataBase_.stream));
    checkCudaErrors(cudaMemsetAsync(d_boundary_rhorAUf, 0, dataBase_.boundary_surface_value_bytes, dataBase_.stream));
    checkCudaErrors(cudaMemsetAsync(d_phiHbyA, 0, dataBase_.surface_value_bytes, dataBase_.stream));
    checkCudaErrors(cudaMemsetAsync(d_boundary_phiHbyA, 0, dataBase_.boundary_surface_value_bytes, dataBase_.stream));
    checkCudaErrors(cudaMemsetAsync(d_flux, 0, dataBase_.surface_value_bytes, dataBase_.stream)); // TODO: introduce of flux is not necessary
    
    update_boundary_coeffs_scalar(dataBase_.stream,
            dataBase_.num_patches, dataBase_.patch_size.data(), patch_type_p.data(),
            dataBase_.d_boundary_delta_coeffs, dataBase_.d_boundary_p, dataBase_.d_boundary_weight,
            d_value_internal_coeffs, d_value_boundary_coeffs,
            d_gradient_internal_coeffs, d_gradient_boundary_coeffs);
    getrhorAUf(dataBase_.stream, dataBase_.num_cells, dataBase_.num_surfaces, 
            dataBase_.d_owner, dataBase_.d_neighbor, dataBase_.d_weight, 
            dataBase_.d_rho, dataBase_.d_rAU, d_rhorAUf, // end for internal
            dataBase_.num_patches, dataBase_.patch_size.data(), dataBase_.patch_type_calculated.data(), dataBase_.d_boundary_weight,
            dataBase_.d_boundary_rho, dataBase_.d_boundary_rAU, d_boundary_rhorAUf);
    getphiHbyA(dataBase_.stream, dataBase_.num_cells, dataBase_.num_surfaces, dataBase_.num_boundary_surfaces, 
            dataBase_.rdelta_t, dataBase_.d_owner, dataBase_.d_neighbor, 
            dataBase_.d_weight, dataBase_.d_u_old, dataBase_.d_rho_old,
            dataBase_.d_phi_old, dataBase_.d_rho, d_rhorAUf, dataBase_.d_HbyA, dataBase_.d_sf, d_phiHbyA, // end for internal
            dataBase_.num_patches, dataBase_.patch_size.data(), dataBase_.patch_type_extropolated.data(),
            dataBase_.d_boundary_sf, dataBase_.d_boundary_u_old, dataBase_.d_boundary_rho, 
            dataBase_.d_boundary_rho_old, dataBase_.d_boundary_phi_old, d_boundary_rhorAUf, dataBase_.d_boundary_HbyA, 
            dataBase_.d_boundary_weight, d_boundary_phiHbyA, 1.0);
    fvm_ddt_scalar(dataBase_.stream, dataBase_.num_cells, dataBase_.rdelta_t, dataBase_.d_p_old, dataBase_.d_volume, d_diag, d_source);
    correctionDiagMtxMultiTPsi(dataBase_.stream, dataBase_.num_cells, dataBase_.d_p, dataBase_.d_thermo_psi, d_diag, d_source);
    fvc_ddt_scalar(dataBase_.stream, dataBase_.num_cells, dataBase_.rdelta_t, dataBase_.d_rho, dataBase_.d_rho_old, dataBase_.d_volume, 
            d_source, -1.);
    fvc_div_surface_scalar(dataBase_.stream, dataBase_.num_cells, dataBase_.num_surfaces, dataBase_.num_boundary_surfaces,
            dataBase_.d_owner, dataBase_.d_neighbor, d_phiHbyA, dataBase_.d_boundary_face_cell, 
            dataBase_.num_patches, dataBase_.patch_size.data(), patch_type_p.data(),
            d_boundary_phiHbyA, dataBase_.d_volume, d_source, -1.);
    fvm_laplacian_surface_scalar_vol_scalar(dataBase_.stream, dataBase_.num_surfaces, dataBase_.num_boundary_surfaces,
            dataBase_.d_owner, dataBase_.d_neighbor, dataBase_.d_mag_sf, dataBase_.d_delta_coeffs, d_rhorAUf, 
            d_lower, d_upper, d_diag, dataBase_.num_patches, dataBase_.patch_size.data(), patch_type_p.data(),
            dataBase_.d_boundary_mag_sf, d_boundary_rhorAUf, d_gradient_internal_coeffs, d_gradient_boundary_coeffs, 
            d_internal_coeffs, d_boundary_coeffs, -1.);
    
#ifdef AMGX_
    nvtxRangePushA("AMGX::ldutocsr()");
    ldu_to_csr_scalar(dataBase_.stream, dataBase_.num_cells, dataBase_.num_surfaces, dataBase_.num_boundary_surfaces,
            dataBase_.num_Nz, dataBase_.d_boundary_face_cell, dataBase_.d_ldu_to_csr_index,
            dataBase_.num_patches, dataBase_.patch_size.data(), patch_type_p.data(),
            d_ldu, d_source, d_internal_coeffs, d_boundary_coeffs, d_A);
    nvtxRangePop();
#endif

    // // load CPU data for test ---start
    // std::string abPath = "/root/gpuversion0417/tmp0418/test0514/cvodeIntegrator_64/";
    // int mpirank;
    // MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    // std::string c = std::to_string(mpirank);        
    // std::string sdiag = "diag";
    // std::string supper = "upper";
    // std::string slower = "lower";
    // std::string ssource = "source";
    // std::string sIntCoef = "IntCoef";
    // std::string sBouCoef = "BouCoef";
    // std::string b = ".bin";
    
    // bool loadCPUdata4test = true;
    // if (loadCPUdata4test){
    //     printf("*** LOADing data from files ***\n");
    //     // std::string filename_diag = "/root/gpuversion0417/tmp0418/test0514/cvodeIntegrator_64/diag.bin";
    //     // std::string filename_upper = "/root/gpuversion0417/tmp0418/test0514/cvodeIntegrator_64/upper.bin";
    //     // std::string filename_lower = "/root/gpuversion0417/tmp0418/test0514/cvodeIntegrator_64/lower.bin";
    //     // std::string filename_source = "/root/gpuversion0417/tmp0418/test0514/cvodeIntegrator_64/source.bin";
    //     std::string filename_diag = abPath + sdiag + c + b;
    //     std::string filename_upper = abPath + supper + c + b;
    //     std::string filename_lower = abPath + slower + c + b;
    //     std::string filename_source = abPath + ssource + c + b;
    //     std::string filename_IntCoef = abPath + sIntCoef + c + b;
    //     std::string filename_BouCoef = abPath + sBouCoef + c + b;
    //     double* data_diag;
    //     double* data_upper;
    //     double* data_lower;
    //     double* data_source;
    //     double* data_IntCoef;
    //     double* data_BouCoef;
    //     data_diag = (double*)malloc(sizeof(double)*dataBase_.num_cells);
    //     data_upper = (double*)malloc(sizeof(double)*dataBase_.num_surfaces);
    //     data_lower = (double*)malloc(sizeof(double)*dataBase_.num_surfaces);
    //     data_source = (double*)malloc(sizeof(double)*dataBase_.num_cells);
    //     data_IntCoef = (double*)malloc(sizeof(double)*dataBase_.num_boundary_surfaces);
    //     data_BouCoef = (double*)malloc(sizeof(double)*dataBase_.num_boundary_surfaces);;
    //     readArrayFromBinaryFile(filename_diag, data_diag, dataBase_.num_cells);
    //     readArrayFromBinaryFile(filename_upper, data_upper, dataBase_.num_surfaces);
    //     readArrayFromBinaryFile(filename_lower, data_lower, dataBase_.num_surfaces);
    //     readArrayFromBinaryFile(filename_source, data_source, dataBase_.num_cells);
    //     readArrayFromBinaryFile(filename_IntCoef, data_IntCoef, dataBase_.num_boundary_surfaces);
    //     readArrayFromBinaryFile(filename_BouCoef, data_BouCoef, dataBase_.num_boundary_surfaces);

    //     cudaMemcpy(d_diag,  &data_diag[0],  sizeof(double)*dataBase_.num_cells,    cudaMemcpyHostToDevice);
    //     cudaMemcpy(d_lower, &data_lower[0], sizeof(double)*dataBase_.num_surfaces, cudaMemcpyHostToDevice);
    //     cudaMemcpy(d_upper, &data_upper[0], sizeof(double)*dataBase_.num_surfaces, cudaMemcpyHostToDevice);
    //     cudaMemcpy(d_source, &data_source[0], sizeof(double)*dataBase_.num_cells, cudaMemcpyHostToDevice);
    //     cudaMemcpy(d_internal_coeffs, &data_IntCoef[0], sizeof(double)*dataBase_.num_boundary_surfaces, cudaMemcpyHostToDevice);
    //     cudaMemcpy(d_boundary_coeffs, &data_BouCoef[0], sizeof(double)*dataBase_.num_boundary_surfaces, cudaMemcpyHostToDevice);
    // }

#ifdef CSR_

        // ========================================================================================
        // startif use GAMG ... save for finest matrix
        std::cout << "=== start agglomeration in dfpEqn::process() " << std::endl;
        std::cout << "dataBase_.num_cells: " << dataBase_.num_cells 
                  << ", dataBase_.num_surfaces: " << dataBase_.num_surfaces
                  << ", dataBase_.num_boundary_surfaces: " << dataBase_.num_boundary_surfaces << std::endl;

        // malloc & memset coarse level
        nvtxRangePushA("Memset CSR Matrix");
        for(int leveli=0; leveli<agglomeration_level; leveli++)
        {
            std::cout << "malloc and memset leveli: " << leveli << std::endl;
            // Set matrix data
            if(leveli==0)
            {

                // --- addInternalCoeffs : diag ---
                // input : d_internal_coeffs
                addInternalCoeffs(dataBase_.stream, dataBase_.num_patches, dataBase_.patch_size, 
                    d_internal_coeffs, dataBase_.d_boundary_face_cell, d_diag, dataBase_.patch_type_extropolated.data());
                
                checkCudaErrors(cudaMemcpyAsync(GAMGdata[leveli].d_lower, d_lower, GAMGdata[leveli].nFace * sizeof(double), cudaMemcpyDeviceToDevice, dataBase_.stream));
                checkCudaErrors(cudaMemcpyAsync(GAMGdata[leveli].d_upper, d_upper, GAMGdata[leveli].nFace * sizeof(double), cudaMemcpyDeviceToDevice, dataBase_.stream));
                checkCudaErrors(cudaMemcpyAsync(GAMGdata[leveli].d_diag, d_diag, GAMGdata[leveli].nCell * sizeof(double), cudaMemcpyDeviceToDevice, dataBase_.stream));
                checkCudaErrors(cudaMemset(GAMGdata[leveli].d_off_diag_value, 0, GAMGdata[leveli].nFace * 2 * sizeof(double)));
                // checkCudaErrors(cudaMemset(GAMGdata[leveli].d_csr_row_index_no_diag, 0, (GAMGdata[leveli].nCell + 1) * sizeof(int)));
                // checkCudaErrors(cudaMemset(GAMGdata[leveli].d_csr_col_index_no_diag, 0, GAMGdata[leveli].nFace * 2 * sizeof(int)));
            }
            else
            {
                checkCudaErrors(cudaMemset(GAMGdata[leveli].d_lower, 0, GAMGdata[leveli].nFace * sizeof(double)));
                checkCudaErrors(cudaMemset(GAMGdata[leveli].d_upper, 0, GAMGdata[leveli].nFace * sizeof(double)));
                checkCudaErrors(cudaMemset(GAMGdata[leveli].d_diag, 0, GAMGdata[leveli].nCell * sizeof(double)));
                checkCudaErrors(cudaMemset(GAMGdata[leveli].d_off_diag_value, 0, GAMGdata[leveli].nFace * 2 * sizeof(double)));
                // checkCudaErrors(cudaMemset(GAMGdata[leveli].d_csr_row_index_no_diag, 0, (GAMGdata[leveli].nCell + 1) * sizeof(int)));
                // checkCudaErrors(cudaMemset(GAMGdata[leveli].d_csr_col_index_no_diag, 0, GAMGdata[leveli].nFace * 2 * sizeof(int)));
            }


#ifdef PARALLEL_
            if (leveli==0)
            {
                getInterfacesCoeffs(dataBase_.stream, dataBase_.num_patches, dataBase_.patch_size, 
                                    dataBase_.interfaceFlag, dataBase_.patch_type_extropolated.data(), 
                                    d_boundary_coeffs, d_internal_coeffs, 
                                    GAMGdata[0].d_interfaceBouCoeffs, GAMGdata[0].d_interfaceIntCoeffs);

                bool loadCPUCoeffdata4test = false;
                if (loadCPUCoeffdata4test){
                    int mpirank = dataBase_.localRank;
                    for(int patchi=0; patchi<GAMGdata[leveli].nPatchFaces.size(); patchi++)
                    {
                        if (GAMGdata[leveli].nPatchFaces[patchi] > 0)
                        {
                            std::stringstream filenameBouCoeff; 
                            std::stringstream filenameIntCoeff; 

                            filenameBouCoeff << "/root/0427/deepflame-dev/examples/dfLowMachFoam/cvodeIntegrator_64/h_bouCoeff_0_" << patchi << "_" << mpirank << "_Ref.txt";
                            filenameIntCoeff << "/root/0427/deepflame-dev/examples/dfLowMachFoam/cvodeIntegrator_64/h_intCoeff_0_" << patchi << "_" << mpirank << "_Ref.txt";

                            std::string filename_bouCoeff = filenameBouCoeff.str();
                            std::string filename_intCoeff = filenameIntCoeff.str();

                            std::vector<double> data_bouCoeff, data_intCoeff;

                            int offset = 0;
                            std::ifstream file_bouCoeff(filename_bouCoeff);
                            if (!file_bouCoeff.is_open()) {
                                    std::cerr << "error open filename_bouCoeff : " << filename_bouCoeff << std::endl;
                            }
                            double number_bouCoeff;
                            while (file_bouCoeff >> std::setprecision(10) >> number_bouCoeff){
                                if(offset < GAMGdata[leveli].nPatchFaces[patchi]){
                                    data_bouCoeff.push_back(number_bouCoeff);
                                }
                                offset++;
                            }
                            file_bouCoeff.close();

                            offset = 0;
                            std::ifstream file_intCoeff(filename_intCoeff);
                            if (!file_intCoeff.is_open()) {
                                    std::cerr << "error open filename_intCoeff : " << filename_intCoeff << std::endl;
                            }
                            double number_intCoeff;
                            while (file_intCoeff >> std::setprecision(10) >> number_intCoeff){
                                if(offset < GAMGdata[leveli].nPatchFaces[patchi]){
                                    data_intCoeff.push_back(number_intCoeff);
                                }
                                offset++;
                            }
                            file_intCoeff.close();

                            cudaMemcpy(GAMGdata[0].d_interfaceBouCoeffs[patchi], &data_bouCoeff[0], sizeof(double)*GAMGdata[leveli].nPatchFaces[patchi], cudaMemcpyHostToDevice);
                            cudaMemcpy(GAMGdata[0].d_interfaceIntCoeffs[patchi], &data_intCoeff[0], sizeof(double)*GAMGdata[leveli].nPatchFaces[patchi], cudaMemcpyHostToDevice);
                        }
                    }
                }
            }
            else
            {
                // Set interface coef data
                for(int patchi=0; patchi<GAMGdata[leveli].nPatchFaces.size(); patchi++)
                {
                    if (GAMGdata[leveli].nPatchFaces[patchi] > 0)
                    {
                        checkCudaErrors(cudaMemset(GAMGdata[leveli].d_interfaceBouCoeffs[patchi], 0, GAMGdata[leveli].nPatchFaces[patchi]*sizeof(double)));
                        checkCudaErrors(cudaMemset(GAMGdata[leveli].d_interfaceIntCoeffs[patchi], 0, GAMGdata[leveli].nPatchFaces[patchi]*sizeof(double)));
                    
                    }
                }
            }
#endif
        }
        nvtxRangePop();

        std::cout << "=============================================" << std::endl;
        std::cout << "====== call pCSRSolver->initGAMGMatrix ======" << std::endl;
        nvtxRangePushA("initGAMGMatrix()");
        pCSRSolver->initGAMGMatrix(dataBase_, GAMGdata, agglomeration_level);
        nvtxRangePop();
        std::cout << "====== end pCSRSolver->initGAMGMatrix =======" << std::endl;
        std::cout << "=============================================" << std::endl;

        // coarse level ldu to csr
        nvtxRangePushA("ldu to csr");
        for(int leveli=0; leveli<agglomeration_level; leveli++)
        {
            // checkCudaErrors(cudaMemcpy(GAMGlduPtr[leveli], GAMGdata[leveli].d_lower, GAMGdata[leveli].nFace * sizeof(double), cudaMemcpyDeviceToDevice));
            // checkCudaErrors(cudaMemcpy(GAMGlduPtr[leveli] + GAMGdata[leveli].nFace, GAMGdata[leveli].d_upper, GAMGdata[leveli].nFace * sizeof(double), cudaMemcpyDeviceToDevice));
            
            nvtxRangePushA("peqn_ldu_to_csr_no_diag()");
            // peqn_ldu_to_csr_no_diag(GAMGdata[leveli].nFace * 2, GAMGldu2csrPtr[leveli], GAMGlduPtr[leveli], GAMGdata[leveli].d_off_diag_value);
            peqn_ldu_to_csr_no_diag(GAMGdata[leveli].nFace * 2, GAMGldu2csrPtr[leveli], GAMGdata[leveli].d_lower, GAMGdata[leveli].d_upper, GAMGdata[leveli].d_off_diag_value);
            nvtxRangePop();

            // checkCudaErrors(cudaMemcpyAsync(GAMGdata[leveli].d_lowerAddr, &GAMGdata[leveli].lowerAddr[0], GAMGdata[leveli].nFace * sizeof(int), cudaMemcpyHostToDevice, dataBase_.stream));
            // checkCudaErrors(cudaMemcpyAsync(GAMGdata[leveli].d_upperAddr, &GAMGdata[leveli].upperAddr[0], GAMGdata[leveli].nFace * sizeof(int), cudaMemcpyHostToDevice, dataBase_.stream));
            // peqn_ldu2csr(dataBase_.stream, GAMGdata[leveli].nCell, GAMGdata[leveli].nFace, GAMGdata[leveli].d_lower, GAMGdata[leveli].d_upper, GAMGdata[leveli].d_lowerAddr, GAMGdata[leveli].d_upperAddr, 
            //     GAMGdata[leveli].d_off_diag_value, GAMGdata[leveli].d_csr_row_index_no_diag, GAMGdata[leveli].d_csr_col_index_no_diag);
        }
        nvtxRangePop();
#endif

#ifdef ELL_
        // ========================================================================================
        // startif use GAMG ... save for finest matrix
        std::cout << "=== start agglomeration in dfpEqn::process() " << std::endl;
        std::cout << "dataBase_.num_cells: " << dataBase_.num_cells 
                  << ", dataBase_.num_surfaces: " << dataBase_.num_surfaces
                  << ", dataBase_.num_boundary_surfaces: " << dataBase_.num_boundary_surfaces << std::endl;

        // malloc & memset coarse level
        nvtxRangePushA("Memset ELL Matrix");
        for(int leveli=0; leveli<agglomeration_level; leveli++)
        {
            std::cout << "malloc and memset leveli: " << leveli << std::endl;
            // Set matrix data
            if(leveli==0)
            {

                // --- addInternalCoeffs : diag ---
                // input : d_internal_coeffs
                addInternalCoeffs(dataBase_.stream, dataBase_.num_patches, dataBase_.patch_size, 
                    d_internal_coeffs, dataBase_.d_boundary_face_cell, d_diag, dataBase_.patch_type_extropolated.data());
                
                checkCudaErrors(cudaMemcpyAsync(GAMGdata[leveli].d_lower, d_lower, GAMGdata[leveli].nFace * sizeof(double), cudaMemcpyDeviceToDevice, dataBase_.stream));
                checkCudaErrors(cudaMemcpyAsync(GAMGdata[leveli].d_upper, d_upper, GAMGdata[leveli].nFace * sizeof(double), cudaMemcpyDeviceToDevice, dataBase_.stream));
                checkCudaErrors(cudaMemcpyAsync(GAMGdata[leveli].d_diag, d_diag, GAMGdata[leveli].nCell * sizeof(double), cudaMemcpyDeviceToDevice, dataBase_.stream));
                // checkCudaErrors(cudaMemset(GAMGdata[leveli].d_ell_cols, 0, GAMGdata[leveli].ell_row_maxcount * GAMGdata[leveli].nCell * sizeof(int)));
                checkCudaErrors(cudaMemset(GAMGdata[leveli].d_ell_values, 0, GAMGdata[leveli].ell_row_maxcount * GAMGdata[leveli].nCell * sizeof(double)));
            }
            else
            {
                checkCudaErrors(cudaMemset(GAMGdata[leveli].d_lower, 0, GAMGdata[leveli].nFace * sizeof(double)));
                checkCudaErrors(cudaMemset(GAMGdata[leveli].d_upper, 0, GAMGdata[leveli].nFace * sizeof(double)));
                checkCudaErrors(cudaMemset(GAMGdata[leveli].d_diag, 0, GAMGdata[leveli].nCell * sizeof(double)));
                // checkCudaErrors(cudaMemset(GAMGdata[leveli].d_ell_cols, 0, GAMGdata[leveli].ell_row_maxcount * GAMGdata[leveli].nCell * sizeof(int)));
                checkCudaErrors(cudaMemset(GAMGdata[leveli].d_ell_values, 0, GAMGdata[leveli].ell_row_maxcount * GAMGdata[leveli].nCell * sizeof(double)));

            }


#ifdef PARALLEL_
            if (leveli==0)
            {
                getInterfacesCoeffs(dataBase_.stream, dataBase_.num_patches, dataBase_.patch_size, 
                                    dataBase_.interfaceFlag, dataBase_.patch_type_extropolated.data(), 
                                    d_boundary_coeffs, d_internal_coeffs, 
                                    GAMGdata[0].d_interfaceBouCoeffs, GAMGdata[0].d_interfaceIntCoeffs);

                bool loadCPUCoeffdata4test = false;
                if (loadCPUCoeffdata4test){
                    int mpirank = dataBase_.localRank;
                    for(int patchi=0; patchi<GAMGdata[leveli].nPatchFaces.size(); patchi++)
                    {
                        if (GAMGdata[leveli].nPatchFaces[patchi] > 0)
                        {
                            std::stringstream filenameBouCoeff; 
                            std::stringstream filenameIntCoeff; 

                            filenameBouCoeff << "/root/0427/deepflame-dev/examples/dfLowMachFoam/cvodeIntegrator_64/h_bouCoeff_0_" << patchi << "_" << mpirank << "_Ref.txt";
                            filenameIntCoeff << "/root/0427/deepflame-dev/examples/dfLowMachFoam/cvodeIntegrator_64/h_intCoeff_0_" << patchi << "_" << mpirank << "_Ref.txt";

                            std::string filename_bouCoeff = filenameBouCoeff.str();
                            std::string filename_intCoeff = filenameIntCoeff.str();

                            std::vector<double> data_bouCoeff, data_intCoeff;

                            int offset = 0;
                            std::ifstream file_bouCoeff(filename_bouCoeff);
                            if (!file_bouCoeff.is_open()) {
                                    std::cerr << "error open filename_bouCoeff : " << filename_bouCoeff << std::endl;
                            }
                            double number_bouCoeff;
                            while (file_bouCoeff >> std::setprecision(10) >> number_bouCoeff){
                                if(offset < GAMGdata[leveli].nPatchFaces[patchi]){
                                    data_bouCoeff.push_back(number_bouCoeff);
                                }
                                offset++;
                            }
                            file_bouCoeff.close();

                            offset = 0;
                            std::ifstream file_intCoeff(filename_intCoeff);
                            if (!file_intCoeff.is_open()) {
                                    std::cerr << "error open filename_intCoeff : " << filename_intCoeff << std::endl;
                            }
                            double number_intCoeff;
                            while (file_intCoeff >> std::setprecision(10) >> number_intCoeff){
                                if(offset < GAMGdata[leveli].nPatchFaces[patchi]){
                                    data_intCoeff.push_back(number_intCoeff);
                                }
                                offset++;
                            }
                            file_intCoeff.close();

                            cudaMemcpy(GAMGdata[0].d_interfaceBouCoeffs[patchi], &data_bouCoeff[0], sizeof(double)*GAMGdata[leveli].nPatchFaces[patchi], cudaMemcpyHostToDevice);
                            cudaMemcpy(GAMGdata[0].d_interfaceIntCoeffs[patchi], &data_intCoeff[0], sizeof(double)*GAMGdata[leveli].nPatchFaces[patchi], cudaMemcpyHostToDevice);
                        }
                    }
                }
            }
            else
            {
                // Set interface coef data
                for(int patchi=0; patchi<GAMGdata[leveli].nPatchFaces.size(); patchi++)
                {
                    if (GAMGdata[leveli].nPatchFaces[patchi] > 0)
                    {
                        std::cout << "memset 0 in patch: " << patchi << std::endl;

                        checkCudaErrors(cudaMemset(GAMGdata[leveli].d_interfaceBouCoeffs[patchi], 0, GAMGdata[leveli].nPatchFaces[patchi]*sizeof(double)));
                        checkCudaErrors(cudaMemset(GAMGdata[leveli].d_interfaceIntCoeffs[patchi], 0, GAMGdata[leveli].nPatchFaces[patchi]*sizeof(double)));
                    
                    }
                }
            }
#endif
        }
        nvtxRangePop();

        std::cout << "=============================================" << std::endl;
        std::cout << "====== call pELLSolver->initGAMGMatrix ======" << std::endl;
        nvtxRangePushA("initGAMGMatrix()");
        pELLSolver->initGAMGMatrix(dataBase_, GAMGdata, agglomeration_level);
        nvtxRangePop();
        std::cout << "====== end pELLSolver->initGAMGMatrix =======" << std::endl;
        std::cout << "=============================================" << std::endl;

        // coarse level ldu to ell
        nvtxRangePushA("ldu to ell");
        for(int leveli=0; leveli<agglomeration_level; leveli++)
        {

            // ldu2ell
            // GAMGdata[leveli].ell_row_maxcount = dataBase_.h_ell_row_maxcount[leveli];
            // checkCudaErrors(cudaMemcpy(GAMGdata[leveli].d_ell_cols, dataBase_.h_ellCols[leveli], GAMGdata[leveli].nCell * GAMGdata[leveli].ell_row_maxcount * sizeof(int), cudaMemcpyHostToDevice));
            // checkCudaErrors(cudaMemcpyAsync(GAMGdata[leveli].d_lowerAddr, &GAMGdata[leveli].lowerAddr[0], GAMGdata[leveli].nFace * sizeof(int), cudaMemcpyHostToDevice, dataBase_.stream));
            // checkCudaErrors(cudaMemcpyAsync(GAMGdata[leveli].d_upperAddr, &GAMGdata[leveli].upperAddr[0], GAMGdata[leveli].nFace * sizeof(int), cudaMemcpyHostToDevice, dataBase_.stream));
            // checkCudaErrors(cudaMemcpy(GAMGlduPtr[leveli], GAMGdata[leveli].d_lower, GAMGdata[leveli].nFace * sizeof(double), cudaMemcpyDeviceToDevice));
            // checkCudaErrors(cudaMemcpy(GAMGlduPtr[leveli] + GAMGdata[leveli].nFace, GAMGdata[leveli].d_upper, GAMGdata[leveli].nFace * sizeof(double), cudaMemcpyDeviceToDevice));
            nvtxRangePushA("peqn_ldu_to_ell_no_diag()");
            // peqn_ldu_to_ell_no_diag(GAMGdata[leveli].nFace * 2, GAMGldu2ellPtr[leveli], GAMGlduPtr[leveli], GAMGdata[leveli].d_ell_values);
            peqn_ldu_to_ell_no_diag(GAMGdata[leveli].nFace * 2, GAMGldu2ellPtr[leveli], GAMGdata[leveli].d_lower, GAMGdata[leveli].d_upper, GAMGdata[leveli].d_ell_values);
            nvtxRangePop();
        }
        nvtxRangePop();

#endif

#ifdef USE_GRAPH
        checkCudaErrors(cudaStreamEndCapture(dataBase_.stream, &graph_pre));
        checkCudaErrors(cudaGraphInstantiate(&graph_instance_pre, graph_pre, NULL, NULL, 0));
        pre_graph_created = true;
    }
    DEBUG_TRACE;
    checkCudaErrors(cudaGraphLaunch(graph_instance_pre, dataBase_.stream));
#endif
    TICK_END_EVENT(pEqn assembly);

    TICK_START_EVENT;
    solve(GAMGdata, agglomeration_level);
    TICK_END_EVENT(pEqn solve);

#ifdef USE_GRAPH
    if(!post_graph_created) {
        checkCudaErrors(cudaStreamBeginCapture(dataBase_.stream, cudaStreamCaptureModeGlobal));
#endif
    
        TICK_START_EVENT;
        correct_boundary_conditions_scalar(dataBase_.stream, dataBase_.nccl_comm, dataBase_.neighbProcNo.data(),
                dataBase_.num_boundary_surfaces, dataBase_.num_patches, dataBase_.patch_size.data(), 
                patch_type_p.data(), dataBase_.d_boundary_delta_coeffs,
                dataBase_.d_boundary_face_cell, dataBase_.d_p, dataBase_.d_boundary_p,
                dataBase_.cyclicNeighbor.data(), dataBase_.patchSizeOffset.data(), dataBase_.d_boundary_weight);
        // update phi
        fvMtx_flux(dataBase_.stream, dataBase_.num_surfaces, dataBase_.num_boundary_surfaces, dataBase_.d_owner, dataBase_.d_neighbor, 
                d_lower, d_upper, dataBase_.d_p, d_flux,
                dataBase_.num_patches, dataBase_.patch_size.data(), patch_type_p.data(), 
                dataBase_.d_boundary_face_cell, d_internal_coeffs, d_boundary_coeffs, dataBase_.cyclicNeighbor.data(), 
                dataBase_.patchSizeOffset.data(), dataBase_.d_boundary_p, d_boundary_flux);
        field_add_scalar(dataBase_.stream, dataBase_.num_surfaces, d_phiHbyA, d_flux, dataBase_.d_phi, 
                dataBase_.num_boundary_surfaces, d_boundary_phiHbyA, d_boundary_flux, dataBase_.d_boundary_phi);
        // correct U
        checkCudaErrors(cudaMemsetAsync(dataBase_.d_u, 0., dataBase_.cell_value_vec_bytes, dataBase_.stream));
        // TODO: may do not need to calculate boundary fields
        fvc_grad_cell_scalar(dataBase_.stream, dataBase_.num_cells, dataBase_.num_surfaces, dataBase_.num_boundary_surfaces,
                dataBase_.d_owner, dataBase_.d_neighbor, 
                dataBase_.d_weight, dataBase_.d_sf, dataBase_.d_p, dataBase_.d_u, 
                dataBase_.num_patches, dataBase_.patch_size.data(), patch_type_p.data(), dataBase_.d_boundary_weight,
                dataBase_.d_boundary_face_cell, dataBase_.d_boundary_p, dataBase_.d_boundary_sf, dataBase_.d_volume, true);
        scalar_field_multiply_vector_field(dataBase_.stream, dataBase_.num_cells, dataBase_.d_rAU, dataBase_.d_u, dataBase_.d_u);
        field_add_vector(dataBase_.stream, dataBase_.num_cells, dataBase_.d_HbyA, dataBase_.d_u, dataBase_.d_u, -1.);
        correct_boundary_conditions_vector(dataBase_.stream, dataBase_.nccl_comm, dataBase_.neighbProcNo.data(), dataBase_.num_boundary_surfaces, 
                dataBase_.num_cells, dataBase_.num_patches, dataBase_.patch_size.data(), patch_type_U.data(), dataBase_.d_boundary_weight,
                dataBase_.d_boundary_face_cell, dataBase_.d_u, dataBase_.d_boundary_u, 
                dataBase_.cyclicNeighbor.data(), dataBase_.patchSizeOffset.data());
        vector_half_mag_square(dataBase_.stream, dataBase_.num_cells, dataBase_.d_u, dataBase_.d_k, dataBase_.num_boundary_surfaces, 
                dataBase_.d_boundary_u, dataBase_.d_boundary_k);
        // calculate dpdt
        fvc_ddt_scalar_field(dataBase_.stream, dataBase_.num_cells, dataBase_.rdelta_t, dataBase_.d_p, dataBase_.d_p_old, dataBase_.d_volume, dataBase_.d_dpdt, 1.);

#ifdef STREAM_ALLOCATOR
        // intermediate fields
        checkCudaErrors(cudaFreeAsync(d_rhorAUf, dataBase_.stream));
        checkCudaErrors(cudaFreeAsync(d_phiHbyA, dataBase_.stream));
        checkCudaErrors(cudaFreeAsync(d_flux, dataBase_.stream));

        // boundary coeffs
        checkCudaErrors(cudaFreeAsync(d_value_internal_coeffs, dataBase_.stream));
        checkCudaErrors(cudaFreeAsync(d_value_boundary_coeffs, dataBase_.stream));
        checkCudaErrors(cudaFreeAsync(d_gradient_internal_coeffs, dataBase_.stream));
        checkCudaErrors(cudaFreeAsync(d_gradient_boundary_coeffs, dataBase_.stream));
        // intermediate boundary fields
        checkCudaErrors(cudaFreeAsync(d_boundary_rhorAUf, dataBase_.stream));
        checkCudaErrors(cudaFreeAsync(d_boundary_phiHbyA, dataBase_.stream));
        checkCudaErrors(cudaFreeAsync(d_boundary_flux, dataBase_.stream));

        // ldu and csr
        checkCudaErrors(cudaFreeAsync(d_source, dataBase_.stream));
        checkCudaErrors(cudaFreeAsync(d_internal_coeffs, dataBase_.stream));
        checkCudaErrors(cudaFreeAsync(d_boundary_coeffs, dataBase_.stream));
        checkCudaErrors(cudaFreeAsync(d_A, dataBase_.stream));
#endif
        TICK_END_EVENT(pEqn post process all);

#ifdef USE_GRAPH
        checkCudaErrors(cudaStreamEndCapture(dataBase_.stream, &graph_post));
        checkCudaErrors(cudaGraphInstantiate(&graph_instance_post, graph_post, NULL, NULL, 0));
        post_graph_created = true;
    }
    checkCudaErrors(cudaGraphLaunch(graph_instance_post, dataBase_.stream));
#endif
    sync();
}
void dfpEqn::postProcess() {}

//void dfpEqn::getFlux()
//{
//    fvMtx_flux(dataBase_.stream, dataBase_.num_surfaces, dataBase_.num_boundary_surfaces, dataBase_.d_owner, dataBase_.d_neighbor, 
//            d_lower, d_upper, dataBase_.d_p, d_flux,
//            dataBase_.num_patches, dataBase_.patch_size.data(), patch_type_p.data(), 
//            dataBase_.d_boundary_face_cell, d_internal_coeffs, d_boundary_coeffs, dataBase_.d_boundary_p, d_boundary_flux);
//    sync();
//}

void dfpEqn::getrhorAUf(cudaStream_t stream, int num_cells, int num_surfaces,
        const int *lowerAddr, const int *upperAddr, 
        const double *weight, const double *vf1, const double *vf2, double *output, // end for internal
        int num_patches, const int *patch_size, const int *patch_type, const double *boundary_weight,
        const double *boundary_vf1, const double *boundary_vf2, double *boundary_output, double sign) 
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_surfaces + threads_per_block - 1) / threads_per_block;
    fvc_interpolate_internal_multi_scalar_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_surfaces,
            lowerAddr, upperAddr, vf1, vf2, weight, output, sign);
    
    int offset = 0;
    for (int i = 0; i < num_patches; i++) {
        if (patch_size[i] == 0) continue;
        threads_per_block = 256;
        blocks_per_grid = (patch_size[i] + threads_per_block - 1) / threads_per_block;
        // TODO: maybe do not need loop boundarys
        if (patch_type[i] == boundaryConditions::zeroGradient
                || patch_type[i] == boundaryConditions::fixedValue
                || patch_type[i] == boundaryConditions::calculated
                || patch_type[i] == boundaryConditions::cyclic) {
            fvc_interpolate_boundary_multi_scalar_kernel_unCouple<<<blocks_per_grid, threads_per_block, 0, stream>>>(patch_size[i], offset,
                    boundary_vf1, boundary_vf2, boundary_output, sign);
        } else if (patch_type[i] == boundaryConditions::processor
                    || patch_type[i] == boundaryConditions::processorCyclic) {
            fvc_interpolate_boundary_multi_scalar_kernel_processor<<<blocks_per_grid, threads_per_block, 0, stream>>>(patch_size[i], offset,
                    boundary_weight, boundary_vf1, boundary_vf2, boundary_output, sign);
            offset += 2 * patch_size[i];
            continue;
        } else {
            fprintf(stderr, "%s %d, boundaryConditions other than zeroGradient are not support yet!\n", __FILE__, __LINE__);
        }
        offset += patch_size[i];
    }
};

void dfpEqn::getphiHbyA(cudaStream_t stream, int num_cells, int num_surfaces, int num_boundary_surfaces, double rDeltaT, 
        const int *lowerAddr, const int *upperAddr, 
        const double *weight, const double *u_old, const double *rho_old, const double *phi_old, const double *rho, 
        const double *rhorAUf, const double *HbyA, const double *Sf, double *output, // end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const double *boundary_Sf, const double *boundary_velocity_old, const double *boundary_rho, 
        const double *boundary_rho_old, const double *boundary_phi_old, const double *boundary_rhorAUf, const double *boundary_HbyA,
        const double *boundary_weight, double *boundary_output, double sign)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_surfaces + threads_per_block - 1) / threads_per_block;
    get_phiCorr_internal_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_surfaces, lowerAddr, upperAddr, 
            phi_old, u_old, rho_old, weight, Sf, output);
    
    int offset = 0;
    for (int i = 0; i < num_patches; i++) {
        if (patch_size[i] == 0) continue;
        threads_per_block = 256;
        blocks_per_grid = (patch_size[i] + threads_per_block - 1) / threads_per_block;
        if (patch_type[i] == boundaryConditions::processor
            || patch_type[i] == boundaryConditions::processorCyclic) {
            get_phiCorr_boundary_kernel_processor<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_boundary_surfaces, patch_size[i], offset,
                    boundary_Sf, boundary_velocity_old, boundary_rho_old, boundary_phi_old, boundary_weight, boundary_output);
            offset += 2 * patch_size[i];
        } else {
            get_phiCorr_boundary_kernel_zeroGradient<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_boundary_surfaces, patch_size[i], offset,
                    boundary_Sf, boundary_velocity_old, boundary_rho_old, boundary_phi_old, boundary_output);
            offset += patch_size[i];
        }
    }

    blocks_per_grid = (num_surfaces + threads_per_block - 1) / threads_per_block;
    get_ddtCorr_internal_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_surfaces, output, phi_old, rDeltaT, output);

    offset = 0;
    for (int i = 0; i < num_patches; i++) {
        if (patch_size[i] == 0) continue;
        threads_per_block = 256;
        blocks_per_grid = (patch_size[i] + threads_per_block - 1) / threads_per_block;
        if (patch_type[i] == boundaryConditions::processor
            || patch_type[i] == boundaryConditions::processorCyclic) {
            get_ddtCorr_boundary_nonZero_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_boundary_surfaces, patch_size[i], offset, 
                    boundary_output, boundary_phi_old, rDeltaT, boundary_output);
            offset += 2 * patch_size[i];
            continue;
        }
        offset += patch_size[i];
    }

    field_multiply_scalar(stream, num_surfaces, output, rhorAUf, output, num_boundary_surfaces, boundary_output, boundary_rhorAUf, boundary_output);

    blocks_per_grid = (num_surfaces + threads_per_block - 1) / threads_per_block;
    multi_fvc_flux_fvc_intepolate_internal_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_surfaces, lowerAddr, upperAddr, 
            HbyA, rho, weight, Sf, output, sign);
    
    offset = 0;
    for (int i = 0; i < num_patches; i++) {
        if (patch_size[i] == 0) continue;
        threads_per_block = 256;
        blocks_per_grid = (patch_size[i] + threads_per_block - 1) / threads_per_block;
        if (patch_type[i] == boundaryConditions::extrapolated
            || patch_type[i] == boundaryConditions::cyclic) {
            multi_fvc_flux_fvc_intepolate_boundary_kernel_zeroGradient<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_boundary_surfaces, patch_size[i], offset, 
                    boundary_Sf, boundary_HbyA, boundary_rho, boundary_output, sign);
        } else if (patch_type[i] == boundaryConditions::processor
                    || patch_type[i] == boundaryConditions::processorCyclic) {
            multi_fvc_flux_fvc_intepolate_boundary_kernel_processor<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_boundary_surfaces, patch_size[i], offset, 
                    boundary_Sf, boundary_HbyA, boundary_weight, boundary_rho, boundary_output, sign);
            offset += 2 * patch_size[i];
            continue;
        } else {
            fprintf(stderr, "%s %d, boundaryConditions other than zeroGradient are not support yet!\n", __FILE__, __LINE__);
        }
        offset += patch_size[i];
    }
}

void dfpEqn::correctionDiagMtxMultiTPsi(cudaStream_t stream, int num_cells, const double *psi, const double *thermo_psi, double *diag, double *source)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    correct_diag_mtx_multi_tpsi_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, psi, thermo_psi, source, diag);
}

void dfpEqn::sync()
{
    checkCudaErrors(cudaStreamSynchronize(dataBase_.stream));
}

void dfpEqn::solve(GAMGStruct *GAMGdata, int agglomeration_level)
{
#ifdef AMGX_
    double amgx_starttime , amgx_endtime;
    amgx_starttime = MPI_Wtime();
    nvtxRangePushA("AMGX::solve()");
    dataBase_.solve(num_iteration, AMGXSetting::p_setting, d_A, dataBase_.d_p, d_source);
    nvtxRangePop();
    amgx_endtime = MPI_Wtime();
    double amgx_totaltime = amgx_endtime - amgx_starttime;
    printf("amgx_totaltime = %.5f\n", amgx_totaltime);
#endif
#ifdef CSR_
    bool useGAMG = true;
    std::cout << "*** call in dfpEqn::solve() for CSR " << std::endl;
    double* d_diag_tmp;
    cudaMallocAsync((void**)&d_diag_tmp, dataBase_.num_cells * sizeof(double), dataBase_.stream);
    cudaMemcpyAsync(d_diag_tmp, d_diag, dataBase_.num_cells * sizeof(double), cudaMemcpyDeviceToDevice, dataBase_.stream);
    // cudaMemcpy(dataBase_.d_csr_row_index_no_diag, GAMGdata[0].d_csr_row_index_no_diag, sizeof(int) * (GAMGdata[0].nCell+1), cudaMemcpyDeviceToDevice);
    // cudaMemcpy(dataBase_.d_csr_col_index_no_diag, GAMGdata[0].d_csr_col_index_no_diag, sizeof(int) * GAMGdata[0].nFace * 2, cudaMemcpyDeviceToDevice);
    if (useGAMG)
    {
        double csr_starttime , csr_endtime;
        csr_starttime = MPI_Wtime();
        nvtxRangePushA("solve_useGAMG()");
        pCSRSolver -> solve_useGAMG(dataBase_, d_internal_coeffs, d_boundary_coeffs, dataBase_.patch_type_extropolated.data(), 
                        d_diag, GAMGdata[0].d_off_diag_value, d_source, dataBase_.d_p,
                        GAMGdata, agglomeration_level);
        nvtxRangePop();
        csr_endtime = MPI_Wtime();
        double csr_totaltime = csr_endtime - csr_starttime;
        printf("csr_totaltime = %.5f\n", csr_totaltime);
    }
    else
    {
        pCSRSolver -> solve(dataBase_, d_internal_coeffs, d_boundary_coeffs, dataBase_.patch_type_extropolated.data(), 
                        d_diag, GAMGdata[0].d_off_diag_value, d_source, dataBase_.d_p);
    }
    printf("peqn PCG solve end\n");
#endif
#ifdef ELL_
    bool useGAMG = true;
    std::cout << "*** call in dfpEqn::solve() for ELL " << std::endl;
    double* d_diag_tmp;
    cudaMallocAsync((void**)&d_diag_tmp, dataBase_.num_cells * sizeof(double), dataBase_.stream);
    cudaMemcpyAsync(d_diag_tmp, d_diag, dataBase_.num_cells * sizeof(double), cudaMemcpyDeviceToDevice, dataBase_.stream);
    if (useGAMG)
    {
        double ell_starttime , ell_endtime;
        ell_starttime = MPI_Wtime();
        nvtxRangePushA("solve_useGAMG()");
        pELLSolver -> solve_useGAMG(dataBase_, d_internal_coeffs, d_boundary_coeffs, dataBase_.patch_type_extropolated.data(), 
                        d_diag, GAMGdata[0].d_ell_values, GAMGdata[0].d_ell_cols, GAMGdata[0].ell_row_maxcount, d_source, dataBase_.d_p,
                        GAMGdata, agglomeration_level);
        nvtxRangePop();
        ell_endtime = MPI_Wtime();
        double ell_totaltime = ell_endtime - ell_starttime;
        printf("ell_totaltime = %.5f\n", ell_totaltime);
    }
    else
    {
        pELLSolver -> solve(dataBase_, d_internal_coeffs, d_boundary_coeffs, dataBase_.patch_type_extropolated.data(), 
                        d_diag, GAMGdata[0].d_ell_values, GAMGdata[0].d_ell_cols, GAMGdata[0].ell_row_maxcount, d_source, dataBase_.d_p);
    }
    printf("peqn PCG solve end\n");
#endif
    num_iteration++;
}

// debug
void dfpEqn::comparerhorAUf(const double *rhorAUf, const double *boundary_rhorAUf, bool printFlag)
{
    double *h_rhorAUf = new double[dataBase_.num_surfaces];
    double *h_boundary_rhorAUf = new double[dataBase_.num_boundary_surfaces];

    checkCudaErrors(cudaMemcpy(h_rhorAUf, d_rhorAUf, dataBase_.surface_value_bytes, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_boundary_rhorAUf, d_boundary_rhorAUf, dataBase_.boundary_surface_value_bytes, cudaMemcpyDeviceToHost));

    fprintf(stderr, "check h_rhorAUf\n");
    checkVectorEqual(dataBase_.num_surfaces, rhorAUf, h_rhorAUf, 1e-10, printFlag);
    fprintf(stderr, "check h_boundary_rhorAUf\n");
    checkVectorEqual(dataBase_.num_boundary_surfaces, boundary_rhorAUf, h_boundary_rhorAUf, 1e-10, printFlag);
}

void dfpEqn::comparephiHbyA(const double *phiHbyA, const double *boundary_phiHbyA, bool printFlag)
{
    double *h_phiHbyA = new double[dataBase_.num_surfaces];
    double *h_boundary_phiHbyA = new double[dataBase_.num_boundary_surfaces];

    checkCudaErrors(cudaMemcpy(h_phiHbyA, d_phiHbyA, dataBase_.surface_value_bytes, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_boundary_phiHbyA, d_boundary_phiHbyA, dataBase_.boundary_surface_value_bytes, cudaMemcpyDeviceToHost));

    fprintf(stderr, "check h_phiHbyA\n");
    checkVectorEqual(dataBase_.num_surfaces, phiHbyA, h_phiHbyA, 1e-10, printFlag);
    fprintf(stderr, "check h_boundary_phiHbyA\n");
    checkVectorEqual(dataBase_.num_boundary_surfaces, boundary_phiHbyA, h_boundary_phiHbyA, 1e-10, printFlag);
}

void dfpEqn::comparephi(const double *phi, const double *boundary_phi, bool printFlag)
{
    double *h_phi = new double[dataBase_.num_surfaces];
    double *h_boundary_phi = new double[dataBase_.num_boundary_surfaces];

    checkCudaErrors(cudaMemcpy(h_phi, dataBase_.d_phi, dataBase_.surface_value_bytes, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_boundary_phi, dataBase_.d_boundary_phi, dataBase_.boundary_surface_value_bytes, cudaMemcpyDeviceToHost));

    fprintf(stderr, "check h_phi\n");
    checkVectorEqual(dataBase_.num_surfaces, phi, h_phi, 1e-10, printFlag);
    fprintf(stderr, "check h_boundary_phi\n");
    checkVectorEqual(dataBase_.num_boundary_surfaces, boundary_phi, h_boundary_phi, 1e-10, printFlag);
}

void dfpEqn::comparephiFlux(const double *flux, const double *boundary_flux, bool printFlag)
{
    double *h_flux = new double[dataBase_.num_surfaces];
    double *h_boundary_flux = new double[dataBase_.num_boundary_surfaces];

    checkCudaErrors(cudaMemcpy(h_flux, d_flux, dataBase_.surface_value_bytes, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_boundary_flux, d_boundary_flux, dataBase_.boundary_surface_value_bytes, cudaMemcpyDeviceToHost));

    fprintf(stderr, "check h_flux\n");
    checkVectorEqual(dataBase_.num_surfaces, flux, h_flux, 1e-10, printFlag);
    fprintf(stderr, "check h_boundary_flux\n");
    checkVectorEqual(dataBase_.num_boundary_surfaces, boundary_flux, h_boundary_flux, 1e-10, printFlag);
}

void dfpEqn::comparep(const double *p, const double *boundary_p, bool printFlag)
{
    double *h_p = new double[dataBase_.num_cells];
    double *h_boundary_p = new double[dataBase_.num_boundary_surfaces];

    checkCudaErrors(cudaMemcpy(h_p, dataBase_.d_p, dataBase_.cell_value_bytes, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_boundary_p, dataBase_.d_boundary_p, dataBase_.boundary_surface_value_bytes, cudaMemcpyDeviceToHost));

    fprintf(stderr, "check h_p\n");
    checkVectorEqual(dataBase_.num_cells, p, h_p, 1e-10, printFlag);
    fprintf(stderr, "check h_boundary_p\n");
    checkVectorEqual(dataBase_.num_boundary_surfaces, boundary_p, h_boundary_p, 1e-10, printFlag);
}

void dfpEqn::compareU(const double *U, const double *boundary_U, bool printFlag)
{
    double *h_u = new double[dataBase_.num_cells * 3];
    double *h_u_ref = new double[dataBase_.num_cells * 3];
    double *h_boundary_u = new double[dataBase_.num_boundary_surfaces * 3];
    double *h_boundary_u_ref = new double[dataBase_.num_boundary_surfaces * 3];

    // permute
    for (int i = 0; i < dataBase_.num_cells; i++)
    {
        h_u_ref[dataBase_.num_cells * 0 + i] = U[i * 3 + 0];
        h_u_ref[dataBase_.num_cells * 1 + i] = U[i * 3 + 1];
        h_u_ref[dataBase_.num_cells * 2 + i] = U[i * 3 + 2];
    }
    for (int i = 0; i < dataBase_.num_boundary_surfaces; i++)
    {
        h_boundary_u_ref[dataBase_.num_boundary_surfaces * 0 + i] = boundary_U[i * 3 + 0];
        h_boundary_u_ref[dataBase_.num_boundary_surfaces * 1 + i] = boundary_U[i * 3 + 1];
        h_boundary_u_ref[dataBase_.num_boundary_surfaces * 2 + i] = boundary_U[i * 3 + 2];
    }
    checkCudaErrors(cudaMemcpy(h_u, dataBase_.d_u, dataBase_.cell_value_vec_bytes, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_boundary_u, dataBase_.d_boundary_u, dataBase_.boundary_surface_value_vec_bytes, cudaMemcpyDeviceToHost));

    // check result
    fprintf(stderr, "check h_u\n");
    checkVectorEqual(dataBase_.num_cells * 3, h_u_ref, h_u, 1e-10, printFlag);
    fprintf(stderr, "check h_boundary_u\n");
    checkVectorEqual(dataBase_.num_boundary_surfaces * 3, h_boundary_u_ref, h_boundary_u, 1e-10, printFlag);
}

void dfpEqn::comparedpdt(const double *dpdt, bool printFlag)
{
    double *h_dpdt = new double[dataBase_.num_cells];
    checkCudaErrors(cudaMemcpy(h_dpdt, dataBase_.d_dpdt, dataBase_.cell_value_bytes, cudaMemcpyDeviceToHost));
    fprintf(stderr, "check h_dpdt\n");
    checkVectorEqual(dataBase_.num_cells, dpdt, h_dpdt, 1e-10, printFlag);
}

void dfpEqn::compareResult(const double *lower, const double *upper, const double *diag, const double *source, const double *internal_coeffs, const double *boundary_coeffs,  
        bool printFlag)
{
    DEBUG_TRACE;
    std::vector<double> h_lower;
    h_lower.resize(dataBase_.num_surfaces);
    checkCudaErrors(cudaMemcpy(h_lower.data(), d_lower, dataBase_.surface_value_bytes, cudaMemcpyDeviceToHost));
    fprintf(stderr, "check h_lower\n");
    checkVectorEqual(dataBase_.num_surfaces, lower, h_lower.data(), 1e-14, printFlag);
    DEBUG_TRACE;

    std::vector<double> h_upper;
    h_upper.resize(dataBase_.num_surfaces);
    checkCudaErrors(cudaMemcpy(h_upper.data(), d_upper, dataBase_.surface_value_bytes, cudaMemcpyDeviceToHost));
    fprintf(stderr, "check h_upper\n");
    checkVectorEqual(dataBase_.num_surfaces, upper, h_upper.data(), 1e-14, printFlag);
    DEBUG_TRACE;

    std::vector<double> h_diag;
    h_diag.resize(dataBase_.num_cells);
    checkCudaErrors(cudaMemcpy(h_diag.data(), d_diag, dataBase_.cell_value_bytes, cudaMemcpyDeviceToHost));
    fprintf(stderr, "check h_diag\n");
    checkVectorEqual(dataBase_.num_cells, diag, h_diag.data(), 1e-14, printFlag);
    DEBUG_TRACE;

    std::vector<double> h_source;
    h_source.resize(dataBase_.num_cells);
    checkCudaErrors(cudaMemcpy(h_source.data(), d_source, dataBase_.cell_value_bytes, cudaMemcpyDeviceToHost));
    fprintf(stderr, "check h_source\n");
    checkVectorEqual(dataBase_.num_cells, source, h_source.data(), 1e-14, printFlag);
    DEBUG_TRACE;

    std::vector<double> h_internal_coeffs;
    h_internal_coeffs.resize(dataBase_.num_boundary_surfaces);
    checkCudaErrors(cudaMemcpy(h_internal_coeffs.data(), d_internal_coeffs, dataBase_.boundary_surface_value_bytes, cudaMemcpyDeviceToHost));
    fprintf(stderr, "check h_internal_coeffs\n");
    checkVectorEqual(dataBase_.num_boundary_surfaces, internal_coeffs, h_internal_coeffs.data(), 1e-14, printFlag);
    DEBUG_TRACE;

    std::vector<double> h_boundary_coeffs;
    h_boundary_coeffs.resize(dataBase_.num_boundary_surfaces);
    checkCudaErrors(cudaMemcpy(h_boundary_coeffs.data(), d_boundary_coeffs, dataBase_.boundary_surface_value_bytes, cudaMemcpyDeviceToHost));
    fprintf(stderr, "check h_boundary_coeffs\n");
    checkVectorEqual(dataBase_.num_boundary_surfaces, boundary_coeffs, h_boundary_coeffs.data(), 1e-14, printFlag);
    DEBUG_TRACE;
}

// cpu version : ldu2csr
// void dfpEqn::peqn_ldu_to_csr_no_diag
// (int row_, int num_surfaces, int *off_diag_rowptr_, double *lower, double *upper, int *lowerAddr, int *upperAddr, double *off_diag_value_)
// {
//     std::vector<int> off_diag_current_index(row_ + 1);
//     off_diag_current_index[0] = 0;
//     for(int i = 0; i < row_; ++i){
//         off_diag_current_index[i + 1] = off_diag_rowptr_[i + 1];
//     }
//     // fill non-zero value
//     for(int i = 0; i < num_surfaces; ++i){
//         int row = upperAddr[i];
//         if(row > row_)printf("overflow\n");
//         int index = off_diag_current_index[row];
//         if(index > num_surfaces * 2)printf("index overflow! index = %d\n",index);
//         off_diag_value_[index] = lower[i]; 
//         off_diag_current_index[row] += 1;
//     }
//     for(int i = 0; i < num_surfaces; ++i){
//         int row = lowerAddr[i];
//         int index = off_diag_current_index[row];
//         off_diag_value_[index] = upper[i];
//         off_diag_current_index[row] += 1;
//     }
// }

__global__ void compute_row_count(cudaStream_t stream, int num_surfaces, int* d_off_diag_count, int* d_lower_count, int* d_upper_count, int* d_lowerAddr, int* d_upperAddr)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_surfaces)
        return;

    atomicAdd(&d_off_diag_count[d_upperAddr[index]], 1);
    atomicAdd(&d_off_diag_count[d_lowerAddr[index]], 1);

    atomicAdd(&d_lower_count[d_upperAddr[index]], 1);
    atomicAdd(&d_upper_count[d_lowerAddr[index]], 1);

}

__global__ void compute_csr_row_index
(
    cudaStream_t stream, int row_, int* d_off_diag_count, int* d_csr_row_index_no_diag, 
    int* d_lower_count, int* d_upper_count, int* d_csr_lower_index, int* d_csr_upper_index
)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= row_)
        return;

    for(int i = 0; i <= index; i++){
        d_csr_row_index_no_diag[index + 1] += d_off_diag_count[i];
        d_csr_lower_index[index + 1] += d_lower_count[i];
        d_csr_upper_index[index + 1] += d_upper_count[i];
    }
}

__global__ void resort_ldu
(
    cudaStream_t stream, int row_, int num_surfaces, int* d_csr_lower_index, int* d_csr_upper_index, int* d_upperAddr, 
    int* d_lowerAddr, double* d_lower, double* d_upper, double* d_resortColValue, int* d_resortColIndex,
    double* d_resortRowValue, int* d_resortRowIndex
)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= row_)
        return;

    // upper
    int offset = d_csr_upper_index[index];
    int count = 0;
    for(int j = 0; j< num_surfaces; j++){
        if(d_lowerAddr[j] == index){
            d_resortColValue[offset + count] = d_upper[j];
            d_resortColIndex[offset + count] = d_upperAddr[j];
            count++;
        }
    }

    // lower
    offset = d_csr_lower_index[index];
    count = 0;
    for(int j = 0; j< num_surfaces; j++){
        if(d_upperAddr[j] == index){
            d_resortRowValue[offset + count] = d_lower[j];
            d_resortRowIndex[offset + count] = d_lowerAddr[j];
            count++;
        }
    }
}

__global__ void compute_csr_col_index_and_value
(
    cudaStream_t stream, int row_, int* d_lower_count, int* d_upper_count,
    int* d_csr_lower_index, int* d_csr_upper_index, double *d_lower, 
    double *d_upper, int *d_lowerAddr, int *d_upperAddr, double *d_off_diag_value, 
    int* d_csr_row_index_no_diag, int *d_csr_col_index_no_diag
)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= row_)
        return;

    int lsize = d_lower_count[i];
    int usize = d_upper_count[i];
    // lower
    for(int j = 0; j < lsize; j++){
        int lduloc = d_csr_lower_index[i] + j;
        int col = d_lowerAddr[lduloc];
        double value = d_lower[lduloc];
        d_csr_col_index_no_diag[d_csr_row_index_no_diag[i] + j] = col;
        d_off_diag_value[d_csr_row_index_no_diag[i] + j] = value;
    }
    // upper
    for(int j = lsize; j < (lsize + usize); j++){
        int lduloc = d_csr_upper_index[i] + j - lsize;
        int col = d_upperAddr[lduloc];
        double value = d_upper[lduloc];
        d_csr_col_index_no_diag[d_csr_row_index_no_diag[i] + j] = col;
        d_off_diag_value[d_csr_row_index_no_diag[i] + j] = value;
    }
}

void dfpEqn::peqn_ldu2csr(
    cudaStream_t stream,
    int row_, 
    int num_surfaces, 
    double *d_lower, 
    double *d_upper, 
    int *d_lowerAddr, 
    int *d_upperAddr, 
    double *d_off_diag_value, 
    int *d_csr_row_index_no_diag, 
    int *d_csr_col_index_no_diag
){

    int* d_off_diag_count;
    int* d_lower_count;
    int* d_upper_count;
    int* d_csr_lower_index;
    int* d_csr_upper_index;
    cudaMalloc(&d_off_diag_count, row_ * sizeof(int));
    cudaMalloc(&d_lower_count, row_ * sizeof(int));
    cudaMalloc(&d_upper_count, row_ * sizeof(int));
    cudaMalloc(&d_csr_lower_index, (row_ + 1) * sizeof(int));
    cudaMalloc(&d_csr_upper_index, (row_ + 1) * sizeof(int));
    cudaMemset(d_off_diag_count, 0, row_ * sizeof(int));
    cudaMemset(d_lower_count, 0, row_ * sizeof(int));
    cudaMemset(d_upper_count, 0, row_ * sizeof(int));
    cudaMemset(d_csr_lower_index, 0, (row_ + 1) * sizeof(int));
    cudaMemset(d_csr_upper_index, 0, (row_ + 1) * sizeof(int));

    int* d_resortColIndex;
    double* d_resortColValue;
    cudaMalloc(&d_resortColIndex, num_surfaces * sizeof(int));
    cudaMalloc(&d_resortColValue, num_surfaces * sizeof(double));
    int* d_resortRowIndex;
    double* d_resortRowValue;
    cudaMalloc(&d_resortRowIndex, num_surfaces * sizeof(int));
    cudaMalloc(&d_resortRowValue, num_surfaces * sizeof(double));

    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_surfaces + threads_per_block - 1) / threads_per_block;
    compute_row_count<<<blocks_per_grid, threads_per_block, 0, stream>>>
        (stream, num_surfaces, d_off_diag_count, d_lower_count, d_upper_count, d_lowerAddr, d_upperAddr);

    threads_per_block = 1024;
    blocks_per_grid = (row_ + threads_per_block - 1) / threads_per_block;
    compute_csr_row_index<<<blocks_per_grid, threads_per_block, 0, stream>>>
        (stream, row_, d_off_diag_count, d_csr_row_index_no_diag, d_lower_count, d_upper_count, d_csr_lower_index, d_csr_upper_index);

    threads_per_block = 1024;
    blocks_per_grid = (row_ + threads_per_block - 1) / threads_per_block;
    resort_ldu<<<blocks_per_grid, threads_per_block, 0, stream>>>
        (stream, row_, num_surfaces, d_csr_lower_index, d_csr_upper_index, d_upperAddr,
        d_lowerAddr, d_lower, d_upper, d_resortColValue, d_resortColIndex, d_resortRowValue, d_resortRowIndex);

    threads_per_block = 1024;
    blocks_per_grid = (row_ + threads_per_block - 1) / threads_per_block;
    compute_csr_col_index_and_value<<<blocks_per_grid, threads_per_block, 0, stream>>>
        (stream, row_, d_lower_count, d_upper_count, d_csr_lower_index, d_csr_upper_index, d_resortRowValue, 
        d_resortColValue, d_resortRowIndex, d_resortColIndex, d_off_diag_value, d_csr_row_index_no_diag, d_csr_col_index_no_diag);

    cudaFree(d_off_diag_count);
    cudaFree(d_lower_count);
    cudaFree(d_upper_count);
    cudaFree(d_csr_lower_index);
    cudaFree(d_csr_upper_index);
    cudaFree(d_resortColIndex);
    cudaFree(d_resortColValue);
    cudaFree(d_resortRowIndex);
    cudaFree(d_resortRowValue);
}

__global__ void kernel_getInterfacesCoeffs
(
    const double* d_boundary_coeffs,
    const double* d_internal_coeffs,
    double* d_interfaceBouCoeffs,
    double* d_interfaceIntCoeffs,
    int interfaceiSize,
    int offset
)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= interfaceiSize)
        return;

    int interface_index = offset + index;
    d_interfaceBouCoeffs[index] = d_boundary_coeffs[interface_index];
    d_interfaceIntCoeffs[index] = d_internal_coeffs[interface_index];
}

void dfpEqn::getInterfacesCoeffs(
    cudaStream_t stream, int num_patches, std::vector<int> patch_size, 
    int *interfaceFlag, int *patch_type, 
    const double *d_boundary_coeffs, const double *d_internal_coeffs, 
    double **d_interfaceBouCoeffs, double **d_interfaceIntCoeffs
){
    int offset = 0;
    for (int i = 0; i < num_patches; i++) {
        if (patch_size[i] == 0) continue;
        else if (interfaceFlag[i] == 0){
            (patch_type[i] == boundaryConditions::processor
                || patch_type[i] == boundaryConditions::processorCyclic) ?
                offset += 2 * patch_size[i] : offset += patch_size[i];
            continue;
        }

        size_t threads_per_block = 1024;
        size_t blocks_per_grid = (patch_size[i] + threads_per_block - 1) / threads_per_block;

        kernel_getInterfacesCoeffs<<<blocks_per_grid, threads_per_block, 0, stream>>>
            (d_boundary_coeffs, d_internal_coeffs, d_interfaceBouCoeffs[i], d_interfaceIntCoeffs[i],
             patch_size[i], offset);

        (patch_type[i] == boundaryConditions::processor
            || patch_type[i] == boundaryConditions::processorCyclic) ?
            offset += 2 * patch_size[i] : offset += patch_size[i];
    }

}


// __global__ void ldu2csr_p(int num_surfaces, int* index, double* value, double* result){
    
//     int i = blockDim.x * blockIdx.x + threadIdx.x;
//     if (i >= num_surfaces)
//         return;

//     result[i] = value[index[i]];
// }
// void dfpEqn::peqn_ldu_to_csr_no_diag
// (int num_surfaces, int* index, double* value, double* result)
// {
//     size_t threads_per_block = 1024;
//     size_t blocks_per_grid = (num_surfaces + threads_per_block - 1) / threads_per_block;

//     ldu2csr_p<<<blocks_per_grid, threads_per_block, 0 ,dataBase_.stream>>>
//         (num_surfaces, index, value, result);
// }
__global__ void ldu2csr_p(int num_surfaces, int* index, double* value1, double* value2, double* result){
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_surfaces)
        return;

    if(index[i] < num_surfaces/2){
        result[i] = value1[index[i]];
    }
    else{
        result[i] = value2[index[i] - num_surfaces/2];
    }

}
void dfpEqn::peqn_ldu_to_csr_no_diag
(int num_surfaces, int* index, double* value1, double* value2, double* result)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_surfaces + threads_per_block - 1) / threads_per_block;

    ldu2csr_p<<<blocks_per_grid, threads_per_block, 0 ,dataBase_.stream>>>
        (num_surfaces, index, value1, value2, result);
}


__global__ void ldu2ell_p(int num_surfaces, int* index, double* value1, double* value2, double* result){
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_surfaces)
        return;

    if(i < num_surfaces/2){
        result[index[i]] = value1[i];
    }
    else{
        result[index[i]] = value2[i - num_surfaces/2];
    }
}
void dfpEqn::peqn_ldu_to_ell_no_diag
(
    int num_surfaces,
    int *ldu2ellIndex,
    double *value1,
    double *value2,
    double *result
)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_surfaces + threads_per_block - 1) / threads_per_block;

    ldu2ell_p<<<blocks_per_grid, threads_per_block, 0 ,dataBase_.stream>>>
        (num_surfaces, ldu2ellIndex, value1, value2, result);
}
