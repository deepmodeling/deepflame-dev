#include "dfUEqn.H"

// #define AMGXSolver_
#define CSRPBiCGSolver_
// #define ELLPBiCGSolver_

__global__ void addAveInternaltoDiagUeqn(int num_cells, int num_boundary_surfaces, const int *face2Cells, 
        const double *internal_coeffs, double *A_pEqn)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_boundary_surfaces)
        return;

    int cellIndex = face2Cells[index];

    double internal_x = internal_coeffs[num_boundary_surfaces * 0 + index];
    double internal_y = internal_coeffs[num_boundary_surfaces * 1 + index];
    double internal_z = internal_coeffs[num_boundary_surfaces * 2 + index];

    double ave_internal = (internal_x + internal_y + internal_z) / 3;

    atomicAdd(&A_pEqn[cellIndex], ave_internal);
}

__global__ void divide_cell_volume_scalar_reverse(int num_cells, const double* volume, double *output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;
    
    double vol = volume[index];

    output[index] = 1/ (output[index] / vol);
}

__global__ void get_calculated_field_boundary(int num_boundary_surfaces, const double* output, 
        const int *face2Cells, double *boundary_output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_boundary_surfaces)
        return;
    
    int cellIndex = face2Cells[index];

    boundary_output[index] = output[cellIndex];
}

__global__ void ueqn_addBoundaryDiag(int num_cells, int num_boundary_surfaces, const int *face2Cells, 
        const double *internal_coeffs, const double *psi, double *H_pEqn)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_boundary_surfaces)
        return;
    
    // addBoundaryDiag(boundaryDiagCmpt, cmpt); // add internal coeffs
    // boundaryDiagCmpt.negate();
    double internal_x = internal_coeffs[num_boundary_surfaces * 0 + index];
    double internal_y = internal_coeffs[num_boundary_surfaces * 1 + index];
    double internal_z = internal_coeffs[num_boundary_surfaces * 2 + index];

    // addCmptAvBoundaryDiag(boundaryDiagCmpt);
    double ave_internal = (internal_x + internal_y + internal_z) / 3;

    int cellIndex = face2Cells[index];

    // do not permute H anymore
    atomicAdd(&H_pEqn[num_cells * 0 + cellIndex], (-internal_x + ave_internal) * psi[num_cells * 0 + cellIndex]);
    atomicAdd(&H_pEqn[num_cells * 1 + cellIndex], (-internal_y + ave_internal) * psi[num_cells * 1 + cellIndex]);
    atomicAdd(&H_pEqn[num_cells * 2 + cellIndex], (-internal_z + ave_internal) * psi[num_cells * 2 + cellIndex]);
}

__global__ void ueqn_lduMatrix_H(int num_cells, int num_surfaces,
        const int *lower_index, const int *upper_index, const double *lower, const double *upper,
        const double *psi, double *H_pEqn)
{
    /*
    for (label face=0; face<nFaces; face++)
    {
        HpsiPtr[uPtr[face]] -= lowerPtr[face]*psiPtr[lPtr[face]];
        HpsiPtr[lPtr[face]] -= upperPtr[face]*psiPtr[uPtr[face]];
    }*/
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_surfaces)
        return;
    
    int l = lower_index[index];
    int u = upper_index[index];

    atomicAdd(&H_pEqn[num_cells * 0 + u], -lower[index] * psi[num_cells * 0 + l]);
    atomicAdd(&H_pEqn[num_cells * 1 + u], -lower[index] * psi[num_cells * 1 + l]);
    atomicAdd(&H_pEqn[num_cells * 2 + u], -lower[index] * psi[num_cells * 2 + l]);
    atomicAdd(&H_pEqn[num_cells * 0 + l], -upper[index] * psi[num_cells * 0 + u]);
    atomicAdd(&H_pEqn[num_cells * 1 + l], -upper[index] * psi[num_cells * 1 + u]);
    atomicAdd(&H_pEqn[num_cells * 2 + l], -upper[index] * psi[num_cells * 2 + u]);

}

__global__ void ueqn_addBoundarySrc_unCoupled(int num_cells, int num, int offset, 
        int num_boundary_surfaces, const int *face2Cells, const double *boundary_coeffs, double *H_pEqn)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num)
        return;

    int start_index = offset + index;

    // addBoundaryDiag(boundaryDiagCmpt, cmpt); // add internal coeffs
    // boundaryDiagCmpt.negate();
    double boundary_x = boundary_coeffs[num_boundary_surfaces * 0 + start_index];
    double boundary_y = boundary_coeffs[num_boundary_surfaces * 1 + start_index];
    double boundary_z = boundary_coeffs[num_boundary_surfaces * 2 + start_index];

    int cellIndex = face2Cells[start_index];

    // do not permute H anymore
    atomicAdd(&H_pEqn[num_cells * 0 + cellIndex], boundary_x);
    atomicAdd(&H_pEqn[num_cells * 1 + cellIndex], boundary_y);
    atomicAdd(&H_pEqn[num_cells * 2 + cellIndex], boundary_z);

}

__global__ void ueqn_addBoundarySrc_processor(int num_cells, int num, int offset, 
        int num_boundary_surfaces, const int *face2Cells, const double *boundary_coeffs, 
        const double *vf_boundary, double *H_pEqn)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num)
        return;

    int neighbor_start_index = offset + index;

    double boundary_x = boundary_coeffs[num_boundary_surfaces * 0 + neighbor_start_index];
    double boundary_y = boundary_coeffs[num_boundary_surfaces * 1 + neighbor_start_index];
    double boundary_z = boundary_coeffs[num_boundary_surfaces * 2 + neighbor_start_index];
    double boundary_vf_x = vf_boundary[num_boundary_surfaces * 0 + neighbor_start_index];
    double boundary_vf_y = vf_boundary[num_boundary_surfaces * 1 + neighbor_start_index];
    double boundary_vf_z = vf_boundary[num_boundary_surfaces * 2 + neighbor_start_index];

    int cellIndex = face2Cells[neighbor_start_index];

    atomicAdd(&H_pEqn[num_cells * 0 + cellIndex], boundary_x * boundary_vf_x);
    atomicAdd(&H_pEqn[num_cells * 1 + cellIndex], boundary_y * boundary_vf_y);
    atomicAdd(&H_pEqn[num_cells * 2 + cellIndex], boundary_z * boundary_vf_z);
}

__global__ void ueqn_addBoundarySrc_cyclic(int num_cells, int num, int internal_offset,
        int neighbor_offset, int num_boundary_surfaces, const int *face2Cells, 
        const double *boundary_coeffs, const double *vf, double *H_pEqn)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num)
        return;

    int internal_start_index = internal_offset + index;
    int neighbor_start_index = neighbor_offset + index;

    int internal_cellIndex = face2Cells[internal_start_index];
    int neighbor_cellIndex = face2Cells[neighbor_start_index];

    double boundary_x = boundary_coeffs[num_boundary_surfaces * 0 + internal_start_index];
    double boundary_y = boundary_coeffs[num_boundary_surfaces * 1 + internal_start_index];
    double boundary_z = boundary_coeffs[num_boundary_surfaces * 2 + internal_start_index];
    double boundary_vf_x = vf[num_cells * 0 + neighbor_cellIndex];
    double boundary_vf_y = vf[num_cells * 1 + neighbor_cellIndex];
    double boundary_vf_z = vf[num_cells * 2 + neighbor_cellIndex];

    atomicAdd(&H_pEqn[num_cells * 0 + internal_cellIndex], boundary_x * boundary_vf_x);
    atomicAdd(&H_pEqn[num_cells * 1 + internal_cellIndex], boundary_y * boundary_vf_y);
    atomicAdd(&H_pEqn[num_cells * 2 + internal_cellIndex], boundary_z * boundary_vf_z);
}

__global__ void divide_vol_multi_rAU(int num_cells, const double *rAU, const double *volume, double *output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;

    // divide volume
    double vol = volume[index];
    double rAU_scalar = rAU[index];

    // multi rAU
    output[num_cells * 0 + index] = output[num_cells * 0 + index] / vol * rAU_scalar;
    output[num_cells * 1 + index] = output[num_cells * 1 + index] / vol * rAU_scalar;
    output[num_cells * 2 + index] = output[num_cells * 2 + index] / vol * rAU_scalar;
}

__global__ void correctBoundary_HbyA_fixedValueU(int num_boundary_surfaces, int num, int offset, 
        const double *boundary_vf, double *boundary_output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num)
        return;

    int start_index = offset + index;

    boundary_output[num_boundary_surfaces * 0 + start_index] = boundary_vf[num_boundary_surfaces * 0 + start_index];
    boundary_output[num_boundary_surfaces * 1 + start_index] = boundary_vf[num_boundary_surfaces * 1 + start_index];
    boundary_output[num_boundary_surfaces * 2 + start_index] = boundary_vf[num_boundary_surfaces * 2 + start_index];
}

__global__ void ueqn_add_external_entry_kernal(int num, int bou_offset, 
        int external_offset, const double *boundary_coeffs, double *external)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num)
        return;

    int bou_start_index = bou_offset + index;
    int external_start_index = external_offset + index;
    external[external_start_index] = - boundary_coeffs[bou_start_index];
}

__global__ void ueqn_add_external_entry_kernal_processCyclic(int num, int bou_offset, 
        int external_offset, const double *boundary_coeffs, double *external)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num)
        return;

    int bou_start_index = bou_offset + index;
    int external_start_index = external_offset + index;
    external[external_start_index] = boundary_coeffs[bou_start_index];
}

__global__ void ueqn_ldu_to_csr_kernel(int nNz, const int *ldu_to_csr_index, 
        const double *ldu, double *A_csr)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= nNz)
        return;
    
    int lduIndex = ldu_to_csr_index[index];
    double csrVal = ldu[lduIndex];
    A_csr[nNz * 0 + index] = csrVal;
    A_csr[nNz * 1 + index] = csrVal;
    A_csr[nNz * 2 + index] = csrVal;
}

__global__ void ueqn_add_boundary_diag_src_unCouple(int num_cells, int num_Nz, int num_boundary_surfaces, 
        int num, int offset, const int *face2Cells, 
        const double *internal_coeffs, const double *boundary_coeffs, const int *diagCSRIndex, 
        double *A_csr, double *b)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num)
        return;
    
    int startIndex = offset + index;
    int cellIndex = face2Cells[startIndex];
    int diagIndex = diagCSRIndex[cellIndex];

    double internalCoeffx = internal_coeffs[num_boundary_surfaces * 0 + startIndex];
    double internalCoeffy = internal_coeffs[num_boundary_surfaces * 1 + startIndex];
    double internalCoeffz = internal_coeffs[num_boundary_surfaces * 2 + startIndex];

    double boundaryCoeffx = boundary_coeffs[num_boundary_surfaces * 0 + startIndex];
    double boundaryCoeffy = boundary_coeffs[num_boundary_surfaces * 1 + startIndex];
    double boundaryCoeffz = boundary_coeffs[num_boundary_surfaces * 2 + startIndex];

    atomicAdd(&A_csr[num_Nz * 0 + diagIndex], internalCoeffx);
    atomicAdd(&A_csr[num_Nz * 1 + diagIndex], internalCoeffy);
    atomicAdd(&A_csr[num_Nz * 2 + diagIndex], internalCoeffz);

    atomicAdd(&b[num_cells * 0 + cellIndex], boundaryCoeffx);
    atomicAdd(&b[num_cells * 1 + cellIndex], boundaryCoeffy);
    atomicAdd(&b[num_cells * 2 + cellIndex], boundaryCoeffz);
}

__global__ void ueqn_add_boundary_diag_src_couple(int num_cells, int num_Nz, int num_boundary_surfaces, 
        int num, int offset, const int *face2Cells, const double *internal_coeffs, 
        const int *diagCSRIndex, double *A_csr)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num)
        return;
    
    int startIndex = offset + index;
    int cellIndex = face2Cells[startIndex];
    int diagIndex = diagCSRIndex[cellIndex];

    double internalCoeffx = internal_coeffs[num_boundary_surfaces * 0 + startIndex];
    double internalCoeffy = internal_coeffs[num_boundary_surfaces * 1 + startIndex];
    double internalCoeffz = internal_coeffs[num_boundary_surfaces * 2 + startIndex];

    atomicAdd(&A_csr[num_Nz * 0 + diagIndex], internalCoeffx);
    atomicAdd(&A_csr[num_Nz * 1 + diagIndex], internalCoeffy);
    atomicAdd(&A_csr[num_Nz * 2 + diagIndex], internalCoeffz);
}

__global__ void ueqn_divide_cell_volume_vec(int num_cells, const double* volume, double *output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;
    
    double vol = volume[index];

    output[num_cells * 0 + index] = output[num_cells * 0 + index] / vol;
    output[num_cells * 1 + index] = output[num_cells * 1 + index] / vol;
    output[num_cells * 2 + index] = output[num_cells * 2 + index] / vol;
}

__global__ void ueqn_calculate_turbulence_k_Smagorinsky(int num_cells, 
        const double *grad_U_tsr, const double *volume, double Ce, double Ck, 
        double *delta, double *output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;
    
    double vol = volume[index];
    double oneThird = (1. / 3.);

    double del = pow(vol, oneThird);

    // D = 0.5*(T+T^T)
    double D_xx = grad_U_tsr[num_cells * 0 + index];
    double D_xy = 0.5 * (grad_U_tsr[num_cells * 1 + index] + grad_U_tsr[num_cells * 3 + index]);
    double D_xz = 0.5 * (grad_U_tsr[num_cells * 2 + index] + grad_U_tsr[num_cells * 6 + index]);
    double D_yy = grad_U_tsr[num_cells * 4 + index];
    double D_yz = 0.5 * (grad_U_tsr[num_cells * 5 + index] + grad_U_tsr[num_cells * 7 + index]);
    double D_zz = grad_U_tsr[num_cells * 8 + index];

    // dev(D)
    double trace = D_xx + D_yy + D_zz;
    double dev_D_xx = D_xx - oneThird * trace;
    double dev_D_yy = D_yy - oneThird * trace;
    double dev_D_zz = D_zz - oneThird * trace;

    // scalar a
    double a = Ce / del;
    // scalar b
    double b = 2 * oneThird * trace;
    // scalar c
    double c = 2 * Ck * del * (dev_D_xx * D_xx + dev_D_yy * D_yy + dev_D_zz * D_zz 
                                    + D_xy * D_xy * 2 + D_xz * D_xz * 2 + D_yz * D_yz * 2);
    
    double sqrt_result = (-b + pow(b * b + 4 * a * c, 0.5)) / (2 * a);
    output[index] = sqrt_result * sqrt_result;
    delta[index] = del;
}

__global__ void ueqn_calculate_turbulence_epsilon_Smagorinsky(int num_cells,
        const double *k, const double *delta, double Ce, double *output)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num_cells)
        return;
    
    output[index] = Ce * pow(k[index], 1.5) / delta[index];
}

void dfUEqn::setConstantValues(const std::string &mode_string, const std::string &setting_path) {
  this->stream = dataBase_.stream;
  this->mode_string = mode_string;
  this->setting_path = setting_path;
  UxSolver = new AmgXSolver(mode_string, setting_path, dataBase_.localRank);
  UySolver = new AmgXSolver(mode_string, setting_path, dataBase_.localRank);
  UzSolver = new AmgXSolver(mode_string, setting_path, dataBase_.localRank);
  
#ifdef CSRPBiCGSolver_
  USolver = new PBiCGStabCSRSolver();
  USolver->initSolvePerformance
    (     
        1e-20, //small_
        2.22507e-308, //vsmall_
        4, //maxIter_ 
        4, //minIter_ 
        1e-9, //tolerance_ 
        0.01 //relTol_
    );
#ifndef STREAM_ALLOCATOR
  USolver->initialize(dataBase_.num_cells, dataBase_.boundary_surface_value_bytes);
#else 
  USolver->initializeStream(dataBase_.num_cells, dataBase_.boundary_surface_value_bytes, dataBase_.stream);
#endif
#endif

#ifdef ELLPBiCGSolver_
  UELLSolver = new PBiCGStabELLSolver();
  UELLSolver->initSolvePerformance
    (     
        1e-20, //small_
        2.22507e-308, //vsmall_
        4, //maxIter_ 
        4, //minIter_ 
        1e-9, //tolerance_ 
        0.01 //relTol_
    );
#ifndef STREAM_ALLOCATOR
  UELLSolver->initialize(dataBase_.num_cells, dataBase_.boundary_surface_value_bytes);
#else
  UELLSolver->initializeStream(dataBase_.num_cells, dataBase_.boundary_surface_value_bytes, dataBase_.stream);
#endif
#endif

}

void dfUEqn::setConstantFields(const std::vector<int> patch_type) {
    this->patch_type = patch_type;

    int offset = 0;
    for (int i = 0; i < dataBase_.num_patches; i++) {
        if (patch_type[i] == boundaryConditions::processor
                || patch_type[i] == boundaryConditions::processorCyclic) {
            dataBase_.patchSizeOffset.push_back(offset);
            offset += dataBase_.patch_size[i] * 2;
            dataBase_.interfaceFlag[i] = 1;
        } else {
            dataBase_.patchSizeOffset.push_back(offset);
            offset += dataBase_.patch_size[i];
        }
    }

}

void dfUEqn::createNonConstantFieldsInternal() {
#ifndef STREAM_ALLOCATOR
  // thermophysical fields
  checkCudaErrors(cudaMalloc((void**)&d_nu_eff, dataBase_.cell_value_bytes));
  // intermediate fields
  checkCudaErrors(cudaMalloc((void**)&d_grad_u, dataBase_.cell_value_tsr_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_delta, dataBase_.cell_value_bytes));

  checkCudaErrors(cudaMalloc((void**)&d_rho_nueff, dataBase_.cell_value_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_fvc_output, dataBase_.cell_value_vec_bytes));
#endif
  checkCudaErrors(cudaMalloc((void**)&d_u_host_order, dataBase_.cell_value_vec_bytes));
  // computed on CPU, used on GPU, need memcpyh2d
  checkCudaErrors(cudaMallocHost((void**)&h_nu_eff , dataBase_.cell_value_bytes));
  checkCudaErrors(cudaMallocHost((void**)&h_A_pEqn , dataBase_.cell_value_bytes));
  checkCudaErrors(cudaMallocHost((void**)&h_H_pEqn , dataBase_.cell_value_vec_bytes));

  // getter for h_nu_eff
  fieldPointerMap["h_nu_eff"] = h_nu_eff;
}
        
void dfUEqn::createNonConstantFieldsBoundary() {
#ifndef STREAM_ALLOCATOR
  // thermophysical fields
  checkCudaErrors(cudaMalloc((void**)&d_boundary_nu_eff, dataBase_.boundary_surface_value_bytes));
  // intermediate fields
  checkCudaErrors(cudaMalloc((void**)&d_boundary_grad_u, dataBase_.boundary_surface_value_tsr_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_boundary_rho_nueff, dataBase_.boundary_surface_value_bytes));
  // boundary coeff fields
  checkCudaErrors(cudaMalloc((void**)&d_value_internal_coeffs, dataBase_.boundary_surface_value_vec_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_value_boundary_coeffs, dataBase_.boundary_surface_value_vec_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_gradient_internal_coeffs, dataBase_.boundary_surface_value_vec_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_gradient_boundary_coeffs, dataBase_.boundary_surface_value_vec_bytes));
#endif
  // intermediate boundary
  checkCudaErrors(cudaMalloc((void**)&d_boundary_u_host_order, dataBase_.boundary_surface_value_vec_bytes));
  // computed on CPU, used on GPU, need memcpyh2d
  checkCudaErrors(cudaMallocHost((void**)&h_boundary_nu_eff, dataBase_.boundary_surface_value_bytes));

  // getter for h_boundary_nu_eff
  fieldPointerMap["h_boundary_nu_eff"] = h_boundary_nu_eff;
}

void dfUEqn::createNonConstantLduAndCsrFields() {
  checkCudaErrors(cudaMalloc((void**)&d_ldu, dataBase_.csr_value_bytes));
#ifdef AMGX_
  d_lower = d_ldu;
  d_diag = d_ldu + dataBase_.num_surfaces;
  d_upper = d_ldu + dataBase_.num_cells + dataBase_.num_surfaces;
#endif
#ifdef CSRPBiCGSolver_
  d_lower = d_ldu;
  d_upper = d_ldu + dataBase_.num_surfaces;
  d_diag = d_ldu + 2 * dataBase_.num_surfaces;
#endif
#ifdef ELLPBiCGSolver_
// TODO
  d_lower = d_ldu;
  d_diag = d_ldu + dataBase_.num_surfaces;
  d_upper = d_ldu + dataBase_.num_cells + dataBase_.num_surfaces;
#endif
  d_extern = d_ldu + dataBase_.num_cells + 2 * dataBase_.num_surfaces;
  checkCudaErrors(cudaMalloc((void**)&d_source, dataBase_.cell_value_vec_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_internal_coeffs, dataBase_.boundary_surface_value_vec_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_boundary_coeffs, dataBase_.boundary_surface_value_vec_bytes));
#ifndef STREAM_ALLOCATOR
  checkCudaErrors(cudaMalloc((void**)&d_A, dataBase_.csr_value_vec_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_b, dataBase_.cell_value_vec_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_ldu_solve, dataBase_.csr_value_bytes));
  d_extern_solve = d_ldu_solve + dataBase_.num_cells + 2 * dataBase_.num_surfaces;
  checkCudaErrors(cudaMalloc((void**)&d_source_solve, dataBase_.cell_value_vec_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_internal_coeffs_solve, dataBase_.boundary_surface_value_vec_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_boundary_coeffs_solve, dataBase_.boundary_surface_value_vec_bytes));
#endif
  checkCudaErrors(cudaMalloc((void**)&d_A_pEqn, dataBase_.cell_value_bytes)); // TODO: delete redundant variables
  checkCudaErrors(cudaMalloc((void**)&d_H_pEqn, dataBase_.cell_value_vec_bytes));
  checkCudaErrors(cudaMalloc((void**)&d_H_pEqn_perm, dataBase_.cell_value_vec_bytes));
}

void dfUEqn::initNonConstantFieldsInternal(const double *u, const double *boundary_u)
{
    checkCudaErrors(cudaMemcpyAsync(d_u_host_order, u, dataBase_.cell_value_vec_bytes, cudaMemcpyHostToDevice, dataBase_.stream));
    checkCudaErrors(cudaMemcpyAsync(d_boundary_u_host_order, boundary_u, dataBase_.boundary_surface_value_vec_bytes, cudaMemcpyHostToDevice, dataBase_.stream));
    permute_vector_h2d(dataBase_.stream, dataBase_.num_cells, d_u_host_order, dataBase_.d_u);
    permute_vector_h2d(dataBase_.stream, dataBase_.num_boundary_surfaces, d_boundary_u_host_order, dataBase_.d_boundary_u);
}

void dfUEqn::initNonConstantFieldsBoundary() {
    // update_boundary_coeffs_vector(dataBase_.stream, dataBase_.num_boundary_surfaces, dataBase_.num_patches,
    //        dataBase_.patch_size.data(), patch_type.data(), dataBase_.d_boundary_u, dataBase_.d_boundary_delta_coeffs,
    //        d_value_internal_coeffs, d_value_boundary_coeffs,
    //        d_gradient_internal_coeffs, 
    // );
}

void dfUEqn::cleanCudaResources() {
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

void dfUEqn::preProcessForRhoEqn(const double *h_rho, const double *h_phi, const double *h_boundary_phi) {
  checkCudaErrors(cudaMemcpyAsync(dataBase_.d_rho, h_rho, dataBase_.cell_value_bytes, cudaMemcpyHostToDevice, dataBase_.stream));
  checkCudaErrors(cudaMemcpyAsync(dataBase_.d_phi, h_phi, dataBase_.surface_value_bytes, cudaMemcpyHostToDevice, dataBase_.stream));
  checkCudaErrors(cudaMemcpyAsync(dataBase_.d_boundary_phi, h_boundary_phi, dataBase_.boundary_surface_value_bytes, cudaMemcpyHostToDevice, dataBase_.stream));
}

void dfUEqn::preProcess(const double *h_u, const double *h_boundary_u, const double *h_p, const double *h_boundary_p, 
        const double *h_nu_eff, const double *h_boundary_nu_eff) {

}

void dfUEqn::process() {
    TICK_INIT_EVENT;
    TICK_START_EVENT;
#ifdef USE_GRAPH
    if(!pre_graph_created) {
        DEBUG_TRACE;
        checkCudaErrors(cudaStreamBeginCapture(dataBase_.stream, cudaStreamCaptureModeGlobal));
#endif

#ifdef STREAM_ALLOCATOR
        // thermophysical fields
        checkCudaErrors(cudaMallocAsync((void**)&d_nu_eff, dataBase_.cell_value_bytes, dataBase_.stream));
        // intermediate fields
        checkCudaErrors(cudaMallocAsync((void**)&d_grad_u, dataBase_.cell_value_tsr_bytes, dataBase_.stream));
        checkCudaErrors(cudaMallocAsync((void**)&d_delta, dataBase_.cell_value_bytes, dataBase_.stream));

        checkCudaErrors(cudaMallocAsync((void**)&d_rho_nueff, dataBase_.cell_value_bytes, dataBase_.stream));
        checkCudaErrors(cudaMallocAsync((void**)&d_fvc_output, dataBase_.cell_value_vec_bytes, dataBase_.stream));

        // thermophysical fields
        checkCudaErrors(cudaMallocAsync((void**)&d_boundary_nu_eff, dataBase_.boundary_surface_value_bytes, dataBase_.stream));
        // intermediate fields
        checkCudaErrors(cudaMallocAsync((void**)&d_boundary_grad_u, dataBase_.boundary_surface_value_tsr_bytes, dataBase_.stream));
        checkCudaErrors(cudaMallocAsync((void**)&d_boundary_rho_nueff, dataBase_.boundary_surface_value_bytes, dataBase_.stream));
        // boundary coeff fields
        checkCudaErrors(cudaMallocAsync((void**)&d_value_internal_coeffs, dataBase_.boundary_surface_value_vec_bytes, dataBase_.stream));
        checkCudaErrors(cudaMallocAsync((void**)&d_value_boundary_coeffs, dataBase_.boundary_surface_value_vec_bytes, dataBase_.stream));
        checkCudaErrors(cudaMallocAsync((void**)&d_gradient_internal_coeffs, dataBase_.boundary_surface_value_vec_bytes, dataBase_.stream));
        checkCudaErrors(cudaMallocAsync((void**)&d_gradient_boundary_coeffs, dataBase_.boundary_surface_value_vec_bytes, dataBase_.stream));

        checkCudaErrors(cudaMallocAsync((void**)&d_A, dataBase_.csr_value_vec_bytes, dataBase_.stream));
        checkCudaErrors(cudaMallocAsync((void**)&d_b, dataBase_.cell_value_vec_bytes, dataBase_.stream));
        checkCudaErrors(cudaMallocAsync((void**)&d_ldu_solve, dataBase_.csr_value_bytes, dataBase_.stream));
        d_extern_solve = d_ldu_solve + dataBase_.num_cells + 2 * dataBase_.num_surfaces;
        checkCudaErrors(cudaMallocAsync((void**)&d_source_solve, dataBase_.cell_value_vec_bytes, dataBase_.stream));
        checkCudaErrors(cudaMallocAsync((void**)&d_internal_coeffs_solve, dataBase_.boundary_surface_value_vec_bytes, dataBase_.stream));
        checkCudaErrors(cudaMallocAsync((void**)&d_boundary_coeffs_solve, dataBase_.boundary_surface_value_vec_bytes, dataBase_.stream));

#endif

        // checkCudaErrors(cudaMemcpyAsync(d_u_host_order, dataBase_.h_u, dataBase_.cell_value_vec_bytes, cudaMemcpyHostToDevice, dataBase_.stream));
        // checkCudaErrors(cudaMemcpyAsync(d_boundary_u_host_order, dataBase_.h_boundary_u, dataBase_.boundary_surface_value_vec_bytes, cudaMemcpyHostToDevice, dataBase_.stream));
        // checkCudaErrors(cudaMemcpyAsync(dataBase_.d_p, dataBase_.h_p, dataBase_.cell_value_bytes, cudaMemcpyHostToDevice, dataBase_.stream));
        // checkCudaErrors(cudaMemcpyAsync(dataBase_.d_boundary_p, dataBase_.h_boundary_p, dataBase_.boundary_surface_value_bytes, cudaMemcpyHostToDevice, dataBase_.stream));
        // checkCudaErrors(cudaMemcpyAsync(d_nu_eff, h_nu_eff, dataBase_.cell_value_bytes, cudaMemcpyHostToDevice, dataBase_.stream));
        // checkCudaErrors(cudaMemcpyAsync(d_boundary_nu_eff, h_boundary_nu_eff, dataBase_.boundary_surface_value_bytes, cudaMemcpyHostToDevice, dataBase_.stream));

        checkCudaErrors(cudaMemsetAsync(d_ldu, 0, dataBase_.csr_value_bytes, dataBase_.stream)); // d_ldu contains d_lower, d_diag, and d_upper
        checkCudaErrors(cudaMemsetAsync(d_source, 0, dataBase_.cell_value_vec_bytes, dataBase_.stream));
        checkCudaErrors(cudaMemsetAsync(d_internal_coeffs, 0, dataBase_.boundary_surface_value_vec_bytes, dataBase_.stream));
        checkCudaErrors(cudaMemsetAsync(d_boundary_coeffs, 0, dataBase_.boundary_surface_value_vec_bytes, dataBase_.stream));
        checkCudaErrors(cudaMemsetAsync(d_A, 0, dataBase_.csr_value_vec_bytes, dataBase_.stream));
        checkCudaErrors(cudaMemsetAsync(d_b, 0, dataBase_.cell_value_vec_bytes, dataBase_.stream));
        checkCudaErrors(cudaMemsetAsync(d_A_pEqn, 0, dataBase_.cell_value_bytes, dataBase_.stream));
        checkCudaErrors(cudaMemsetAsync(d_H_pEqn, 0, dataBase_.cell_value_vec_bytes, dataBase_.stream));

        checkCudaErrors(cudaMemsetAsync(d_grad_u, 0, dataBase_.cell_value_tsr_bytes, dataBase_.stream));
        checkCudaErrors(cudaMemsetAsync(d_boundary_grad_u, 0, dataBase_.boundary_surface_value_tsr_bytes, dataBase_.stream));

        checkCudaErrors(cudaMemsetAsync(d_delta, 0, dataBase_.cell_value_bytes, dataBase_.stream));

        // permute_vector_h2d(dataBase_.stream, dataBase_.num_cells, d_u_host_order, dataBase_.d_u);
        // permute_vector_h2d(dataBase_.stream, dataBase_.num_boundary_surfaces, d_boundary_u_host_order, dataBase_.d_boundary_u);
        update_boundary_coeffs_vector(dataBase_.stream, dataBase_.num_boundary_surfaces, dataBase_.num_patches,
            dataBase_.patch_size.data(), patch_type.data(), dataBase_.d_boundary_u, dataBase_.d_boundary_delta_coeffs, dataBase_.d_boundary_weight,
            d_value_internal_coeffs, d_value_boundary_coeffs,
            d_gradient_internal_coeffs, d_gradient_boundary_coeffs);
        fvm_ddt_vector(dataBase_.stream, dataBase_.num_cells, dataBase_.rdelta_t,
                dataBase_.d_rho, dataBase_.d_rho_old, dataBase_.d_u, dataBase_.d_volume,
                d_diag, d_source, 1.);
        fvm_div_vector(dataBase_.stream, dataBase_.num_surfaces, dataBase_.num_boundary_surfaces, 
                dataBase_.d_owner, dataBase_.d_neighbor,
                dataBase_.d_phi, dataBase_.d_weight,
                d_lower, d_upper, d_diag, // end for internal
                dataBase_.num_patches, dataBase_.patch_size.data(), patch_type.data(),
                dataBase_.d_boundary_phi, d_value_internal_coeffs, d_value_boundary_coeffs,
                d_internal_coeffs, d_boundary_coeffs, 1.);
        // field_multiply_scalar(dataBase_.stream,
        //        dataBase_.num_cells, dataBase_.d_rho, d_nu_eff, d_rho_nueff, // end for internal
        //        dataBase_.num_boundary_surfaces, dataBase_.d_boundary_rho, d_boundary_nu_eff, d_boundary_rho_nueff);
        fvm_laplacian_vector(dataBase_.stream, dataBase_.num_surfaces, dataBase_.num_boundary_surfaces, 
               dataBase_.d_owner, dataBase_.d_neighbor,
               dataBase_.d_weight, dataBase_.d_mag_sf, dataBase_.d_delta_coeffs, dataBase_.d_mu,
               d_lower, d_upper, d_diag, // end for internal
               dataBase_.num_patches, dataBase_.patch_size.data(), patch_type.data(),
               dataBase_.d_boundary_mag_sf, dataBase_.d_boundary_mu,
               d_gradient_internal_coeffs, d_gradient_boundary_coeffs,
               d_internal_coeffs, d_boundary_coeffs, -1);
        fvc_grad_vector(dataBase_.stream, dataBase_.nccl_comm,
                dataBase_.num_cells, dataBase_.num_surfaces, dataBase_.num_boundary_surfaces,
                dataBase_.neighbProcNo.data(), dataBase_.d_owner, dataBase_.d_neighbor,
                dataBase_.d_weight, dataBase_.d_sf, dataBase_.d_u, d_grad_u,
                dataBase_.num_patches, dataBase_.patch_size.data(), patch_type.data(),
                dataBase_.d_boundary_face_cell, dataBase_.d_boundary_u, dataBase_.d_boundary_sf, dataBase_.d_boundary_weight, 
                dataBase_.d_volume, dataBase_.d_boundary_mag_sf, d_boundary_grad_u, dataBase_.cyclicNeighbor.data(),
                dataBase_.patchSizeOffset.data(), dataBase_.d_boundary_delta_coeffs);

        // **if use turbulence model**
        // calculate k & epsilon
        getTurbulenceKEpsilon_Smagorinsky(dataBase_.stream, dataBase_.num_cells, dataBase_.num_boundary_surfaces, d_grad_u, dataBase_.d_volume, 
                d_delta, dataBase_.d_turbulence_k, dataBase_.d_turbulence_epsilon);
        // calculate nut
        // **end use turbulence model**

        scale_dev2T_tensor(dataBase_.stream, dataBase_.num_cells, dataBase_.d_mu, d_grad_u, // end for internal
                dataBase_.num_boundary_surfaces, dataBase_.d_boundary_mu, d_boundary_grad_u);
        fvc_div_cell_tensor(dataBase_.stream, dataBase_.num_cells, dataBase_.num_surfaces, dataBase_.num_boundary_surfaces,
                dataBase_.d_owner, dataBase_.d_neighbor,
                dataBase_.d_weight, dataBase_.d_sf, d_grad_u, d_source, // end for internal
                dataBase_.num_patches, dataBase_.patch_size.data(), dataBase_.patch_type_calculated.data(), dataBase_.d_boundary_weight,
                dataBase_.d_boundary_face_cell, d_boundary_grad_u, dataBase_.d_boundary_sf, dataBase_.d_volume);
        
        checkCudaErrors(cudaMemcpyAsync(d_ldu_solve, d_ldu, dataBase_.csr_value_bytes, cudaMemcpyDeviceToDevice, dataBase_.stream));
        checkCudaErrors(cudaMemcpyAsync(d_source_solve, d_source, dataBase_.cell_value_vec_bytes, cudaMemcpyDeviceToDevice, dataBase_.stream));
        checkCudaErrors(cudaMemcpyAsync(d_internal_coeffs_solve, d_internal_coeffs, dataBase_.boundary_surface_value_vec_bytes, cudaMemcpyDeviceToDevice, dataBase_.stream));
        checkCudaErrors(cudaMemcpyAsync(d_boundary_coeffs_solve, d_boundary_coeffs, dataBase_.boundary_surface_value_vec_bytes, cudaMemcpyDeviceToDevice, dataBase_.stream));

        fvc_grad_cell_scalar(dataBase_.stream, dataBase_.num_cells, dataBase_.num_surfaces, dataBase_.num_boundary_surfaces,
                dataBase_.d_owner, dataBase_.d_neighbor,
                dataBase_.d_weight, dataBase_.d_sf, dataBase_.d_p, d_source_solve,
                dataBase_.num_patches, dataBase_.patch_size.data(), patch_type.data(), dataBase_.d_boundary_weight,
                dataBase_.d_boundary_face_cell, dataBase_.d_boundary_p, dataBase_.d_boundary_sf, dataBase_.d_volume, -1.);
        getrAU(dataBase_.stream, dataBase_.nccl_comm, dataBase_.num_cells, dataBase_.num_surfaces, dataBase_.num_boundary_surfaces, 
                dataBase_.neighbProcNo.data(), dataBase_.num_patches, dataBase_.patch_size.data(), dataBase_.patch_type_extropolated.data(),
                dataBase_.d_boundary_face_cell, dataBase_.d_boundary_delta_coeffs, d_internal_coeffs, dataBase_.d_volume, d_diag, 
                dataBase_.d_rAU, dataBase_.d_boundary_rAU);
#ifndef DEBUG_CHECK_LDU   

#ifdef AMGXSolver_
        ueqn_ldu_to_csr(dataBase_.stream, dataBase_.num_cells, dataBase_.num_surfaces, dataBase_.num_boundary_surfaces, dataBase_.num_Nz,
            dataBase_.d_boundary_face_cell, dataBase_.d_ldu_to_csr_index, dataBase_.d_diag_to_csr_index, 
            dataBase_.num_patches, dataBase_.patch_size.data(), patch_type.data(), dataBase_.d_u, dataBase_.d_boundary_u,
            d_ldu_solve, d_extern_solve, d_source_solve, d_internal_coeffs_solve, d_boundary_coeffs_solve, dataBase_.cyclicNeighbor.data(), 
            dataBase_.patchSizeOffset.data(), d_A, d_b);
#endif
#ifdef CSRPBiCGSolver_
        checkCudaErrors(cudaMallocAsync((void**)&d_off_diag, dataBase_.num_surfaces * 2 * sizeof(double), dataBase_.stream));        
        ueqn_ldu_to_csr_no_diag(dataBase_.num_surfaces * 2, dataBase_.d_ldu_to_csr_no_diag, d_ldu, d_off_diag);
#endif
#ifdef ELLPBiCGSolver_
// TODO
        double *h_lower;
        double *h_upper;
        int *h_owner;
        int *h_neighbor;
        h_lower = (double*)malloc(dataBase_.num_surfaces * sizeof(double));
        h_upper = (double*)malloc(dataBase_.num_surfaces * sizeof(double));
        h_owner = (int*)malloc(dataBase_.num_surfaces * sizeof(int));
        h_neighbor = (int*)malloc(dataBase_.num_surfaces * sizeof(int));
        cudaMemcpyAsync(h_lower, d_lower, dataBase_.num_surfaces * sizeof(double), cudaMemcpyDeviceToHost, dataBase_.stream);
        cudaMemcpyAsync(h_upper, d_upper, dataBase_.num_surfaces * sizeof(double), cudaMemcpyDeviceToHost, dataBase_.stream);
        cudaMemcpyAsync(h_owner, dataBase_.d_owner, dataBase_.num_surfaces * sizeof(int), cudaMemcpyDeviceToHost, dataBase_.stream);
        cudaMemcpyAsync(h_neighbor, dataBase_.d_neighbor, dataBase_.num_surfaces * sizeof(int), cudaMemcpyDeviceToHost, dataBase_.stream);
        
        maxcount = get_ell_row_maxcount(dataBase_.num_cells, dataBase_.num_surfaces, h_owner, h_neighbor);
        double* ellValues;
        int* ellCols;
        ellValues = (double*)malloc(sizeof(double) * dataBase_.num_cells * maxcount);
        ellCols = (int*)malloc(sizeof(int) * dataBase_.num_cells * maxcount);
        for(int i = 0; i < dataBase_.num_cells * maxcount; i++){
            ellValues[i] = 0;
            ellCols[i] = -1;
        }
        ueqn_ldu_to_ell_no_diag(dataBase_.num_cells, dataBase_.num_surfaces, h_lower, h_upper, h_owner, h_neighbor, ellValues, ellCols, maxcount);
        cudaMallocAsync((void**)&d_ellValues, dataBase_.num_cells * maxcount * sizeof(double), dataBase_.stream);
        cudaMallocAsync((void**)&d_ellCols, dataBase_.num_cells * maxcount * sizeof(int), dataBase_.stream);
        cudaMemcpyAsync(d_ellValues, ellValues, dataBase_.num_cells * maxcount * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_ellCols, ellCols, dataBase_.num_cells * maxcount * sizeof(int), cudaMemcpyHostToDevice);
#endif
#endif
#ifdef USE_GRAPH
        checkCudaErrors(cudaStreamEndCapture(dataBase_.stream, &graph_pre));
        checkCudaErrors(cudaGraphInstantiate(&graph_instance_pre, graph_pre, NULL, NULL, 0));
        pre_graph_created = true;
    }
    DEBUG_TRACE;
    checkCudaErrors(cudaGraphLaunch(graph_instance_pre, dataBase_.stream));
#endif
    TICK_END_EVENT(UEqn assembly);

    TICK_START_EVENT;
#ifndef DEBUG_CHECK_LDU
    solve();
#endif
    TICK_END_EVENT(UEqn solve);

#ifdef USE_GRAPH
    if(!post_graph_created) {
        checkCudaErrors(cudaStreamBeginCapture(dataBase_.stream, cudaStreamCaptureModeGlobal));
#endif

        TICK_START_EVENT;
        correct_boundary_conditions_vector(dataBase_.stream, dataBase_.nccl_comm, dataBase_.neighbProcNo.data(), dataBase_.num_boundary_surfaces, 
                dataBase_.num_cells, dataBase_.num_patches, dataBase_.patch_size.data(), patch_type.data(), dataBase_.d_boundary_weight,
                dataBase_.d_boundary_face_cell, dataBase_.d_u, dataBase_.d_boundary_u, 
                dataBase_.cyclicNeighbor.data(), dataBase_.patchSizeOffset.data());
        vector_half_mag_square(dataBase_.stream, dataBase_.num_cells, dataBase_.d_u, dataBase_.d_k, dataBase_.num_boundary_surfaces, 
                dataBase_.d_boundary_u, dataBase_.d_boundary_k);
        TICK_END_EVENT(UEqn post process correctBC);

        TICK_START_EVENT;
#ifdef STREAM_ALLOCATOR
        // free
        // thermophysical fields
        checkCudaErrors(cudaFreeAsync(d_nu_eff, dataBase_.stream));
        // intermediate fields
        checkCudaErrors(cudaFreeAsync(d_grad_u, dataBase_.stream));
        checkCudaErrors(cudaFreeAsync(d_delta, dataBase_.stream));
        checkCudaErrors(cudaFreeAsync(d_rho_nueff, dataBase_.stream));
        checkCudaErrors(cudaFreeAsync(d_fvc_output, dataBase_.stream));
    
        // thermophysical fields
        checkCudaErrors(cudaFreeAsync(d_boundary_nu_eff, dataBase_.stream));
        // intermediate fields
        checkCudaErrors(cudaFreeAsync(d_boundary_grad_u, dataBase_.stream));
        checkCudaErrors(cudaFreeAsync(d_boundary_rho_nueff, dataBase_.stream));
        // boundary coeff fields
        checkCudaErrors(cudaFreeAsync(d_value_internal_coeffs, dataBase_.stream));
        checkCudaErrors(cudaFreeAsync(d_value_boundary_coeffs, dataBase_.stream));
        checkCudaErrors(cudaFreeAsync(d_gradient_internal_coeffs, dataBase_.stream));
        checkCudaErrors(cudaFreeAsync(d_gradient_boundary_coeffs, dataBase_.stream));
    
        checkCudaErrors(cudaFreeAsync(d_A, dataBase_.stream));
        checkCudaErrors(cudaFreeAsync(d_b, dataBase_.stream));
        checkCudaErrors(cudaFreeAsync(d_ldu_solve, dataBase_.stream));
        checkCudaErrors(cudaFreeAsync(d_source_solve, dataBase_.stream));
        checkCudaErrors(cudaFreeAsync(d_internal_coeffs_solve, dataBase_.stream));
        checkCudaErrors(cudaFreeAsync(d_boundary_coeffs_solve, dataBase_.stream));
#endif
        TICK_END_EVENT(UEqn post process free);

#ifdef CSRPBiCGSolver_
        USolver->freeInit();

#ifdef USE_GRAPH
        checkCudaErrors(cudaStreamEndCapture(dataBase_.stream, &graph_post));
        checkCudaErrors(cudaGraphInstantiate(&graph_instance_post, graph_post, NULL, NULL, 0));
        post_graph_created = true;
    }
    checkCudaErrors(cudaGraphLaunch(graph_instance_post, dataBase_.stream));
#endif
    sync();
}

void dfUEqn::sync()
{
    checkCudaErrors(cudaStreamSynchronize(dataBase_.stream));
}

void dfUEqn::solve() {

#ifdef AMGXSolver_
    TICK_INIT_EVENT;
    TICK_START_EVENT;
    // nvtxRangePushA("AMGX solve x");
    dataBase_.solve(num_iteration, AMGXSetting::u_setting, d_A, dataBase_.d_u, d_b);
    // nvtxRangePop();
    dataBase_.solve(num_iteration, AMGXSetting::u_setting, d_A + dataBase_.num_Nz, dataBase_.d_u + dataBase_.num_cells, d_b + dataBase_.num_cells);
    dataBase_.solve(num_iteration, AMGXSetting::u_setting, d_A + 2 * dataBase_.num_Nz, dataBase_.d_u + 2 * dataBase_.num_cells, d_b + 2 * dataBase_.num_cells);
    TICK_END_EVENT(AMGX solve);
#endif
#ifdef CSRPBiCGSolver_
    double* d_diag_tmp;
    cudaMallocAsync((void**)&d_diag_tmp, dataBase_.num_cells * sizeof(double), dataBase_.stream);
    cudaMemcpyAsync(d_diag_tmp, d_diag, dataBase_.num_cells * sizeof(double), cudaMemcpyDeviceToDevice, dataBase_.stream);
    TICK_INIT_EVENT;
    TICK_START_EVENT;
    // nvtxRangePushA("CSR PBiCGStab solve x");
    USolver->solve(dataBase_, d_internal_coeffs_solve, d_boundary_coeffs_solve, &patch_type[0], d_diag_tmp, d_off_diag, d_source_solve, dataBase_.d_u);
    // nvtxRangePop();
    cudaMemcpyAsync(d_diag_tmp, d_diag, dataBase_.num_cells * sizeof(double), cudaMemcpyDeviceToDevice, dataBase_.stream);
    USolver->solve(dataBase_, d_internal_coeffs_solve + dataBase_.num_boundary_surfaces, d_boundary_coeffs_solve + dataBase_.num_boundary_surfaces, &patch_type[0], d_diag_tmp, d_off_diag, d_source_solve + dataBase_.num_cells, dataBase_.d_u + dataBase_.num_cells);
    cudaMemcpyAsync(d_diag_tmp, d_diag, dataBase_.num_cells * sizeof(double), cudaMemcpyDeviceToDevice, dataBase_.stream);
    USolver->solve(dataBase_, d_internal_coeffs_solve + 2 * dataBase_.num_boundary_surfaces, d_boundary_coeffs_solve + 2 * dataBase_.num_boundary_surfaces, &patch_type[0], d_diag_tmp, d_off_diag, d_source_solve + 2 * dataBase_.num_cells, dataBase_.d_u + 2 * dataBase_.num_cells);
    TICK_END_EVENT(CSR PBiCGStab solve);
#endif
#ifdef ELLPBiCGSolver_
    double* d_diag_tmp;
    cudaMallocAsync((void**)&d_diag_tmp, dataBase_.num_cells * sizeof(double), dataBase_.stream);
    cudaMemcpyAsync(d_diag_tmp, d_diag, dataBase_.num_cells * sizeof(double), cudaMemcpyDeviceToDevice, dataBase_.stream);
    TICK_INIT_EVENT;
    TICK_START_EVENT;
    // nvtxRangePushA("ELL PBiCGStab solve x");
    UELLSolver->solve(dataBase_, d_internal_coeffs_solve, d_boundary_coeffs_solve, &patch_type[0], d_diag_tmp, d_ellValues, d_ellCols, maxcount, d_source_solve, dataBase_.d_u);
    // nvtxRangePop();
    cudaMemcpyAsync(d_diag_tmp, d_diag, dataBase_.num_cells * sizeof(double), cudaMemcpyDeviceToDevice, dataBase_.stream);
    UELLSolver->solve(dataBase_, d_internal_coeffs_solve + dataBase_.num_boundary_surfaces, d_boundary_coeffs_solve + dataBase_.num_boundary_surfaces, &patch_type[0], d_diag_tmp, d_ellValues, d_ellCols, maxcount,  d_source_solve + dataBase_.num_cells, dataBase_.d_u + dataBase_.num_cells);
    cudaMemcpyAsync(d_diag_tmp, d_diag, dataBase_.num_cells * sizeof(double), cudaMemcpyDeviceToDevice, dataBase_.stream);
    UELLSolver->solve(dataBase_, d_internal_coeffs_solve + 2 * dataBase_.num_boundary_surfaces, d_boundary_coeffs_solve + 2 * dataBase_.num_boundary_surfaces, &patch_type[0], d_diag_tmp, d_ellValues, d_ellCols, maxcount, d_source_solve + 2 * dataBase_.num_cells, dataBase_.d_u + 2 * dataBase_.num_cells);
    TICK_END_EVENT(ELL PBiCGStab solve);
#endif
    num_iteration++;
}

void dfUEqn::postProcess() {
    // postProcess of dfUEqn can not be moved to the end of dfUEqn::process(),
    // because dataBase_.d_u is modified in dfpEqn and we only need the result of the last change
    // copy u and boundary_u to host
    permute_vector_d2h(dataBase_.stream, dataBase_.num_cells, dataBase_.d_u, d_u_host_order);
    checkCudaErrors(cudaMemcpyAsync(dataBase_.h_u, d_u_host_order, dataBase_.cell_value_vec_bytes, cudaMemcpyDeviceToHost, dataBase_.stream));
    sync();
}

void dfUEqn::A(double *Psi) {
    fvMtx_A(dataBase_.stream, dataBase_.num_cells, dataBase_.num_surfaces, dataBase_.num_boundary_surfaces, 
            dataBase_.d_boundary_face_cell, d_internal_coeffs, dataBase_.d_volume, d_diag, d_A_pEqn);
    checkCudaErrors(cudaMemcpyAsync(h_A_pEqn, d_A_pEqn, dataBase_.cell_value_bytes, cudaMemcpyDeviceToHost, dataBase_.stream));
    sync();
    // TODO: correct Boundary conditions
    memcpy(Psi, h_A_pEqn, dataBase_.cell_value_bytes);
}

void dfUEqn::getrAU(cudaStream_t stream, ncclComm_t comm, int num_cells, int num_surfaces, int num_boundary_surfaces, 
        const int *neighbor_peer, int num_patches, const int *patch_size, const int *patch_type,
        const int *boundary_cell_face, const double *boundary_delta_coeffs, const double *internal_coeffs, const double *volume, 
        const double *diag, double *rAU, double *boundary_rAU)
{
    checkCudaErrors(cudaMemcpyAsync(rAU, diag, num_cells * sizeof(double), cudaMemcpyDeviceToDevice, stream));

    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_boundary_surfaces + threads_per_block - 1) / threads_per_block;
    addAveInternaltoDiagUeqn<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_boundary_surfaces, boundary_cell_face, 
            internal_coeffs, rAU);
    
    blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    divide_cell_volume_scalar_reverse<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, volume, rAU);

    correct_boundary_conditions_scalar(stream, comm, neighbor_peer, num_boundary_surfaces, num_patches, patch_size, patch_type, boundary_delta_coeffs,
            boundary_cell_face, rAU, boundary_rAU, dataBase_.cyclicNeighbor.data(), dataBase_.patchSizeOffset.data(), dataBase_.d_boundary_weight);
}

void dfUEqn::getTurbulenceKEpsilon_Smagorinsky(cudaStream_t stream, int num_cells, int num_boundary_surfaces, 
        const double *grad_U_tsr, const double *volume, 
        double *delta, double *turbulence_k, double *turbulence_epsilon)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    ueqn_calculate_turbulence_k_Smagorinsky<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, grad_U_tsr, volume,
            1.048, 0.094, delta, turbulence_k);
    
    ueqn_calculate_turbulence_epsilon_Smagorinsky<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, turbulence_k, 
            delta, 1.048, turbulence_epsilon);
}

void dfUEqn::UEqnGetHbyA(cudaStream_t stream, ncclComm_t comm, const int *neighbor_peer, 
        int num_cells, int num_surfaces, int num_boundary_surfaces, 
        const int *lowerAddr, const int *upperAddr, const double *volume, const double *u,
        int num_patches, const int *patch_size, const int *patch_type, const int *patch_type_U,
        const int *boundary_cell_face, const double *internal_coffs, const double *boundary_coeffs, const double *boundary_weight,
        const double *lower, const double *upper, const double *source, const double *psi, 
        const double *rAU, const double *boundary_rAU, const double *boundary_u,
        const int *cyclicNeighbor, const int *patchSizeOffset,
        double *HbyA, double *boundary_HbyA)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_boundary_surfaces + threads_per_block - 1) / threads_per_block;
    checkCudaErrors(cudaMemcpyAsync(HbyA, source, num_cells * 3 * sizeof(double), cudaMemcpyDeviceToDevice, stream));

    ueqn_addBoundaryDiag<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_boundary_surfaces, boundary_cell_face, 
            internal_coffs, psi, HbyA);

    blocks_per_grid = (num_surfaces + threads_per_block - 1) / threads_per_block;
    ueqn_lduMatrix_H<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_surfaces, lowerAddr, upperAddr, 
            lower, upper, psi, HbyA);
    
    int offset = 0;
    for (int i = 0; i < num_patches; i++) {
        if (patch_size[i] == 0) continue;
        threads_per_block = 64;
        blocks_per_grid = (patch_size[i] + threads_per_block - 1) / threads_per_block;
        // TODO: just non-coupled patch type now
        if (patch_type[i] == boundaryConditions::extrapolated) {
            ueqn_addBoundarySrc_unCoupled<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, patch_size[i], offset, 
                    num_boundary_surfaces, boundary_cell_face, boundary_coeffs, HbyA);
        } else if (patch_type[i] == boundaryConditions::processor
                    || patch_type[i] == boundaryConditions::processorCyclic) {
            ueqn_addBoundarySrc_processor<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, patch_size[i], offset, 
                    num_boundary_surfaces, boundary_cell_face, boundary_coeffs, boundary_u, HbyA);
            offset += patch_size[i] * 2;
            continue;
        } else if (patch_type[i] == boundaryConditions::cyclic) {
            ueqn_addBoundarySrc_cyclic<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, patch_size[i], offset,
                    patchSizeOffset[cyclicNeighbor[i]], num_boundary_surfaces, boundary_cell_face, boundary_coeffs, u, HbyA);
        } else {
            fprintf(stderr, "%s %d, boundaryConditions other than zeroGradient are not support yet!\n", __FILE__, __LINE__);
        }
        offset += patch_size[i];
    }

    // divide volume and correct boundary conditions
    blocks_per_grid = (num_cells + threads_per_block - 1) / threads_per_block;
    ueqn_divide_cell_volume_vec<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, volume, HbyA);
    correct_boundary_conditions_vector(stream, comm, neighbor_peer, num_boundary_surfaces, num_cells, num_patches, 
            patch_size, patch_type, boundary_weight, boundary_cell_face, HbyA, boundary_HbyA,
            cyclicNeighbor, patchSizeOffset);

    // multi rAU
    scalar_field_multiply_vector_field(stream, num_cells, rAU, HbyA, HbyA, num_boundary_surfaces, boundary_rAU, 
            boundary_HbyA, boundary_HbyA);

    // constrainHbyA
    offset = 0;
    for (int i = 0; i < num_patches; i++) {
        if (patch_size[i] == 0) continue;
        threads_per_block = 64;
        blocks_per_grid = (patch_size[i] + threads_per_block - 1) / threads_per_block;
        // TODO: just non-coupled patch type now
        if (patch_type_U[i] == boundaryConditions::fixedValue) {
            correctBoundary_HbyA_fixedValueU<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_boundary_surfaces, patch_size[i], offset, 
                    boundary_u, boundary_HbyA);
        }
        offset += patch_size[i];
    }
}

void dfUEqn::getHbyA()
{
    UEqnGetHbyA(dataBase_.stream, dataBase_.nccl_comm, dataBase_.neighbProcNo.data(),
            dataBase_.num_cells, dataBase_.num_surfaces, dataBase_.num_boundary_surfaces,
            dataBase_.d_owner, dataBase_.d_neighbor, dataBase_.d_volume, dataBase_.d_u, 
            dataBase_.num_patches, dataBase_.patch_size.data(), dataBase_.patch_type_extropolated.data(),
            patch_type.data(),dataBase_.d_boundary_face_cell, d_internal_coeffs, d_boundary_coeffs, dataBase_.d_boundary_weight,
            d_lower, d_upper, d_source, dataBase_.d_u, dataBase_.d_rAU, dataBase_.d_boundary_rAU, 
            dataBase_.d_boundary_u, dataBase_.cyclicNeighbor.data(), dataBase_.patchSizeOffset.data(),
            dataBase_.d_HbyA, dataBase_.d_boundary_HbyA);
}

void dfUEqn::ueqn_ldu_to_csr(cudaStream_t stream, int num_cells, int num_surfaces, int num_boundary_surface, int num_Nz, 
        const int* boundary_cell_face, const int *ldu_to_csr_index, const int *diag_to_csr_index,
        int num_patches, const int *patch_size, const int *patch_type, const double *vf, const double *boundary_vf,
        const double *ldu, double *external, const double *source, const double *internal_coeffs, const double *boundary_coeffs,
        const int *cyclicNeighbor, const int *patchSizeOffset, double *A, double *b)
{
    // add external to ldu
    int bou_offset = 0, ext_offset = 0;
    size_t threads_per_block, blocks_per_grid;
    for (int i = 0; i < num_patches; i++) {
        if (patch_size[i] == 0) continue;
        if (patch_type[i] == boundaryConditions::processor
            || patch_type[i] == boundaryConditions::processorCyclic) {
            threads_per_block = 64;
            blocks_per_grid = (patch_size[i] + threads_per_block - 1) / threads_per_block;
            ueqn_add_external_entry_kernal<<<blocks_per_grid, threads_per_block, 0, stream>>>(patch_size[i], bou_offset, 
                    ext_offset, boundary_coeffs, external);
            bou_offset += patch_size[i] * 2;
            ext_offset += patch_size[i];
        } else if (patch_type[i] == boundaryConditions::cyclic) {
            threads_per_block = 64;
            blocks_per_grid = (patch_size[i] + threads_per_block - 1) / threads_per_block;
            ueqn_add_external_entry_kernal<<<blocks_per_grid, threads_per_block, 0, stream>>>(patch_size[i], bou_offset, 
                    ext_offset, boundary_coeffs, external);
            bou_offset += patch_size[i];
            ext_offset += patch_size[i];
        } else {
            bou_offset += patch_size[i];
        }
    }
    
    // construct csr matrix and RHS vec
    threads_per_block = 1024;
    blocks_per_grid = (num_Nz + threads_per_block - 1) / threads_per_block;
    checkCudaErrors(cudaMemcpyAsync(b, source, num_cells * 3 * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    ueqn_ldu_to_csr_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_Nz, ldu_to_csr_index, ldu, A);

    // add coeff to source and diagnal
    bou_offset = 0;
    for (int i = 0; i < num_patches; i++) {
        if (patch_size[i] == 0) continue;
        threads_per_block = 64;
        blocks_per_grid = (patch_size[i] + threads_per_block - 1) / threads_per_block;
        if (patch_type[i] == boundaryConditions::processor
            || patch_type[i] == boundaryConditions::processorCyclic) {
            ueqn_add_boundary_diag_src_couple<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_Nz, num_boundary_surface, 
                    patch_size[i], bou_offset, boundary_cell_face, internal_coeffs, diag_to_csr_index, A);
            bou_offset += patch_size[i] * 2;
        } else if (patch_type[i] == boundaryConditions::cyclic) {
            ueqn_add_boundary_diag_src_couple<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_Nz, num_boundary_surface, 
                    patch_size[i], bou_offset, boundary_cell_face, internal_coeffs, diag_to_csr_index, A);
            bou_offset += patch_size[i];
        } else {
            ueqn_add_boundary_diag_src_unCouple<<<blocks_per_grid, threads_per_block, 0, stream>>>(num_cells, num_Nz, num_boundary_surface, 
                    patch_size[i], bou_offset, boundary_cell_face, internal_coeffs, boundary_coeffs, diag_to_csr_index, A, b);
            bou_offset += patch_size[i];
        }
    }
}

__global__ void ldu2csr(int num_surfaces, int* index, double* value, double* result){
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_surfaces)
        return;

    result[i] = value[index[i]];
}
void dfUEqn::ueqn_ldu_to_csr_no_diag
(int num_surfaces, int* index, double* value, double* result)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (num_surfaces + threads_per_block - 1) / threads_per_block;

    ldu2csr<<<blocks_per_grid, threads_per_block, 0 ,dataBase_.stream>>>
        (num_surfaces, index, value, result);
}

int dfUEqn::get_ell_row_maxcount
(
    int row_,
    int num_surfaces,
    int* lowerAddr,
    int* upperAddr
)
{
    int maxcount = 0;
    int* ellColsperRow;
    ellColsperRow = (int*)malloc(sizeof(int) * row_);
    for(int i = 0; i < row_; i++){
        ellColsperRow[i] = 0;
    }
    for(int i = 0; i < num_surfaces; i++){
        int lowerRow = upperAddr[i];
        int upperRow = lowerAddr[i];
        ellColsperRow[lowerRow] += 1;
        ellColsperRow[upperRow] += 1;
    }
    for(int i = 0; i < row_; i++){
        maxcount = std::max(maxcount, ellColsperRow[i]);
    }
    return maxcount;
}

void dfUEqn::ueqn_ldu_to_ell_no_diag
(
    int row_,
    int num_surfaces,
    double* lower,
    double* upper,
    int* lowerAddr,
    int* upperAddr,
    double* ellValues,
    int* ellCols,
    int maxcount
)
{
    std::vector<int> rowOffset(row_, 0);
    // lower
    for(int i = 0; i < num_surfaces; i++){
        int row = upperAddr[i];
        int index = maxcount * row + rowOffset[row];
        ellValues[index] = lower[i];
        ellCols[index] = lowerAddr[i];
        rowOffset[row] += 1;
    }
    // upper
    for(int i = 0; i < num_surfaces; i++){
        int row = lowerAddr[i];
        int index = maxcount * row + rowOffset[row];
        ellValues[index] = upper[i];
        ellCols[index] = upperAddr[i];    
        rowOffset[row] += 1;
    }
}

void dfUEqn::H(double *Psi) {
    fvMtx_H(dataBase_.stream, dataBase_.num_cells, dataBase_.num_surfaces, dataBase_.num_boundary_surfaces, 
            dataBase_.d_owner, dataBase_.d_neighbor, dataBase_.d_volume,
            dataBase_.num_patches, dataBase_.patch_size.data(), dataBase_.patch_type_extropolated.data(),
            dataBase_.d_boundary_face_cell, d_internal_coeffs, d_boundary_coeffs,
            d_lower, d_upper, d_source, dataBase_.d_u, d_H_pEqn, d_H_pEqn_perm);
    checkCudaErrors(cudaMemcpyAsync(h_H_pEqn, d_H_pEqn_perm, dataBase_.cell_value_vec_bytes, cudaMemcpyDeviceToHost, dataBase_.stream));
    sync();
    // TODO: correct Boundary conditions
    memcpy(Psi, h_H_pEqn, dataBase_.cell_value_vec_bytes);
}

void dfUEqn::correctPsi(double *Psi, double *boundary_psi) {
    checkCudaErrors(cudaMemcpy(d_u_host_order, Psi, dataBase_.cell_value_vec_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_boundary_u_host_order, boundary_psi, dataBase_.boundary_surface_value_vec_bytes, cudaMemcpyHostToDevice));
    permute_vector_h2d(dataBase_.stream, dataBase_.num_cells, d_u_host_order, dataBase_.d_u);
    permute_vector_h2d(dataBase_.stream, dataBase_.num_boundary_surfaces, d_boundary_u_host_order, dataBase_.d_boundary_u);
}

double* dfUEqn::getFieldPointer(const char* fieldAlias, location loc, position pos) {
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

    return pointer;
}

// #if defined DEBUG_
void dfUEqn::compareResult(const double *lower, const double *upper, const double *diag, 
        const double *source, const double *internal_coeffs, const double *boundary_coeffs, 
        // const double *tmpVal, const double *boundary_val,
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

    std::vector<double> h_source, h_source_ref;
    h_source.resize(dataBase_.num_cells * 3);
    h_source_ref.resize(dataBase_.num_cells * 3);
    for (int i = 0; i < dataBase_.num_cells; i++) {
        h_source_ref[0 * dataBase_.num_cells + i] = source[i * 3 + 0];
        h_source_ref[1 * dataBase_.num_cells + i] = source[i * 3 + 1];
        h_source_ref[2 * dataBase_.num_cells + i] = source[i * 3 + 2];
    }
    checkCudaErrors(cudaMemcpy(h_source.data(), d_source, dataBase_.cell_value_vec_bytes, cudaMemcpyDeviceToHost));
    fprintf(stderr, "check h_source\n");
    checkVectorEqual(dataBase_.num_cells * 3, h_source_ref.data(), h_source.data(), 1e-14, printFlag);
    DEBUG_TRACE;

    std::vector<double> h_internal_coeffs, h_internal_coeffs_ref;
    h_internal_coeffs.resize(dataBase_.num_boundary_surfaces * 3);
    h_internal_coeffs_ref.resize(dataBase_.num_boundary_surfaces * 3);
    for (int i = 0; i < dataBase_.num_boundary_surfaces; i++) {
        h_internal_coeffs_ref[0 * dataBase_.num_boundary_surfaces + i] = internal_coeffs[i * 3 + 0];
        h_internal_coeffs_ref[1 * dataBase_.num_boundary_surfaces + i] = internal_coeffs[i * 3 + 1];
        h_internal_coeffs_ref[2 * dataBase_.num_boundary_surfaces + i] = internal_coeffs[i * 3 + 2];
    }
    checkCudaErrors(cudaMemcpy(h_internal_coeffs.data(), d_internal_coeffs, dataBase_.boundary_surface_value_vec_bytes, cudaMemcpyDeviceToHost));
    fprintf(stderr, "check h_internal_coeffs\n");
    checkVectorEqual(dataBase_.num_boundary_surfaces * 3, h_internal_coeffs_ref.data(), h_internal_coeffs.data(), 1e-14, printFlag);
    DEBUG_TRACE;

    std::vector<double> h_boundary_coeffs, h_boundary_coeffs_ref;
    h_boundary_coeffs.resize(dataBase_.num_boundary_surfaces * 3);
    h_boundary_coeffs_ref.resize(dataBase_.num_boundary_surfaces * 3);
    for (int i = 0; i < dataBase_.num_boundary_surfaces; i++) {
        h_boundary_coeffs_ref[0 * dataBase_.num_boundary_surfaces + i] = boundary_coeffs[i * 3 + 0];
        h_boundary_coeffs_ref[1 * dataBase_.num_boundary_surfaces + i] = boundary_coeffs[i * 3 + 1];
        h_boundary_coeffs_ref[2 * dataBase_.num_boundary_surfaces + i] = boundary_coeffs[i * 3 + 2];
    }
    checkCudaErrors(cudaMemcpy(h_boundary_coeffs.data(), d_boundary_coeffs, dataBase_.boundary_surface_value_vec_bytes, cudaMemcpyDeviceToHost));
    fprintf(stderr, "check h_boundary_coeffs\n");
    checkVectorEqual(dataBase_.num_boundary_surfaces * 3, h_boundary_coeffs_ref.data(), h_boundary_coeffs.data(), 1e-14, printFlag);
    DEBUG_TRACE;

    // std::vector<double> h_tmpVal, h_tmpVal_ref, h_boundary_val, h_boundary_val_ref;
    // h_tmpVal.resize(dataBase_.num_cells * 9);
    // h_tmpVal_ref.resize(dataBase_.num_cells * 9);
    // h_boundary_val.resize(dataBase_.num_boundary_surfaces * 9);
    // h_boundary_val_ref.resize(dataBase_.num_boundary_surfaces * 9);
    // for (int i = 0; i < dataBase_.num_cells; i++) {
    //     h_tmpVal_ref[0 * dataBase_.num_cells + i] = tmpVal[i * 9 + 0];
    //     h_tmpVal_ref[1 * dataBase_.num_cells + i] = tmpVal[i * 9 + 1];
    //     h_tmpVal_ref[2 * dataBase_.num_cells + i] = tmpVal[i * 9 + 2];
    //     h_tmpVal_ref[3 * dataBase_.num_cells + i] = tmpVal[i * 9 + 3];
    //     h_tmpVal_ref[4 * dataBase_.num_cells + i] = tmpVal[i * 9 + 4];
    //     h_tmpVal_ref[5 * dataBase_.num_cells + i] = tmpVal[i * 9 + 5];
    //     h_tmpVal_ref[6 * dataBase_.num_cells + i] = tmpVal[i * 9 + 6];
    //     h_tmpVal_ref[7 * dataBase_.num_cells + i] = tmpVal[i * 9 + 7];
    //     h_tmpVal_ref[8 * dataBase_.num_cells + i] = tmpVal[i * 9 + 8];
    // }
    // for (int i = 0; i < dataBase_.num_boundary_surfaces; i++){
    //     h_boundary_val_ref[0 * dataBase_.num_boundary_surfaces + i] = boundary_val[i * 9 + 0];
    //     h_boundary_val_ref[1 * dataBase_.num_boundary_surfaces + i] = boundary_val[i * 9 + 1];
    //     h_boundary_val_ref[2 * dataBase_.num_boundary_surfaces + i] = boundary_val[i * 9 + 2];
    //     h_boundary_val_ref[3 * dataBase_.num_boundary_surfaces + i] = boundary_val[i * 9 + 3];
    //     h_boundary_val_ref[4 * dataBase_.num_boundary_surfaces + i] = boundary_val[i * 9 + 4];
    //     h_boundary_val_ref[5 * dataBase_.num_boundary_surfaces + i] = boundary_val[i * 9 + 5];
    //     h_boundary_val_ref[6 * dataBase_.num_boundary_surfaces + i] = boundary_val[i * 9 + 6];
    //     h_boundary_val_ref[7 * dataBase_.num_boundary_surfaces + i] = boundary_val[i * 9 + 7];
    //     h_boundary_val_ref[8 * dataBase_.num_boundary_surfaces + i] = boundary_val[i * 9 + 8];
    // }
    // checkCudaErrors(cudaMemcpy(h_tmpVal.data(), d_grad_u, dataBase_.cell_value_tsr_bytes, cudaMemcpyDeviceToHost));
    // checkCudaErrors(cudaMemcpy(h_boundary_val.data(), d_boundary_grad_u, dataBase_.boundary_surface_value_tsr_bytes, cudaMemcpyDeviceToHost));
    // fprintf(stderr, "check h_grad_U\n");
    // checkVectorEqual(dataBase_.num_cells * 9, h_tmpVal_ref.data(), h_tmpVal.data(), 1e-14, printFlag);
    // fprintf(stderr, "check h_boundary_grad_U\n");
    // checkVectorEqual(dataBase_.num_boundary_surfaces * 9, h_boundary_val_ref.data(), h_boundary_val.data(), 1e-14, printFlag);
    // DEBUG_TRACE;
}
// #endif
void dfUEqn::compareHbyA(const double *HbyA, const double *boundary_HbyA, bool printFlag)
{
    double *h_HbyA = new double[dataBase_.num_cells * 3];
    double *h_HbyA_ref = new double[dataBase_.num_cells * 3];
    double *h_boundary_HbyA = new double[dataBase_.num_boundary_surfaces * 3];
    double *h_boundary_HbyA_ref = new double[dataBase_.num_boundary_surfaces * 3];

    // permute
    for (int i = 0; i < dataBase_.num_cells; i++)
    {
        h_HbyA_ref[dataBase_.num_cells * 0 + i] = HbyA[i * 3 + 0];
        h_HbyA_ref[dataBase_.num_cells * 1 + i] = HbyA[i * 3 + 1];
        h_HbyA_ref[dataBase_.num_cells * 2 + i] = HbyA[i * 3 + 2];
    }
    for (int i = 0; i < dataBase_.num_boundary_surfaces; i++)
    {
        h_boundary_HbyA_ref[dataBase_.num_boundary_surfaces * 0 + i] = boundary_HbyA[i * 3 + 0];
        h_boundary_HbyA_ref[dataBase_.num_boundary_surfaces * 1 + i] = boundary_HbyA[i * 3 + 1];
        h_boundary_HbyA_ref[dataBase_.num_boundary_surfaces * 2 + i] = boundary_HbyA[i * 3 + 2];
    }
    checkCudaErrors(cudaMemcpy(h_HbyA, dataBase_.d_HbyA, dataBase_.cell_value_vec_bytes, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_boundary_HbyA, dataBase_.d_boundary_HbyA, dataBase_.boundary_surface_value_vec_bytes, cudaMemcpyDeviceToHost));
    
    // check result
    fprintf(stderr, "check h_HbyA\n");
    checkVectorEqual(dataBase_.num_cells * 3, h_HbyA_ref, h_HbyA, 1e-10, printFlag);
    fprintf(stderr, "check h_boundary_HbyA\n");
    checkVectorEqual(dataBase_.num_boundary_surfaces * 3, h_boundary_HbyA_ref, h_boundary_HbyA, 1e-10, printFlag);
}

void dfUEqn::compareU(const double *U, const double *boundary_U, bool printFlag)
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

void dfUEqn::comparerAU(const double *rAU, const double *boundary_rAU, bool printFlag)
{
    double *h_rAU = new double[dataBase_.num_cells];
    double *h_boundary_rAU = new double[dataBase_.num_boundary_surfaces];

    checkCudaErrors(cudaMemcpy(h_rAU, dataBase_.d_rAU, dataBase_.cell_value_bytes, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_boundary_rAU, dataBase_.d_boundary_rAU, dataBase_.boundary_surface_value_bytes, cudaMemcpyDeviceToHost));

    fprintf(stderr, "check h_rAU\n");
    checkVectorEqual(dataBase_.num_cells, rAU, h_rAU, 1e-10, printFlag);
    fprintf(stderr, "check h_boundary_rAU\n");
    checkVectorEqual(dataBase_.num_boundary_surfaces, boundary_rAU, h_boundary_rAU, 1e-10, printFlag);

    delete h_rAU;
    delete h_boundary_rAU;
}
