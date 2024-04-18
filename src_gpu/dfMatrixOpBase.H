#pragma once
#include <cuda_runtime.h>
#include <nccl.h>
#include "dfThermo.H"
class dfThermo;

// macros need to be in a certain order

// sequence 0: STREAM_ALLOCATOR can be open or not
#ifndef STREAM_ALLOCATOR
    #define STREAM_ALLOCATOR
#endif

// sequence 0: USE_GRAPH can be open or not
#ifndef USE_GRAPH
    #define USE_GRAPH
#endif

// sequence 0: TIME_GPU can be open or not
#ifndef TIME_GPU
    #define TIME_GPU
#endif

// sequence 1: TIME_GPU and USE_GRAPH can not be open at the same time
#if (defined TIME_GPU) && (defined USE_GRAPH)
    #undef USE_GRAPH
#endif

// sequence 2: STREAM_ALLOCATOR must be open if USE_GRAPH is open
#if (defined USE_GRAPH) && (!defined STREAM_ALLOCATOR)
    #define STREAM_ALLOCATOR
#endif

extern int myRank;

#define PRINT_PTR(x) { \
    fprintf(stderr, "rank[%d], %s %d, print ptr %s: %p\n", myRank, __FILE__, __LINE__, #x, x); \
}

extern __global__ void warmup();

#ifdef TIME_GPU
    #define WARM_UP \
        warmup<<<10, 1024, 0, stream>>>();

    #define TICK_INIT_EVENT \
        float time_elapsed_kernel=0;\
        cudaEvent_t start_kernel, stop_kernel;\
        checkCudaErrors(cudaEventCreate(&start_kernel));\
        checkCudaErrors(cudaEventCreate(&stop_kernel));

    #define TICK_START_EVENT \
        checkCudaErrors(cudaEventRecord(start_kernel,stream));

    #define TICK_END_EVENT(prefix) \
        checkCudaErrors(cudaEventRecord(stop_kernel,stream));\
        checkCudaErrors(cudaEventSynchronize(start_kernel));\
        checkCudaErrors(cudaEventSynchronize(stop_kernel));\
        checkCudaErrors(cudaEventElapsedTime(&time_elapsed_kernel,start_kernel,stop_kernel));\
        fprintf(stderr, "rank[%d], name: %s, time: %lf(ms)\n", myRank, #prefix, time_elapsed_kernel);

/*
    // the usage description:
    // if you want to profile the first kernel, please use WARM_UP before TICK_INIT_EVENT.
    // otherwise there is no need to use WARM_UP
    WARM_UP;
    // init event
    TICK_INIT_EVENT;
    // start event
    TICK_START_EVENT;
    // call your kernel, or kernels, or wrapper functions, e.g.:
    my_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(num, input, output);
    // end event with your specified name string, e.g.:
    TICK_END_EVENT(my_kernel);
*/

#else
    #define WARM_UP
    #define TICK_INIT_EVENT
    #define TICK_START_EVENT
    #define TICK_END_EVENT(prefix)
#endif

// tools
void permute_vector_d2h(cudaStream_t stream, int num_cells, const double *input, double *output);
void permute_vector_h2d(cudaStream_t stream, int num_cells, const double *input, double *output);

void field_add_scalar(cudaStream_t stream,
        int num, const double *input1, const double *input2, double *output,
        int num_boundary_surfaces, const double *boundary_input1, const double *boundary_input2, double *boundary_output);

void field_add_vector(cudaStream_t stream,
        int num_cells, const double *input1, const double *input2, double *output,
        int num_boundary_surfaces, const double *boundary_input1, const double *boundary_input2, double *boundary_output, double sign = 1.);

void field_add_vector(cudaStream_t stream,
        int num_cells, const double *input1, const double *input2, double *output, double sign = 1.);

void field_multiply_scalar(cudaStream_t stream,
        int num_cells, const double *input1, const double *input2, double *output,
        int num_boundary_surfaces, const double *boundary_input1, const double *boundary_input2, double *boundary_output);

void vector_half_mag_square(cudaStream_t stream, int num_cells, const double *vec_input, double *scalar_output,
        int num_boundary_surfaces, const double *boundary_vec_input, double *boundary_scalar_output);

void scalar_field_multiply_vector_field(cudaStream_t stream,
        int num_cells, const double *input1, const double *input2, double *output,
        int num_boundary_surfaces, const double *boundary_input1, const double *boundary_input2, double *boundary_output, double sign = 1.);

void scalar_field_multiply_vector_field(cudaStream_t stream,
        int num_cells, const double *input1, const double *input2, double *output, double sign = 1.);

void fvc_to_source_vector(cudaStream_t stream, int num_cells, const double *volume, const double *fvc_output, double *source);

void fvc_to_source_scalar(cudaStream_t stream, int num_cells, const double *volume, const double *fvc_output, double *source, double sign = 1.);

void ldu_to_csr_scalar(cudaStream_t stream, int num_cells, int num_surfaces, int num_boundary_surface, int num_Nz, 
        const int* boundary_cell_face, const int *ldu_to_csr_index,
        int num_patches, const int *patch_size, const int *patch_type,
        double* ldu, double *source, // b = source
        const double *internal_coeffs, const double *boundary_coeffs, double *A);

void ldu_to_csr(cudaStream_t stream, int num_cells, int num_surfaces, int num_boundary_surface,
        const int* boundary_cell_face, const int *ldu_to_csr_index, const int *diag_to_csr_index,
        const double *ldu, const double *internal_coeffs, const double *boundary_coeffs, double *source, double *A);

void update_boundary_coeffs_scalar(cudaStream_t stream,
        int num_patches, const int *patch_size, const int *patch_type,
        const double *boundary_delta_coeffs, const double *boundary_vf, const double *boundary_weight, 
        double *value_internal_coeffs, double *value_boundary_coeffs,
        double *gradient_internal_coeffs, double *gradient_boundary_coeffs, const double *energy_gradient = nullptr);

void update_boundary_coeffs_vector(cudaStream_t stream, int num_boundary_surfaces, int num_patches,
        const int *patch_size, const int *patch_type, const double *boundary_vf, 
        const double *boundary_deltaCoeffs, const double *boundary_weight,
        double *value_internal_coeffs, double *value_boundary_coeffs,
        double *gradient_internal_coeffs, double *gradient_boundary_coeffs);

void correct_boundary_conditions_processor_scalar(cudaStream_t stream, ncclComm_t comm,
        int peer, int num, int offset,
        const double *vf, const int *boundary_cell_face, double *vf_boundary);

void correct_boundary_conditions_scalar(cudaStream_t stream, ncclComm_t comm,
        const int *neighbor_peer, int num_boundary_surfaces, int num_patches,
        const int *patch_size, const int *patch_type, const double *boundary_delta_coeffs, 
        const int *boundary_cell_face, const double *vf, double *boundary_vf,
        const int *cyclicNeighbor, const int *patchSizeOffset, const double *boundary_weight,
        const double *boundary_T = nullptr, const double *boundary_y = nullptr,
        const double *thermo_gradient = nullptr, dfThermo *GPUThermo = nullptr);

void correct_boundary_conditions_vector(cudaStream_t stream, ncclComm_t comm,
        const int *neighbor_peer, int num_boundary_surfaces, int num_cells, int num_patches,
        const int *patch_size, const int *patch_type, const double *boundary_weight, 
        const int *boundary_cell_face, const double *vf, double *boundary_vf,
        const int *cyclicNeighbor, const int *patchSizeOffset);

void compute_upwind_weight(cudaStream_t stream, int num_surfaces, const double *phi, double *weight);

void compute_limitedLinear_weight(cudaStream_t stream, ncclComm_t comm, const int *neighbor_peer, 
        int num_surfaces, int num_cells, int num_boundary_surfaces,
        const int *lowerAddr, const int *upperAddr, const double *mesh_distance, 
        const double *weight, const double *Sf, const double *vf, const double *phi,  double *output, // end for internal
        int num_patches, const int *patch_size, const int *patch_type, const double *boundary_weight,
        const int *boundary_cell_face, const double *boundary_vf, const double *boundary_Sf,
        const double *volume, const double *boundary_mag_Sf, const double *boundary_phi, 
        // const double *boundary_distance, double *boundary_output, 
        const int *cyclicNeighbor, const int *patchSizeOffset,
        const double *boundary_deltaCoeffs);

// fvm ops

void fvm_ddt_vol_scalar_vol_scalar(cudaStream_t stream, int num_cells, double rDeltaT,
        const double *rho, const double *rho_old, const double *vf, const double *volume,
        double *diag, double *source, double sign = 1.);

void fvm_ddt_scalar(cudaStream_t stream, int num_cells, double rDeltaT, 
        const double *vf_old, const double *volume, 
        double *diag, double *source, double sign = 1.);

void fvm_ddt_vector(cudaStream_t stream, int num_cells, double rDeltaT,
        const double *rho, const double *rho_old, const double *vf, const double *volume,
        double *diag, double *source, double sign = 1.);

void fvm_div_scalar(cudaStream_t stream, int num_surfaces, const int *lowerAddr, const int *upperAddr,
        const double *phi, const double *weight,
        double *lower, double *upper, double *diag, // end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const double *boundary_phi, const double *value_internal_coeffs, const double *value_boundary_coeffs,
        double *internal_coeffs, double *boundary_coeffs, double sign = 1.);

void fvm_div_vector(cudaStream_t stream, int num_surfaces, int num_boundary_sourfaces, 
        const int *lowerAddr, const int *upperAddr,
        const double *phi, const double *weight,
        double *lower, double *upper, double *diag, // end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const double *boundary_phi, const double *value_internal_coeffs, const double *value_boundary_coeffs,
        double *internal_coeffs, double *boundary_coeffs, double sign = 1.);

void fvm_laplacian_scalar(cudaStream_t stream, int num_surfaces, int num_boundary_surfaces, 
        const int *lowerAddr, const int *upperAddr,
        const double *weight, const double *mag_sf, const double *delta_coeffs, const double *gamma,
        double *lower, double *upper, double *diag, // end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const double *boundary_mag_sf, const double *boundary_gamma,
        const double *gradient_internal_coeffs, const double *gradient_boundary_coeffs,
        double *internal_coeffs, double *boundary_coeffs, double sign = 1.);

void fvm_laplacian_vector(cudaStream_t stream, int num_surfaces, int num_boundary_surfaces, 
        const int *lowerAddr, const int *upperAddr,
        const double *weight, const double *mag_sf, const double *delta_coeffs, const double *gamma,
        double *lower, double *upper, double *diag, // end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const double *boundary_mag_sf, const double *boundary_gamma,
        const double *gradient_internal_coeffs, const double *gradient_boundary_coeffs,
        double *internal_coeffs, double *boundary_coeffs, double sign = 1.);

void fvm_laplacian_surface_scalar_vol_scalar(cudaStream_t stream, int num_surfaces, int num_boundary_surfaces, 
        const int *lowerAddr, const int *upperAddr,
        const double *mag_sf, const double *delta_coeffs, const double *gamma,
        double *lower, double *upper, double *diag, // end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const double *boundary_mag_sf, const double *boundary_gamma,
        const double *gradient_internal_coeffs, const double *gradient_boundary_coeffs,
        double *internal_coeffs, double *boundary_coeffs, double sign = 1.);

// fvc ops
// fvc_ddt doesn't consider to add fvc_output to source yet, which needs (fvc_output * volume * sign).
void fvc_ddt_vol_scalar_vol_scalar(cudaStream_t stream, int num_cells, double rDeltaT,
        const double *rho, const double *rho_old, const double *vf, const double *vf_old, const double *volume, 
        double *output, double sign = 1.);

void fvc_ddt_scalar(cudaStream_t stream, int num_cells, double rDeltaT,
        const double *vf, const double *vf_old, const double *volume, double *source, double sign);

void fvc_ddt_scalar_field(cudaStream_t stream, int num_cells, double rDeltaT,
        const double *vf, const double *vf_old, const double *volume, double *source, double sign = 1.);

void fvc_grad_vector(cudaStream_t stream, ncclComm_t comm, 
        int num_cells, int num_surfaces, int num_boundary_surfaces,
        const int *neighbor_peer, const int *lowerAddr, const int *upperAddr, 
        const double *weight, const double *Sf, const double *vf, double *output, // end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const int *boundary_cell_face, const double *boundary_vf, const double *boundary_Sf, const double *boundary_weight, 
        const double *volume, const double *boundary_mag_Sf, double *boundary_output,
        const int *cyclicNeighbor, const int *patchSizeOffset,
        const double *boundary_deltaCoeffs, double sign = 1.);

void fvc_div_surface_scalar(cudaStream_t stream, int num_cells, int num_surfaces, int num_boundary_surfaces,
        const int *lowerAddr, const int *upperAddr, const double *ssf, const int *boundary_cell_face,
        int num_patches, const int *patch_size, const int *patch_type,
        const double *boundary_ssf, const double *volume, double *output, double sign = 1.);

void fvc_div_cell_vector(cudaStream_t stream, int num_cells, int num_surfaces, int num_boundary_surfaces,
        const int *lowerAddr, const int *upperAddr,
        const double *weight, const double *Sf, const double *vf, double *output, // end for internal
        int num_patches, const int *patch_size, const int *patch_type, const int *boundary_cell_face,
        const double *boundary_weight, const double *boundary_vf, const double *boundary_Sf,
        const double *volume, double sign = 1.);

void fvc_div_cell_tensor(cudaStream_t stream, int num_cells, int num_surfaces, int num_boundary_surfaces, 
        const int *lowerAddr, const int *upperAddr, 
        const double *weight, const double *Sf, const double *vf, double *output, // end for internal
        int num_patches, const int *patch_size, const int *patch_type, const double *boundary_weight, 
        const int *boundary_cell_face, const double *boundary_vf, const double *boundary_Sf,
        const double *volume, double sign = 1.);

void fvc_div_surface_scalar_vol_scalar(cudaStream_t stream, int num_surfaces, 
        const int *lowerAddr, const int *upperAddr,
        const double *weight, const double *vf, const double *ssf, double *output, // end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const int *boundary_cell_face, const double *boundary_vf, const double *boundary_ssf, 
        double sign = 1.);

void fvc_grad_cell_scalar(cudaStream_t stream, int num_cells, int num_surfaces, int num_boundary_surfaces, 
        const int *lowerAddr, const int *upperAddr, 
        const double *weight, const double *Sf, const double *vf, double *output, // end for internal
        int num_patches, const int *patch_size, const int *patch_type, const double *boundary_weight,
        const int *boundary_cell_face, const double *boundary_vf, const double *boundary_Sf, const double *volume, double sign = 1.);

void fvc_grad_cell_scalar(cudaStream_t stream, int num_cells, int num_surfaces, int num_boundary_surfaces, 
        const int *lowerAddr, const int *upperAddr, 
        const double *weight, const double *Sf, const double *vf, double *output, // end for internal
        int num_patches, const int *patch_size, const int *patch_type, const double *boundary_weight,
        const int *boundary_cell_face, const double *boundary_vf, const double *boundary_Sf, const double *volume, 
        bool dividVol, double sign = 1.);

void fvc_grad_cell_scalar_withBC(cudaStream_t stream, ncclComm_t comm, const int *neighbor_peer,
        int num_cells, int num_surfaces, int num_boundary_surfaces,
        const int *lowerAddr, const int *upperAddr,
        const double *weight, const double *Sf, const double *vf, double *output, // end for internal
        int num_patches, const int *patch_size, const int *patch_type, const double *boundary_weight,
        const int *boundary_cell_face, const double *boundary_vf, const double *boundary_Sf,
        const double *volume, const double *boundary_mag_Sf, double *boundary_output,
        const int *cyclicNeighbor, const int *patchSizeOffset,
        const double *boundary_deltaCoeffs);

void fvc_laplacian_scalar(cudaStream_t stream, int num_cells, int num_surfaces, int num_boundary_surfaces,
        const int *lowerAddr, const int *upperAddr,
        const double *weight, const double *mag_sf, const double *delta_coeffs, const double *volume,
        const double *gamma, const double *vf, double *output, // end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const int *boundary_cell_face,
        const double *boundary_mag_sf, const double *boundary_delta_coeffs,
        const double *boundary_gamma, const double *boundary_vf, double sign = 1.);

void fvc_flux(cudaStream_t stream, int num_cells, int num_surfaces, int num_boundary_surfaces,
        const int *lowerAddr, const int *upperAddr, 
        const double *weight, const double *Sf, const double *vf, double *output, // end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const int *boundary_cell_face, const double *boundary_vf, const double *boundary_Sf, 
        double *boundary_output, double sign);

void fvc_interpolate(cudaStream_t stream, int num_cells, int num_surfaces,
        const int *lowerAddr, const int *upperAddr, 
        const double *weight, const double *vf, double *output, // end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const double *boundary_vf, double *boundary_output, double sign);

// others
void scale_dev2T_tensor(cudaStream_t stream, int num_cells, const double *vf1, double *vf2,
        int num_boundary_surfaces, const double *boundary_vf1, double *boundary_vf2);

void fvMtx_A(cudaStream_t stream, int num_cells, int num_surfaces, int num_boundary_surfaces, 
        const int *boundary_cell_face, const double *internal_coeffs, const double *volume, const double *diag, 
        double *A_pEqn);

void fvMtx_H(cudaStream_t stream, int num_cells, int num_surfaces, int num_boundary_surfaces, 
        const int *lowerAddr, const int *upperAddr, const double *volume,
        int num_patches, const int *patch_size, const int *patch_type,
        const int *boundary_cell_face, const double *internal_coffs, const double *boundary_coeffs, 
        const double *lower, const double *upper, const double *source, const double *psi, 
        double *H_pEqn, double *H_pEqn_perm);

void fvMtx_flux(cudaStream_t stream, int num_surfaces, int num_boundary_surfaces, 
        const int *lowerAddr, const int *upperAddr, const double *lower, const double *upper,
        const double *psi, double *output, //end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const int *boundary_cell_face, const double *internal_coeffs, const double *boundary_coeffs, 
        const int *cyclicNeighbor, const int *patchSizeOffset, const double *boundary_psi, double *boundary_output);

void solve_explicit_scalar(cudaStream_t stream, int num_cells, const double *diag, const double *source,
        double *psi);
