#ifndef dfMatrix_H
#define dfMatrix_H

#include <stdio.h>
#include <unistd.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <map>
#include <iostream>
#include <ctime>
#include <math.h>

class dfMatrix
{
private:
    /**
     * @brief data size
     */
    // - number of cell size
    int num_cells;
    // - number of face size
    int num_surfaces;
    // - number of offdiagnal entry size (2*num_surfaces)
    int num_faces;
    // - number of boundary cells
    int num_boundary_cells;
    // - number of boundary faces
    int num_boundary_faces;

    /**
     * @brief input data (now these values are named according to UEqn)
     */
    // - mesh variables
    // - csr_row_index
    std::vector<int> h_csr_row_index_vec;
    // - csr_col_index
    std::vector<int> h_csr_col_index_vec;
    // - csr_diag_index
    std::vector<int> h_csr_diag_index_vec;

    // - the pre-permutated and post-permutated interpolation weight list
    std::vector<double> h_weight_vec_init, h_weight_vec;
    // - the pre-permutated and post-permutated flux (phi) list
    std::vector<double> h_phi_vec_init, h_phi_vec;
    // - the pre-permutated and post-permutated cell face vector list
    std::vector<double> h_face_vector_vec_init, h_face_vector_vec;
    // - the host pointer to rho_new, rho_old, velocity_old, pressure and volume list
    double *h_rho_new = nullptr, *h_rho_old = nullptr, *h_velocity_old = nullptr, 
    *h_pressure = nullptr;
    const double *h_volume = nullptr;
    // - the host pointer to the pre-permutated and post-permutated interpolation weight list
    double *h_weight_init = nullptr, *h_weight = nullptr;
    // - the host pointer to the pre-permutated and post-permutated flux (phi) list
    double *h_phi_init = nullptr, *h_phi = nullptr;
    // - the host pointer to the pre-permutated and post-permutated cell face vector list
    double *h_face_vector_init = nullptr, *h_face_vector = nullptr;
    // - the device pointer to rho_new, rho_old, velocity_old, pressure and volume list
    double *d_rho_new = nullptr, *d_rho_old = nullptr, *d_velocity_old = nullptr, 
    *d_pressure = nullptr, *d_volume = nullptr;
    // - the device pointer to the pre-permutated and post-permutated interpolation weight list
    double *d_weight_init = nullptr, *d_weight = nullptr;
    // - the device pointer to the pre-permutated and post-permutated flux (phi) list
    double *d_phi_init = nullptr, *d_phi = nullptr;
    // - the device pointer to the pre-permutated and post-permutated cell face vector list
    double *d_face_vector_init = nullptr, *d_face_vector = nullptr;

    double rdelta_t = 1/1e-6;

    /**
     * @brief boundary related variables
     */
    int *h_boundary_cell_offset = nullptr, *d_boundary_cell_offset=nullptr;
    int *h_boundary_cell_id = nullptr, *d_boundary_cell_id = nullptr;
    double *h_internal_coeffs = nullptr, *h_boundary_coeffs = nullptr,
    *h_boundary_pressure = nullptr, *h_boundary_face_vector = nullptr,
    *d_internal_coeffs = nullptr, *d_boundary_coeffs = nullptr,
    *d_internal_coeffs_init = nullptr, *d_boundary_coeffs_init = nullptr,
    *d_boundary_pressure = nullptr, *d_boundary_face_vector = nullptr,
    *d_boundary_pressure_init = nullptr;
    std::vector<int> boundPermutationList;
    std::vector<double> ueqn_internalCoeffs, ueqn_boundaryCoeffs;
    std::vector<double> boundary_face_vector;
    std::vector<double> boundary_pressure;
    std::vector<int> boundary_cell_offset;
    std::vector<int> boundary_cell_id;

    // - the device pointer to the permutated index list
    std::vector<int> permedIndex;
    int *d_permedIndex=nullptr;
    int *d_bouPermedIndex = nullptr;


    // bytesize
    // - bytes of diagnal entries
    size_t cell_bytes;
    // - bytes of diagnal entries (vector)
    size_t cell_vec_bytes;
    // - bytes of diagnal index
    size_t cell_index_bytes;
     // - bytes of diagnal index
    size_t face_bytes;
    size_t face_vec_bytes;
    size_t face_index_bytes;

    size_t boundary_cell_bytes;
    size_t boundary_cell_vec_bytes;
    size_t boundary_cell_index_bytes;

    size_t boundary_face_bytes;
    size_t boundary_face_vec_bytes;
    size_t boundary_face_index_bytes;

    // A_csr has one more element in each row: itself
    size_t csr_row_index_bytes;
    size_t csr_col_index_bytes;
    size_t csr_value_bytes;
    size_t csr_value_vec_bytes;

    // extra matrix information
    double *d_turbSrc_A = nullptr, *d_turbSrc_b = nullptr, *d_turbSrc_A_init = nullptr;
    std::vector<double> h_turbSrc_init_mtx_vec, h_turbSrc_init_1mtx;
    std::vector<double> h_turbSrc_init_src_vec, h_turbSrc_src_vec;
    std::vector<int> tmpPermutatedList;
    int * d_tmpPermutatedList = nullptr;

    double *h_A_csr = nullptr, *h_b = nullptr, *h_psi = nullptr;
    double *d_A_csr = nullptr, *d_b = nullptr, *d_psi = nullptr;

    double time_monitor_CPU, time_monitor_GPU;



public:
    dfMatrix();
    dfMatrix(int num_surfaces, int num_cells, int num_boundary_faces, int &num_boundary_cells_output,
    const int *neighbour, const int *owner, const double* volume, const double* weight, const double* face_vector, 
    std::vector<double> boundary_face_vector_init, std::vector<int> boundary_cell_id_init);
    ~dfMatrix();

    void checkValue(bool print);

    void fvm_ddt(double *rho_old, double *rho_new, double* vector_old);

    void fvm_div(double* phi, double* ueqn_internalCoeffs_init,
    double* ueqn_boundaryCoeffs_init, double* boundary_pressure_init);

    void fvc_grad(double* pressure);

    void add_fvMatrix(double* turbSrc_low, double* turbSrc_diag, double* turbSrc_upp, double* turbSrc_source);

    void solve();
    void updatePsi(double* Psi);
};

#endif