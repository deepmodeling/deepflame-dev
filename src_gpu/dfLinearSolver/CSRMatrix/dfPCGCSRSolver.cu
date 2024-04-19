#include "dfCSRSolver.H"

// kernel functions for PBiCGStab solver

void PCGCSRSolver::initialize(const int nCells, const size_t boundary_surface_value_bytes)
{
    // cudamalloc variables related to PCGSolver

    // preconditioner
    precond_ = new GAMGCSRPreconditioner();
    precond_->initialize();
}

void PCGCSRSolver::solve
(
    const dfMatrixDataBase& dataBase,
    const double* d_internal_coeffs,
    const double* d_boundary_coeffs,
    int* patch_type,
    double* diagPtr,
    const double* off_diag_value,
    const double *rhs, 
    double *psi
)
{
    // Implement the PCG solver with GAMG preconditioner here
    // dataBase.d_csr_col_index_no_diag
    // dataBase.d_csr_row_index_no_diag
    // dataBase.interfaceFlag
};