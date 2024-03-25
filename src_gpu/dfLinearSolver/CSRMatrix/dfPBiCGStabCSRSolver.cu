#include "dfCSRSolver.H"

// kernel functions for PBiCGStab solver

void PBiCGStabCSRSolver::solve
(
    const dfMatrixDataBase& dataBase,
    const double* diagPtr,
    const double* off_diag_value,
    const double *rhs, 
    double *psi
)
{
    // Implement the PBiCGStab solver here
    // dataBase.d_csr_col_index_no_diag
    // dataBase.d_csr_row_index_no_diag
    // dataBase.interfaceFlag
};