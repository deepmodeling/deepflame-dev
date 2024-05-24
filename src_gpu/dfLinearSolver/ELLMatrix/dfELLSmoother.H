#pragma once
#include "dfMatrixDataBase.H"

class ELLSmoother
{

public:

    ELLSmoother(){}
    virtual ~ELLSmoother(){}

    virtual void smooth(
        cudaStream_t stream,
        int nSweeps,
        int nCells,
        double* psi,
        double* source,
        int ell_row_maxcount,
        int* d_ell_cols, 
        double* d_ell_values,
        double* diagPtr
    ) = 0;
};
    
class ELLJacobiSmoother : public ELLSmoother{
public:

    ELLJacobiSmoother(){}
    virtual ~ELLJacobiSmoother(){}

    virtual void smooth(
        cudaStream_t stream,
        int nSweeps,
        int nCells,
        double* psi,
        double* source,
        int ell_row_maxcount,
        int* d_ell_cols, 
        double* d_ell_values,
        double* diagPtr
    )override;
};