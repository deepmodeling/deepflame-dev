#pragma once
#include "dfMatrixDataBase.H"

class CSRSmoother
{

public:

    CSRSmoother(){}
    virtual ~CSRSmoother(){}

    virtual void smooth(
        cudaStream_t stream,
        int nSweeps,
        int nCells,
        double* psi,
        double* source,
        double* off_diag_value_Ptr,
        int* off_diag_rowptr_Ptr, 
        int* off_diag_colidx_Ptr,
        double* diagPtr
    ) = 0;
};

class CSRJacobiSmoother : public CSRSmoother{
public:

    CSRJacobiSmoother(){}
    virtual ~CSRJacobiSmoother(){}

    virtual void smooth(
        cudaStream_t stream,
        int nSweeps,
        int nCells,
        double* psi,
        double* source,
        double* off_diag_value_Ptr,
        int* off_diag_rowptr_Ptr, 
        int* off_diag_colidx_Ptr,
        double* diagPtr
    )override;
};