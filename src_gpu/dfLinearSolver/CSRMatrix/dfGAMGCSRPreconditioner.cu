#include "dfCSRPreconditioner.H"

// kernel functions for PBiCGStab solver

void GAMGCSRPreconditioner::initialize()
{
    // Implement the GAMG preconditioner here
};

void GAMGCSRPreconditioner::precondition
(
    double *wA,
    const double *rA
)
{
    // Implement the GAMG precondition procedure here
};