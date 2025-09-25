#include "fvc.H"
#include "fvm.H"
#include "GenFvMatrix.H"

// namespace Foam
// {

// tmp<fvVectorMatrix>
// turbulenceModelLinearViscousStressDivDevRhoReff
// (
//     volVectorField& U,
//     incompressible::turbulenceModel& turbulence
// )
// {
//     return
//     (
//       - fvc::div((turbulence.alpha()*turbulence.rho()*turbulence.nuEff())*dev2(T(fvc::grad(U))))
//       - fvm::laplacian(turbulence.alpha()*turbulence.rho()*turbulence.nuEff(), U)
//     );
// }

// }