Info << "Enter pEqn_CPU" << endl;

if (!pimple.simpleRho())
{
    rho = thermo.rho();
}

// Thermodynamic density needs to be updated by psi*d(p) after the
// pressure solution
const volScalarField psip0(psi*p);

volScalarField rAU(1.0/UEqn.A());
surfaceScalarField rhorAUf("rhorAUf", fvc::interpolate(rho*rAU));
volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U, p));

if (pimple.nCorrPiso() <= 1)
{
    tUEqn.clear();
}

surfaceScalarField phiHbyA
(
    "phiHbyA",
    fvc::interpolate(rho)*fvc::flux(HbyA)
  + rhorAUf*fvc::ddtCorr(rho, U, phi, rhoUf)
);

fvc::makeRelative(phiHbyA, rho, U);

int flag_mpi_init;
MPI_Initialized(&flag_mpi_init);
if(flag_mpi_init) MPI_Barrier(PstreamGlobals::MPI_COMM_FOAM);

// Update the pressure BCs to ensure flux consistency
constrainPressure(p, rho, U, phiHbyA, rhorAUf);
//start = MPI_Wtime();
if (pimple.transonic())
{
    surfaceScalarField phid
    (
        "phid",
        (fvc::interpolate(psi)/fvc::interpolate(rho))*phiHbyA
    );

    phiHbyA -= fvc::interpolate(psi*p)*phiHbyA/fvc::interpolate(rho);

    fvScalarMatrix pDDtEqn
    (
        fvc::ddt(rho) + psi*correction(fvm::ddt(p))
      + fvc::div(phiHbyA) + fvm::div(phid, p)
    );

    while (pimple.correctNonOrthogonal())
    {
        fvScalarMatrix pEqn(pDDtEqn - fvm::laplacian(rhorAUf, p));

        // Relax the pressure equation to ensure diagonal-dominance
        pEqn.relax();
        
        assert(false);
        pEqn.solve();

        if (pimple.finalNonOrthogonalIter())
        {
            phi = phiHbyA + pEqn.flux();
        }
    }
}
else
{
    fvScalarMatrix pDDtEqn
    (
        fvc::ddt(rho) + 
        psi*correction(fvm::ddt(p))
        + fvc::div(phiHbyA)
    );

    while (pimple.correctNonOrthogonal())
    {
        fvScalarMatrix pEqn(pDDtEqn - fvm::laplacian(rhorAUf, p));
        
#ifdef USE_DF_MATRIX
        dfMatrix df_pEqn(pEqn, regionPtr);
        auto& psi = const_cast<volScalarField&>(pEqn.psi());
        auto& source = pEqn.source();
        auto& internalCoeffs = pEqn.internalCoeffs();
        auto& boundaryCoeffs = pEqn.boundaryCoeffs();
        df_pEqn.solve(psi, source, internalCoeffs, boundaryCoeffs);
#else
        pEqn.solve();
#endif
        
        if (pimple.finalNonOrthogonalIter())
        {
            phi = phiHbyA + pEqn.flux();
        }
    }

}


bool limitedp = pressureControl.limit(p);

// Thermodynamic density update
thermo.correctRho(psi*p - psip0);


if (limitedp)
{
    rho = thermo.rho();
}

#include "rhoEqn.H"
#include "compressibleContinuityErrs.H"

// Explicitly relax pressure for momentum corrector
// p.relax();

U = HbyA - rAU*fvc::grad(p);
U.correctBoundaryConditions();
K = 0.5*magSqr(U);

if (pimple.simpleRho())
{
    rho = thermo.rho();
}

// Correct rhoUf if the mesh is moving
fvc::correctRhoUf(rhoUf, rho, U, phi);

if (thermo.dpdt())
{
    dpdt = fvc::ddt(p);

    if (mesh.moving())
    {
        dpdt -= fvc::div(fvc::meshPhi(rho, U), p);
    }
}

Info << "Exit pEqn_CPU" << endl;