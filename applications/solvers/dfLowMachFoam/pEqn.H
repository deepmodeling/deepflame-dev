
if (!pimple.simpleRho())
{
    rho = thermo.rho();
}

// Thermodynamic density needs to be updated by psi*d(p) after the
// pressure solution
const volScalarField psip0(psi*p);

double constructMs = std::clock();

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
double constructMe = std::clock();
constructM_p += double( constructMe -  constructMs) / double(CLOCKS_PER_SEC);

MPI_Barrier(PstreamGlobals::MPI_COMM_FOAM);

// Update the pressure BCs to ensure flux consistency
constrainPressure(p, rho, U, phiHbyA, rhorAUf);
//start = std::clock();
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

        pEqn.solve();

        if (pimple.finalNonOrthogonalIter())
        {
            phi = phiHbyA + pEqn.flux();
        }
    }
}
else
{
    double start_psolve = std::clock();
    fvScalarMatrix pDDtEqn
    (
        fvc::ddt(rho) + psi*correction(fvm::ddt(p))
      + fvc::div(phiHbyA)
    );

    while (pimple.correctNonOrthogonal())
    {
        fvScalarMatrix pEqn(pDDtEqn - fvm::laplacian(rhorAUf, p));
        
 
        pEqn.solve();

        
        if (pimple.finalNonOrthogonalIter())
        {
            phi = phiHbyA + pEqn.flux();
        }
    }
    double end_psolve = std::clock();
    p_solve  += double(end_psolve - start_psolve) / double(CLOCKS_PER_SEC);

}


bool limitedp = pressureControl.limit(p);

// Thermodynamic density update
double thermodensityus = std::clock();
thermo.correctRho(psi*p - psip0);
double thermodensityue = std::clock();
thermodensityu  += double(thermodensityue - thermodensityus) / double(CLOCKS_PER_SEC);


if (limitedp)
{
    rho = thermo.rho();
}

double rhoEqns = std::clock();
#include "rhoEqn.H"
#include "compressibleContinuityErrs.H"
double rhoEqne = std::clock();
rhoEqn  += double(rhoEqne -rhoEqns) / double(CLOCKS_PER_SEC);

// Explicitly relax pressure for momentum corrector
double start_prelax = std::clock();
p.relax();
double end_prelax = std::clock();
p_relax  += double(end_prelax - start_prelax) / double(CLOCKS_PER_SEC);

double updateUs = std::clock();
U = HbyA - rAU*fvc::grad(p);
U.correctBoundaryConditions();
K = 0.5*magSqr(U);
double updateUe = std::clock();
updateU  += double(updateUe - updateUs) / double(CLOCKS_PER_SEC);

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
//end_p = std::clock();
//p_eq +=  double(end_p - start_p) / double(CLOCKS_PER_SEC);
