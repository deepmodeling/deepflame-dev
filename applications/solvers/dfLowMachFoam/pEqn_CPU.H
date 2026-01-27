Info << "Enter pEqn_CPU" << endl;
assert(!pimple.transonic());

clockTime pEqnClock;

if (!pimple.simpleRho())
{
    rho = thermo.rho();
}

// Thermodynamic density needs to be updated by psi*d(p) after the
// pressure solution
const volScalarField psip0(psi*p);

surfaceScalarField rhorAUf
(
    IOobject
    (
        "rhorAUf",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar(dimensionSet(0,0,1,0,0,0,0), 0)
);

volVectorField HbyA
(
    IOobject
    (
        "HbyA",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedVector(dimensionSet(0,1,-1,0,0,0,0), Zero)
);

volScalarField rAU
(
    IOobject
    (
        "rAU",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar(dimensionSet(-1,3,1,0,0,0,0), 0)
);

#ifndef OPT_GenMatrix_p
rAU = (1.0/UEqn.A());
rhorAUf = fvc::interpolate(rho*rAU);
HbyA = constrainHbyA(rAU*UEqn.H(), U, p);
#else
getrAUandHbyA(rAU, HbyA, UEqn, U);
#endif

time_monitor_pEqn_pre += pEqnClock.timeIncrement();

#ifdef OPT_GenMatrix_p_check
tmp<volScalarField> trAUref = 1.0/UEqn.A();
volScalarField& rAUref = trAUref.ref();
tmp<volVectorField> tHbyAref = constrainHbyA(rAUref*UEqn.H(), U, p);
volVectorField& HbyAref = tHbyAref.ref();
tmp<surfaceScalarField> trhorAUfref = fvc::interpolate(rho*rAUref);
surfaceScalarField& rhorAUfref = trhorAUfref.ref();
check_field_boundary_equal(rAUref, rAU, "rAU");
check_field_boundary_equal(HbyAref, HbyA, "HbyA");
#endif

time_monitor_pEqn_pre_check += pEqnClock.timeIncrement();

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

// Update the pressure BCs to ensure flux consistency
constrainPressure(p, rho, U, phiHbyA, rhorAUf);

time_monitor_pEqn_pre += pEqnClock.timeIncrement();

while (pimple.correctNonOrthogonal())
{
    #ifndef OPT_GenMatrix_p
    fvScalarMatrix pEqn
    (
        fvc::ddt(rho) + 
        psi*correction(fvm::ddt(p))
        + fvc::div(phiHbyA)
        - fvm::laplacian(rhorAUf, p)
    );
    #else
        tmp<fvScalarMatrix> tpEqn = GenMatrix_p(rho, p, phi, rAU, U, HbyA, psi, rhorAUf, phiHbyA);
        fvScalarMatrix pEqn = tpEqn.ref();
    #endif

    time_monitor_pEqn_pre += pEqnClock.timeIncrement();

    #ifdef OPT_GenMatrix_p_check
    fvScalarMatrix pEqn_answer
    (
        fvc::ddt(rho) + psi * correction(fvm::ddt(p))
      + fvc::div(phiHbyA)
      - fvm::laplacian(rhorAUfref, p)
    );
    check_fvmatrix_equal(pEqn, pEqn_answer, "pEqn");
    #endif

    time_monitor_pEqn_pre_check += pEqnClock.timeIncrement();

    #ifdef USE_DF_MATRIX
    // dfMatrix df_pEqn(pEqn, regionPtr);
    auto& psi = const_cast<volScalarField&>(pEqn.psi());
    auto& source = pEqn.source();
    auto& internalCoeffs = pEqn.internalCoeffs();
    auto& boundaryCoeffs = pEqn.boundaryCoeffs();

    matrixPtr->valueCopy(pEqn);

    time_monitor_pEqn_convert += pEqnClock.timeIncrement();

    // df_pEqn.solve(psi, source, internalCoeffs, boundaryCoeffs);
    matrixPtr->solve(psi, source, internalCoeffs, boundaryCoeffs);
    #else
    pEqn.solve();
    #endif

    time_monitor_pEqn_solve += pEqnClock.timeIncrement(); 

    #ifndef OPT_GenMatrix_p
    if (pimple.finalNonOrthogonalIter())
    {
        phi = phiHbyA + pEqn.flux();
    }
    #else
        postProcess_P(p, pEqn, phi, U, rAU, HbyA, K, dpdt, phiHbyA);
    #endif

    time_monitor_pEqn_post += pEqnClock.timeIncrement();
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

time_monitor_pEqn_post += pEqnClock.timeIncrement();
time_monitor_pEqn += pEqnClock.elapsedTime(); 
Info << "Exit pEqn_CPU" << endl;