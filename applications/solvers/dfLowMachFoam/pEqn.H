if (!pimple.SIMPLErho())
{
    rho = thermo.rho();
}

// Thermodynamic density needs to be updated by psi*d(p) after the
// pressure solution
const volScalarField psip0(psi*p);

#ifdef GPUSolver_
    // UEqn.H()
    start1 = std::clock();
    volVectorField UEqn_H
    (
        IOobject
        (
            "H("+U.name()+')',
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedVector(dimensionSet(1,-2,-2,0,0,0,0), Zero),
        extrapolatedCalculatedFvPatchScalarField::typeName
    );
    UEqn_GPU.H(&UEqn_H[0][0]);
    end1 = std::clock();
    time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_UEqn_H += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_UEqn_H_GPU_run += double(end1 - start1) / double(CLOCKS_PER_SEC);

    start1 = std::clock();
    UEqn_H.correctBoundaryConditions();
    end1 = std::clock();
    time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_UEqn_H += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_UEqn_H_correctBC += double(end1 - start1) / double(CLOCKS_PER_SEC);

    // UEqn.A()
    start1 = std::clock();
    volScalarField UEqn_A
    (
        IOobject
        (
            "A("+U.name()+')',
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedScalar(dimensionSet(1,-3,-1,0,0,0,0), Zero),
        extrapolatedCalculatedFvPatchScalarField::typeName
    );
    UEqn_GPU.A(&UEqn_A[0]);
    end1 = std::clock();
    time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_UEqn_A += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_UEqn_A_correctBC += double(end1 - start1) / double(CLOCKS_PER_SEC);

    start1 = std::clock();
    UEqn_A.correctBoundaryConditions();
    end1 = std::clock();
    time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_UEqn_A += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_UEqn_A_correctBC += double(end1 - start1) / double(CLOCKS_PER_SEC);
#endif

start2 = std::clock();
#ifdef GPUSolver_
    volScalarField rAU(1.0/UEqn_A);
    surfaceScalarField rhorAUf("rhorAUf", fvc::interpolate(rho*rAU));
    volVectorField HbyA(constrainHbyA(rAU*UEqn_H, U, p));
#else
    volScalarField rAU(1.0/UEqn.A());
    surfaceScalarField rhorAUf("rhorAUf", fvc::interpolate(rho*rAU));
    volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U, p));

    if (pimple.nCorrPISO() <= 1)
    {
        tUEqn.clear();
    }
#endif

surfaceScalarField phiHbyA
(
    "phiHbyA",
    fvc::interpolate(rho)*fvc::flux(HbyA)
  + rhorAUf*fvc::ddtCorr(rho, U, phi, rhoUf)
);

fvc::makeRelative(phiHbyA, rho, U);

// Update the pressure BCs to ensure flux consistency
constrainPressure(p, rho, U, phiHbyA, rhorAUf);

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

        start1 = std::clock();
        pEqn.solve();
        end1 = std::clock();
        time_monitor_pEqn_solve += double(end1 - start1) / double(CLOCKS_PER_SEC);

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
        fvc::ddt(rho) + psi*correction(fvm::ddt(p))
      + fvc::div(phiHbyA)
    );

    while (pimple.correctNonOrthogonal())
    {
        fvScalarMatrix pEqn(pDDtEqn - fvm::laplacian(rhorAUf, p));

        start1 = std::clock();
        pEqn.solve();
        end1 = std::clock();
        time_monitor_pEqn_solve += double(end1 - start1) / double(CLOCKS_PER_SEC);

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
p.relax();

U = HbyA - rAU*fvc::grad(p);
U.correctBoundaryConditions();
K = 0.5*magSqr(U);

#ifdef GPUSolver_
    start1 = std::clock();
    UEqn_GPU.correctPsi(&U[0][0]);
    end1 = std::clock();
    time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_UEqn_correctBC += double(end1 - start1) / double(CLOCKS_PER_SEC);
#endif

if (pimple.SIMPLErho())
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
end2 = std::clock();
time_monitor_pEqn += double(end2 - start2) / double(CLOCKS_PER_SEC);

