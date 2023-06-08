surfaceScalarField sigmaDotU
(
    "sigmaDotU",
    (
        fvc::interpolate(turbulence->muEff())*mesh.magSf()*fvc::snGrad(U)
      + (mesh.Sf() & fvc::interpolate(tauMC))
    )
    & (a_pos*U_pos + a_neg*U_neg)
);

solve
(
      fvm::ddt(rhoE)
    + fvc::div(phiEp)
    - fvc::div(sigmaDotU)
);

ea = rhoE/rho - 0.5*magSqr(U);
ea.correctBoundaryConditions();

ha = ea + p/rho;
chemistry->correctThermo(); // before this, we must update ha = e + p/rho

rhoE.boundaryFieldRef() == rho.boundaryField()*(ea.boundaryField() + 0.5*magSqr(U.boundaryField()));

if (!inviscid)
{
    fvScalarMatrix eEqn
    (
           fvm::ddt(rho, ea) - fvc::ddt(rho, ea)
        // alpha in deepflame is considered to calculate h by default (kappa/Cp), so multiply gamma to correct alpha
         - fvm::laplacian(turbulence->alphaEff()*thermo.gamma(), ea)
         + diffAlphaD
      ==
           fvc::div(hDiffCorrFlux)
    );

    eEqn.solve("ea");

    ha = ea + p/rho;
    chemistry->correctThermo();
    rhoE = rho*(ea + 0.5*magSqr(U));
}

Info<< "min/max(T) = "
        << min(T).value() << ", " << max(T).value() << endl;

p.ref() = rho()/psi();
p.correctBoundaryConditions();
rho.boundaryFieldRef() == psi.boundaryField()*p.boundaryField();
