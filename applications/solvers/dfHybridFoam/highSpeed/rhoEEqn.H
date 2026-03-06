/*---------------------------------------------------------------------------*\
    highSpeed/rhoEEqn.H — Energy equation for highSpeed region (Euler branch)

    Mirrors dfHighSpeedFoam/rhoEEqn.H (Euler branch) for the multi-region
    context. Uses central-upwind weighted face velocity in sigmaDotU.
\*---------------------------------------------------------------------------*/

surfaceScalarField sigmaDotU
(
    "sigmaDotU",
    (
        fvc::interpolate(turbulence.muEff())*mesh.magSf()*fvc::snGrad(U)
      + (mesh.Sf() & fvc::interpolate(tauMC))
    )
    & (a_pos*U_pos + a_neg*U_neg)
);

solve
(
    fvm::ddt(rhoE)
  + fvc::div(rhoEPhi)
  - fvc::div(sigmaDotU)
);

ea = rhoE/rho - 0.5*magSqr(U);
ea.correctBoundaryConditions();

chemistry->correctThermo();

rhoE.boundaryFieldRef() == rho.boundaryField()
  * (ea.boundaryField() + 0.5*magSqr(U.boundaryField()));

if (!inviscid)
{
    fvScalarMatrix eEqn
    (
        fvm::ddt(rho, ea) - fvc::ddt(rho, ea)
      ==
        (
            turbName == "laminar"
            ? (
                  fvm::laplacian(turbulence.alphaEff()*thermo.gamma(), ea)
                - diffAlphaD
                + fvc::div(hDiffCorrFlux)
              )
            : fvm::laplacian(turbulence.alphaEff()*thermo.gamma(), ea)
        )
    );

    eEqn.solve("ea");

    chemistry->correctThermo();
    rhoE = rho*(ea + 0.5*magSqr(U));
    rhoE.boundaryFieldRef() == rho.boundaryField()
      * (ea.boundaryField() + 0.5*magSqr(U.boundaryField()));
}

Info<< "min/max(T) = " << min(T).value() << ", "
    << max(T).value() << endl;

p.ref() = rho()/psi();
p.correctBoundaryConditions();
rho.boundaryFieldRef() == psi.boundaryField()*p.boundaryField();
