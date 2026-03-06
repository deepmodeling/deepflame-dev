/*---------------------------------------------------------------------------*\
    highSpeed/rhoUEqn.H — Momentum equation for highSpeed region (Euler branch)
\*---------------------------------------------------------------------------*/

MRF.correctBoundaryVelocity(U);

solve
(
    fvm::ddt(rhoU)
  + fvc::div(rhoUPhi)
  + MRF.DDt(rho, U)
);

U.ref() = rhoU()/rho();
U.correctBoundaryConditions();
rhoU.boundaryFieldRef() == rho.boundaryField()*U.boundaryField();

if (!inviscid)
{
    solve
    (
        fvm::ddt(rho, U) - fvc::ddt(rho, U)
      - fvm::laplacian(turbulence.muEff(), U)
      - fvc::div(tauMC)
    );
    rhoU = rho*U;
}

Info<< "min/max mag(U) = " << gMin(mag(U)()()) << " / "
    << gMax(mag(U)()()) << endl;
