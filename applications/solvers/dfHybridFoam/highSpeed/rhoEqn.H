/*---------------------------------------------------------------------------*\
    highSpeed/rhoEqn.H — Continuity equation for highSpeed region (Euler branch)
\*---------------------------------------------------------------------------*/

solve
(
    fvm::ddt(rho)
  + fvc::div(rhoPhi)
);
