/*---------------------------------------------------------------------------*\
    lowMach/UEqn.H — Momentum equation for lowMach regions
\*---------------------------------------------------------------------------*/

// Solve the Momentum equation
tmp<fvVectorMatrix> tUEqn
(
    fvm::ddt(rho, U) + fvm::div(phi, U)
  + turbulence.divDevRhoReff(U)
);
fvVectorMatrix& UEqn = tUEqn.ref();

if (pimple.momentumPredictor())
{
    solve(UEqn == -fvc::grad(p));
    K.oldTime();
    K = 0.5*magSqr(U);
}
