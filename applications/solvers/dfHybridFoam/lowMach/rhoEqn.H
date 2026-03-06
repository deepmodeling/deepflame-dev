/*---------------------------------------------------------------------------*\
    lowMach/rhoEqn.H — Continuity equation for lowMach region
\*---------------------------------------------------------------------------*/
{
    fvScalarMatrix rhoEqn
    (
        fvm::ddt(rho)
      + fvc::div(phi)
    );

    rhoEqn.solve();
}
