if ((ddtSchemes == "RK2SSP") || (ddtSchemes == "RK3SSP"))
{
    rhoU_rhs = -fvc::div(phiUp);

    rhoU =  rkcoe1[nrk]*rhoU_save
          + rkcoe2[nrk]*rhoU
          + rkcoe3[nrk]*rhoU_rhs*runTime.deltaT();
    
    U.ref() = rhoU()/rho();
    U.correctBoundaryConditions();
    rhoU.boundaryFieldRef() == rho.boundaryField()*U.boundaryField();

    if (!inviscid)
    {
        rhoU_rhs = fvc::laplacian(turbulence->muEff(),U) + fvc::div(tauMC);
        rhoU += rkcoe3[nrk]*rhoU_rhs*runTime.deltaT();
        
        U.ref() = rhoU()/rho();
        U.correctBoundaryConditions();
        rhoU.boundaryFieldRef() == rho.boundaryField()*U.boundaryField();
    }
              
    Info <<"in rk"<<nrk+1<< ", min / max mag(U) ; " << gMin(mag(U)()()) << " / " << gMax(mag(U)()()) << nl << endl;
}
else
{
    solve
    (
        fvm::ddt(rhoU)
      + fvc::div(phiUp)
    );
    U.ref() = rhoU()/rho();
    U.correctBoundaryConditions();
    rhoU.boundaryFieldRef() == rho.boundaryField()*U.boundaryField();

    if (!inviscid)
    {
        solve
        (
            fvm::ddt(rho, U) - fvc::ddt(rho, U)
          - fvm::laplacian(turbulence->muEff(), U)
          - fvc::div(tauMC)
        );
        rhoU = rho*U;
    }

    Info << "\nmin / max mag(U) ; " << gMin(mag(U)()()) << " / " << gMax(mag(U)()()) << nl << endl;
}


