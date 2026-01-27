if ((ddtSchemes == "RK2SSP") || (ddtSchemes == "RK3SSP"))
{
    rho_rhs = -fvc::div(phi);

    rho =   rkcoe1[nrk]*rho_save
          + rkcoe2[nrk]*rho 
          + rkcoe3[nrk]*rho_rhs*runTime.deltaT();

    Info <<"in rk"<< nrk+1 << " finish calculate rho" << nl << endl;
}
else
{
    solve
    (
        fvm::ddt(rho)
      + fvc::div(phi)
    );
}


