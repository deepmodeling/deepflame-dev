{
    volScalarField& he = thermo.he();
    double startEE_cons = std::clock();
    fvScalarMatrix EEqn
    (
        fvm::ddt(rho, he) + fvm::div(phi, he)
      + fvc::ddt(rho, K) + fvc::div(phi, K)
      - dpdt
      - fvm::laplacian(turbulence->alphaEff(), he)
      + diffAlphaD
     ==
        fvc::div(hDiffCorrFlux)
    );
    
    double endEE_cons = std::clock();
    EE_cons += double(endEE_cons - startEE_cons)/double(CLOCKS_PER_SEC);

    MPI_Barrier(PstreamGlobals::MPI_COMM_FOAM);    
    
    double startEE_r = std::clock();
    EEqn.relax();
    double endEE_r = std::clock();
    EE_relax += double(endEE_r - startEE_r)/double(CLOCKS_PER_SEC);   

    MPI_Barrier(PstreamGlobals::MPI_COMM_FOAM); 
    
    double startEE = std::clock();
    //double time_amgx_solve = EEqn.solver().time_amgx();
    EEqn.solve();
    double endEE = std::clock();
    EE_solve += double(endEE - startEE)/double(CLOCKS_PER_SEC);
}
