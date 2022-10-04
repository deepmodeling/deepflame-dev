{
    volScalarField& he = thermo.he();


    tmp<fvScalarMatrix> thcSource(new fvScalarMatrix(he, dimEnergy/dimTime));
    fvScalarMatrix& hcSource = thcSource.ref();
    forAll(Y, i)
    {
        hcSource.source() -= parcels.rhoTrans(i)*chemistry->mixture().Hc(i)/runTime.deltaT();
    }
    //hSource.source() -= parcels.hsTrans()/runTime.deltaT();// equivalent to parcels.Sh(he)
    fvScalarMatrix EEqn
    (
        fvm::ddt(rho, he) + mvConvection->fvmDiv(phi, he)
      + fvc::ddt(rho, K) + fvc::div(phi, K)
      - dpdt
      - fvm::laplacian(turbulence->alphaEff(), he)
      + diffAlphaD
     ==
        rho*(U&g)
      + parcels.Sh(he)
      + hcSource
      + fvc::div(hDiffCorrFlux)
      //+ radiation->Sh(thermo, he)
      //+ fvOptions(rho, he)
    );

    EEqn.relax();

    //fvOptions.constrain(EEqn);

    EEqn.solve();

    //fvOptions.correct(he);

    //radiation->correct();
}
