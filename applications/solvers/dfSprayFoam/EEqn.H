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
        turbName == "laminar"
        ?
        (
        fvm::ddt(rho, he) + mvConvection->fvmDiv(phi, he)
    +   fvc::ddt(rho, K) + fvc::div(phi, K)
    -   dpdt
    -   fvm::laplacian(turbulence->alphaEff(), he)
    +   diffAlphaD
    ==
       rho*(U&g)
    +   parcels.Sh(he)
    +   fvc::div(hDiffCorrFlux)
    //+ radiation->Sh(thermo, he)
    //+ fvOptions(rho, he)
        )
        :
        (
        fvm::ddt(rho, he) + mvConvection->fvmDiv(phi, he)
    +   fvc::ddt(rho, K) + fvc::div(phi, K)
    -   dpdt
    -   fvm::laplacian(turbulence->alphaEff(), he)
    ==
        rho*(U&g)
    +   parcels.Sh(he)
    //+ radiation->Sh(thermo, he)
    //+ fvOptions(rho, he)
        )
    );

    if(CanteraMixture::getEnergyName() == word("hs"))
    {
        EEqn -= combustion->Qdot();
    }
    else if(CanteraMixture::getEnergyName() == word("ha"))
    {
        EEqn -= hcSource;
    }
    else
    {
        FatalErrorInFunction
            << "Wrong CanteraMixture enregy name: " << CanteraMixture::getEnergyName()
            << abort(FatalError);
    }

    EEqn.relax();

    //fvOptions.constrain(EEqn);

    EEqn.solve("h");

    //fvOptions.correct(he);

    //radiation->correct();

}
