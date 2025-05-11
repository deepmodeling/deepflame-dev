{
    volScalarField& he = thermo.he();


    tmp<fvScalarMatrix> thcSource(new fvScalarMatrix(he, dimEnergy/dimTime));
    fvScalarMatrix& hcSource = thcSource.ref();
    forAll(Y, i)
    {
        hcSource.source() -= parcels.rhoTrans(i)*chemistry->mixture().Hc(i)/runTime.deltaT();
    }
    //hSource.source() -= parcels.hsTrans()/runTime.deltaT();// equivalent to parcels.Sh(he)


    start1 = std::clock();
    fvScalarMatrix EEqn
    (

        fvm::ddt(rho, he) + mvConvection->fvmDiv(phi, he)
    +   fvc::ddt(rho, K) + fvc::div(phi, K)
    -   dpdt
    ==
    (
        turbName == "laminar"
        ?
        (
            fvm::laplacian(turbulence->alpha(), he)
        -   diffAlphaD
        +   fvc::div(hDiffCorrFlux)
        )
        :
        fvm::laplacian(turbulence->alphaEff(), he)
    )
    +   parcels.Sh(he)
    // +   rho*(U&g)
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

    end1 = std::clock();
    time_monitor_EEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_EEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);

    // EEqn.relax();
    start1 = std::clock();
    EEqn.solve("h");
    end1 = std::clock();
    time_monitor_EEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_EEqn_solve += double(end1 - start1) / double(CLOCKS_PER_SEC);
}
