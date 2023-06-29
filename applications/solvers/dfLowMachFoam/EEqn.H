{
    volScalarField& he = thermo.he();
    
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
                (
                    fvm::laplacian(turbulence->alphaEff(), he)
                )
            )
        );
    end1 = std::clock();
    time_monitor_EEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_EEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);

    EEqn.relax();
    start1 = std::clock();
    EEqn.solve("ha");
    end1 = std::clock();
    time_monitor_EEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_EEqn_solve += double(end1 - start1) / double(CLOCKS_PER_SEC);
}
