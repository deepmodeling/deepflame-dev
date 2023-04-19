{
    volScalarField& he = thermo.he();

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
            fvc::div(hDiffCorrFlux)
        )
        :
        (
            fvm::ddt(rho, he) + mvConvection->fvmDiv(phi, he)
        +   fvc::ddt(rho, K) + fvc::div(phi, K)
        -   dpdt
        -   fvm::laplacian(turbulence->alphaEff(), he)
        )
        );

        EEqn.relax();

        EEqn.solve();
    
    
}
