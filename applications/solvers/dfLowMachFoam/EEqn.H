Info << "Enter EEqn" << endl;

{   
    clockTime EEqnClock;

    volScalarField& he = thermo.he();

    // volScalarField turbulence_alphaEff
    // (
    //     IOobject
    //     (
    //         "turbulence_alphaEff",
    //         runTime.timeName(),
    //         mesh,
    //         IOobject::NO_READ,
    //         IOobject::NO_WRITE
    //     ),
    //     mesh,
    //     dimensionedScalar(dimensionSet(1,-1,-1,0,0,0,0), 0)
    // );

    const auto& turbulence_alphaEff = turbulence->alphaEff();

#ifndef OPT_GenMatrix_E
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
#else
    tmp<fvScalarMatrix> tEEqn = GenMatrix_E(rho, he, phi, K, dpdt, turbulence_alphaEff, diffAlphaD, hDiffCorrFlux, linear_weights);
    fvScalarMatrix& EEqn = tEEqn.ref();
#endif

time_monitor_EEqn_pre += EEqnClock.timeIncrement();

#ifdef OPT_GenMatrix_E_check
    fvScalarMatrix EEqn_answer
    (
        fvm::ddt(rho, he) 
        + fvm::div(phi, he)
        + fvc::ddt(rho, K) 
        + fvc::div(phi, K)
        - dpdt
        - fvm::laplacian(turbulence->alphaEff(), he)
        + diffAlphaD
         ==
            fvc::div(hDiffCorrFlux)
    );
    check_fvmatrix_equal(EEqn, EEqn_answer, "EEqn");

    time_monitor_EEqn_pre_check += EEqnClock.timeIncrement();
#endif



#ifdef USE_DF_MATRIX
    // dfMatrix df_EEqn(EEqn, regionPtr);
    auto& psi = const_cast<volScalarField&>(EEqn.psi());
    auto& source = EEqn.source();
    auto& internalCoeffs = EEqn.internalCoeffs();
    auto& boundaryCoeffs = EEqn.boundaryCoeffs();

    matrixPtr->valueCopy(EEqn);
    time_monitor_EEqn_convert += EEqnClock.timeIncrement();

    // df_EEqn.solve(psi, source, internalCoeffs, boundaryCoeffs, "ha");
    matrixPtr->solve(psi, source, internalCoeffs, boundaryCoeffs, "ha");
#else
    EEqn.solve("ha");
#endif

    time_monitor_EEqn_solve += EEqnClock.timeIncrement();

    time_monitor_EEqn += EEqnClock.elapsedTime();
}

Info << "Exit EEqn" << endl;