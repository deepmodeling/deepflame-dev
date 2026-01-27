Info << "Enter YEqn" << endl;

syncClockTime YEqnClock;

hDiffCorrFlux = Zero;
diffAlphaD = Zero;
sumYDiffError = Zero;
surfaceScalarField phiUc
(
    IOobject
    (
        "phiUc",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar(dimensionSet(1,0,-1,0,0,0,0), 0)
);
surfaceScalarField upwindWeights
(
    IOobject
    (
        "upwindWeights",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar(dimensionSet(0,0,0,0,0,0,0), 0)
);

#ifndef OPT_GenMatrix_Y
tmp<fv::convectionScheme<scalar>> mvConvection
(
    fv::convectionScheme<scalar>::New
    (
        mesh,
        fields,
        phi,
        mesh.divScheme("div(phi,Yi_h)")
    )
);
#endif

time_monitor_YEqn_pre += YEqnClock.timeIncrement();

{
    #ifdef OPT_GenMatrix_Y
    preProcess_Y(Y, phiUc, sumYDiffError, chemistry, upwindWeights, thermo.alpha(), phi);
    #else
    forAll(Y, i)
    {
        sumYDiffError += chemistry->rhoD(i) * fvc::grad(Y[i]);;
    }
    phiUc = (linearInterpolate(sumYDiffError) & mesh.Sf()).ref();
    #endif

    // forAll(Y, i)
    // {
    //     sumYDiffError += chemistry->rhoD(i)*fvc::grad(Y[i]);
    // }
    // Info << "sumYDiffError\n" << sumYDiffError << endl;
    // const surfaceScalarField phiUc = (linearInterpolate(sumYDiffError) & mesh.Sf()).ref();
    time_monitor_YEqn_pre += YEqnClock.timeIncrement();

    #ifdef OPT_GenMatrix_Y_check
    volVectorField sumYDiffErrorRef = sumYDiffError;
    sumYDiffErrorRef = Zero;
    forAll(Y, i)
    {
        sumYDiffErrorRef += chemistry->rhoD(i) * fvc::grad(Y[i]);;
    }
    check_field_boundary_equal(sumYDiffErrorRef, sumYDiffError, "sumYDiffError");
    const surfaceScalarField phiUcRef = (linearInterpolate(sumYDiffErrorRef) & mesh.Sf()).ref();
    check_field_boundary_equal(phiUcRef, phiUc, "phiUc");
    #endif

    time_monitor_YEqn_pre_check += YEqnClock.timeIncrement();

    if (!splitting)
    {
        combustion->correct();
    }

    time_monitor_chem = YEqnClock.timeIncrement();

    volScalarField Yt(0.0*Y[0]);
    forAll(Y, i)
    {
        volScalarField& Yi = Y[i];
        hDiffCorrFlux += chemistry->hai(i)*(chemistry->rhoD(i)*fvc::grad(Yi) - Yi*sumYDiffError);
        diffAlphaD += fvc::laplacian(thermo.alpha()*chemistry->hai(i), Yi);
        
        time_monitor_YEqn_pre += YEqnClock.timeIncrement();

        if (i != inertIndex)
        {
            #ifndef OPT_GenMatrix_Y
            tmp<volScalarField> DEff = chemistry->rhoD(i) + turbulence->mut()/Sct;
            fvScalarMatrix YiEqn
            (
                fvm::ddt(rho, Yi)
            +
                (
                    turbName == "laminar"
                    ?  (mvConvection->fvmDiv(phi, Yi) + mvConvection->fvmDiv(phiUc, Yi))
                    :   mvConvection->fvmDiv(phi, Yi)
                )
            ==
                (
                    splitting
                    ?   fvm::laplacian(DEff(), Yi)
                    :  (fvm::laplacian(DEff(), Yi) + combustion->R(Yi))
                    )
            );
            #else
            tmp<fvScalarMatrix> tYiEqn = GenMatrix_Y(rho, Yi, phi, phiUc, chemistry->rhoD(i), turbulence->mut(), splitting, Sct, combustion(), upwindWeights);
            fvScalarMatrix& YiEqn = tYiEqn.ref();
            #endif

            time_monitor_YEqn_pre += YEqnClock.timeIncrement();
            // YiEqn.relax();


            #ifdef OPT_GenMatrix_Y_check
                tmp<fv::convectionScheme<scalar>> mvConvection
                (
                    fv::convectionScheme<scalar>::New
                    (
                        mesh,
                        fields,
                        phi,
                        mesh.divScheme("div(phi,Yi_h)")
                    )
                );
                tmp<volScalarField> DEff = chemistry->rhoD(i);
                fvScalarMatrix YiEqn_answer
                (
                    fvm::ddt(rho, Yi)
                    + mvConvection->fvmDiv(phi, Yi)
                    + mvConvection->fvmDiv(phiUc, Yi)
                    ==
                    (
                        splitting
                    ?   fvm::laplacian(DEff(), Yi)
                    :  (fvm::laplacian(DEff(), Yi) + combustion->R(Yi))
                    )
                );
                check_fvmatrix_equal(YiEqn_answer, YiEqn, "YiEqn");
            #endif

            time_monitor_YEqn_pre_check += YEqnClock.timeIncrement();

#ifdef USE_DF_MATRIX
            // dfMatrix df_YiEqn(YiEqn, regionPtr);
            auto& psi = const_cast<volScalarField&>(YiEqn.psi());
            auto& source = YiEqn.source();
            auto& internalCoeffs = YiEqn.internalCoeffs();
            auto& boundaryCoeffs = YiEqn.boundaryCoeffs();

            matrixPtr->valueCopy(YiEqn);
            time_monitor_YEqn_convert += YEqnClock.timeIncrement();

            // df_YiEqn.solve(psi, source, internalCoeffs, boundaryCoeffs, "Yi");
            matrixPtr->solve(psi, source, internalCoeffs, boundaryCoeffs, "Yi");
#else
            YiEqn.solve("Yi");
#endif

            time_monitor_YEqn_solve += YEqnClock.timeIncrement();

            Yi.max(0.0);
            Yt += Yi;

            time_monitor_YEqn_post += YEqnClock.timeIncrement();
        }
    }

    Y[inertIndex] = scalar(1) - Yt;
    Y[inertIndex].max(0.0);
    
}

time_monitor_YEqn_post += YEqnClock.timeIncrement(); 
time_monitor_YEqn += YEqnClock.elapsedTime();

time_monitor_YEqn -= time_monitor_chem;

Info << "Exit YEqn" << endl;