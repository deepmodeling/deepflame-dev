if (!inviscid)
{
    hDiffCorrFlux = Zero;
    diffAlphaD = Zero;
    sumYDiffError = Zero;
    
    forAll(Y, i)
    {
        sumYDiffError += chemistry->rhoD(i)*fvc::grad(Y[i]);
    }
}

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

{
    start = std::clock();
    combustion->correct();
    label flag_mpi_init;
    MPI_Initialized(&flag_mpi_init);
    if(flag_mpi_init) MPI_Barrier(PstreamGlobals::MPI_COMM_FOAM);
    end = std::clock();
    time_monitor_chem += double(end - start) / double(CLOCKS_PER_SEC);

    volScalarField Yt(0.0*Y[0]);

    start = std::clock();
    forAll(Y, i)
    {
        volScalarField& Yi = Y[i];

        if (!inviscid)
        {
            hDiffCorrFlux += chemistry->hai(i)*(chemistry->rhoD(i)*fvc::grad(Yi) - Yi*sumYDiffError);
            diffAlphaD += fvc::laplacian(thermo.alpha()*chemistry->hai(i), Yi);
        }

        if (i != inertIndex)
        {          
            // original convection term
            // fvScalarMatrix YiEqn
            // (
            //      fvm::ddt(rho, Yi)
            //    + mvConvection->fvmDiv(phi, Yi)
            //   ==
            //     combustion->R(Yi)
            // );

            solve
            (
                  fvm::ddt(rhoYi[i])
                + fvc::div(phiYi[i])
               == 
                  chemistry->RR(i)
            );

            Yi=rhoYi[i]/rho;
            Yi.max(0.0);

            if (!inviscid)
            {
                const surfaceScalarField phiUc = linearInterpolate(sumYDiffError) & mesh.Sf();
                tmp<volScalarField> DEff = chemistry->rhoD(i) + turbulence->mut()/Sct;

                // original term
                // YiEqn -= fvm::laplacian(DEff(), Yi) - mvConvection->fvmDiv(phiUc, Yi);

                fvScalarMatrix YiEqn
                (
                    fvm::ddt(rho, Yi) - fvc::ddt(rho, Yi)
                  - fvm::laplacian(DEff(), Yi)
                  + mvConvection->fvmDiv(phiUc, Yi)
                );

                YiEqn.relax();

                YiEqn.solve("Yi");

                Yi.max(0.0);

            }

            // original term
            // YiEqn.relax();
            // YiEqn.solve("Yi");
            // Yi.max(0.0);

            rhoYi[i] = rho*Yi;
            Yt += Yi;
        }
    }

    Y[inertIndex] = scalar(1) - Yt;
    Y[inertIndex].max(0.0);

    end = std::clock();
    time_monitor_Y += double(end - start) / double(CLOCKS_PER_SEC);
}
