start = std::clock();

hDiffCorrFlux = Zero;
diffAlphaD = Zero;
sumYDiffError = Zero;

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

forAll(Y, i)
{
    sumYDiffError += chemistry->rhoD(i)*fvc::grad(Y[i]);
}
const surfaceScalarField phiUc = linearInterpolate(sumYDiffError) & mesh.Sf();

//MPI_Barrier(PstreamGlobals::MPI_COMM_FOAM);
label flag_mpi_init;
MPI_Initialized(&flag_mpi_init);
if(flag_mpi_init) MPI_Barrier(PstreamGlobals::MPI_COMM_FOAM);
end = std::clock();
time_monitor_corrDiff += double(end - start) / double(CLOCKS_PER_SEC);

{
    if (!splitting)
    {
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        combustion->correct();
        //label flag_mpi_init;
        //MPI_Initialized(&flag_mpi_init);
        if(flag_mpi_init) MPI_Barrier(PstreamGlobals::MPI_COMM_FOAM);
        std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();
        std::chrono::duration<double> processingTime = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
        time_monitor_chem += processingTime.count();
    }

    volScalarField Yt(0.0*Y[0]);

    start = std::clock();
    forAll(Y, i)
    {
        volScalarField& Yi = Y[i];
        hDiffCorrFlux += chemistry->hai(i)*(chemistry->rhoD(i)*fvc::grad(Yi) - Yi*sumYDiffError);
        diffAlphaD += fvc::laplacian(thermo.alpha()*chemistry->hai(i), Yi);

        if (i != inertIndex)
        {
            tmp<volScalarField> DEff = chemistry->rhoD(i) + turbulence->mut()/Sct;
            fvScalarMatrix YiEqn
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

            YiEqn.relax();

            YiEqn.solve("Yi");

            Yi.max(0.0);
            Yt += Yi;
        }
    }

    Y[inertIndex] = scalar(1) - Yt;
    Y[inertIndex].max(0.0);

    end = std::clock();
    time_monitor_Y += double(end - start) / double(CLOCKS_PER_SEC);
}
