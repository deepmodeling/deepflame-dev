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

start1 = std::clock();
forAll(Y, i)
{
    sumYDiffError += chemistry->rhoD(i)*fvc::grad(Y[i]);
}
const surfaceScalarField phiUc = linearInterpolate(sumYDiffError) & mesh.Sf();
start1 = std::clock();
time_monitor_YEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
time_monitor_YEqn_solve += double(end1 - start1) / double(CLOCKS_PER_SEC);  

//MPI_Barrier(PstreamGlobals::MPI_COMM_FOAM);
label flag_mpi_init;
MPI_Initialized(&flag_mpi_init);
if(flag_mpi_init) MPI_Barrier(PstreamGlobals::MPI_COMM_FOAM);

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

    start2 = std::clock();
    volScalarField Yt(0.0*Y[0]);
    forAll(Y, i)
    {
        volScalarField& Yi = Y[i];
        hDiffCorrFlux += chemistry->hai(i)*(chemistry->rhoD(i)*fvc::grad(Yi) - Yi*sumYDiffError);
        diffAlphaD += fvc::laplacian(thermo.alpha()*chemistry->hai(i), Yi);
        if (i != inertIndex)
        {
            start1 = std::clock();
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
            
            end1 = std::clock();
            time_monitor_YEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);
            YiEqn.relax();

            start1 = std::clock();
            YiEqn.solve("Yi");
            end1 = std::clock();
            time_monitor_YEqn_solve += double(end1 - start1) / double(CLOCKS_PER_SEC);

            Yi.max(0.0);
            Yt += Yi;
        }
    }

    Y[inertIndex] = scalar(1) - Yt;
    Y[inertIndex].max(0.0);
    end2 = std::clock();
    time_monitor_YEqn += double(end2 - start2) / double(CLOCKS_PER_SEC);
}
