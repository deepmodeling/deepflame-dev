double start_Yeq = std::clock();
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
        double cfs = std::clock();
        hDiffCorrFlux += chemistry->hai(i)*(chemistry->rhoD(i)*fvc::grad(Yi) - Yi*sumYDiffError);
        double cfe = std::clock();
        correctflux_t += double(cfe - cfs) / double(CLOCKS_PER_SEC);
        
        double lss = std::clock();
        diffAlphaD += fvc::laplacian(thermo.alpha()*chemistry->hai(i), Yi);
        double lse = std::clock();
        laplaciansolve += double(lse - lss) / double(CLOCKS_PER_SEC);
        if (i != inertIndex)
        {   
            double constructMs = std::clock();
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
            double constructMe = std::clock();
            constructM_Y  += double(constructMe - constructMs) / double(CLOCKS_PER_SEC);
            
            double startY_relax = std::clock();
            YiEqn.relax();
            double endY_relax = std::clock();
            Y_relax += double(endY_relax - startY_relax) / double(CLOCKS_PER_SEC);
            
            double start_Ysolve = std::clock();
            YiEqn.solve("Yi");
            //end = MPI_Wtime();
            double end_Ysolve = std::clock();
            Y_solve += double(end_Ysolve - start_Ysolve) / double(CLOCKS_PER_SEC);
            
            Yi.max(0.0);
            Yt += Yi;
        }

    }

    Y[inertIndex] = scalar(1) - Yt;
    Y[inertIndex].max(0.0);

    end = std::clock();
    
    time_monitor_Y += double(end-start)/double(CLOCKS_PER_SEC);
}
double end_Yeq = std::clock();
Yeq += double(end_Yeq-start_Yeq)/double(CLOCKS_PER_SEC);