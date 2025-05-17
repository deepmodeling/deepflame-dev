     // should only for CPUSolver
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

    // auto& mgcs = dynamic_cast<fv::multivariateGaussConvectionScheme<scalar>&>(mvConvection.ref());
    // tmp<surfaceInterpolationScheme<scalar>> tinterpScheme_ = mgcs.interpolationScheme()()(Y[0]);
    // tmp<surfaceScalarField> tweights = tinterpScheme_().weights(Y[0]);
    // const surfaceScalarField& weights = tweights();
    // Info << "CPU weights\n" << weights << endl;

    start1 = std::clock();
    forAll(Y, i)
    {
        sumYDiffError += chemistry->rhoD(i)*fvc::grad(Y[i]);
    }
    // Info << "sumYDiffError\n" << sumYDiffError << endl;
    const surfaceScalarField phiUc = linearInterpolate(sumYDiffError) & mesh.Sf();
    start1 = std::clock();
    time_monitor_YEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_YEqn_solve += double(end1 - start1) / double(CLOCKS_PER_SEC);



    label flag_mpi_init;
    MPI_Initialized(&flag_mpi_init);
    if(flag_mpi_init) MPI_Barrier(PstreamGlobals::MPI_COMM_FOAM);

    {
         if (!splitting)
    {
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

        #ifdef ODE_GPU_SOLVER
	scalar dt = runTime.deltaTValue();

        forAll(Y, speciesI) {
            volScalarField& Yi = Y[speciesI];
            memcpy(h_y + speciesI * num_cells, &Yi[0], num_cells * sizeof(double));
        }

        for (int i = 0; i < num_cells; i++) {
            for (int j = 0; j < sp_num; j++) {
                h_y_t[j + i*sp_num] = h_y[i + j*num_cells];
            }
        }

        opencc_ode_all(&T[0], &p[0], h_y_t, 1e-10, dt, CPU);

        for (int i = 0; i < num_cells; i++) {
            for (int j = 0; j < sp_num; j++) {
                Ynew[i + j*num_cells] = h_y_t[j + i*sp_num];
            }
        }

        QdotGPU = Zero;
        forAll(QdotGPU,celli)
        {
            for (int sp = 0; sp < sp_num; sp++)
            {
                RRGPU[sp][celli] = (Ynew[sp*num_cells+celli]-Y[sp][celli])*rho[celli]/dt;
            }
        }
        #else
        combustion->correct();
        #endif
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
        hDiffCorrFlux += chemistry->hei(i)*(chemistry->rhoD(i)*fvc::grad(Yi) - Yi*sumYDiffError);
        diffAlphaD += fvc::laplacian(thermo.alpha()*chemistry->hei(i), Yi);
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
                    ?  (fvm::laplacian(DEff(), Yi) + parcels.SYi(i, Yi) )
                    :  (fvm::laplacian(DEff(), Yi) + parcels.SYi(i, Yi) + combustion->R(Yi))
                    )
            );

            end1 = std::clock();
            time_monitor_YEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);
            // YiEqn.relax();

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
