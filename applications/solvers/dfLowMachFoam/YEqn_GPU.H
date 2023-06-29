start1 = std::clock();
UEqn_GPU.solve();
end1 = std::clock();
time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
time_monitor_UEqn_solve += double(end1 - start1) / double(CLOCKS_PER_SEC);

start1 = std::clock();
std::vector<double*> Y_old(Y.size()), boundary_Y(Y.size()), boundary_hai(Y.size()), boundary_rhoD(Y.size());
std::vector<const double*> hai(Y.size()), rhoD(Y.size());
for (size_t i = 0; i < Y.size(); ++i)
{
    volScalarField& Yi = Y[i];
    Yi.oldTime();
    Y_old[i] = &Yi.oldTime()[0];
    if (updateBoundaryFields)
    {
        cudaMallocHost(&boundary_Y[i], num_boundary_faces*sizeof(double));
    }
    const volScalarField& haii = chemistry->hai(i);
    const volScalarField& rhoDi = chemistry->rhoD(i);
    // hai[i] = &haii[0];
    rhoD[i] = &rhoDi[0];
    // cudaMallocHost(&boundary_hai[i], num_boundary_faces*sizeof(double));
    cudaMallocHost(&boundary_rhoD[i], num_boundary_faces*sizeof(double));
    int offset = 0;
    forAll(Yi.boundaryField(), patchi)
    {
        const scalarField& patchYi = Yi.boundaryField()[patchi];
        // const scalarField& patchHaii = haii.boundaryField()[patchi];
        const scalarField& patchRhoDi = rhoDi.boundaryField()[patchi];
        int patchSize = patchYi.size();

        if (updateBoundaryFields)
        {
            memcpy(boundary_Y[i] + offset, &patchYi[0], patchSize*sizeof(double));
        }
        // memcpy(boundary_hai[i] + offset, &patchHaii[0], patchSize*sizeof(double));
        memcpy(boundary_rhoD[i] + offset, &patchRhoDi[0], patchSize*sizeof(double));
        offset += patchSize;
    }
    // if (i == 5)
    // {
    //     Info << "rhoD_CPU" << rhoDi << endl;
    // }
    
}
// Info << "rhoD from nuEff\n" << nuEff * rho / 0.7 << endl;
updateBoundaryFields = false;
volScalarField mut_sct = turbulence->mut().ref()/Sct;
double *boundary_mutsct = nullptr;
cudaMallocHost(&boundary_mutsct, num_boundary_faces*sizeof(double));
int offset = 0;
forAll(p.boundaryField(), patchi)
{
    const scalarField& patchMut_sct = mut_sct.boundaryField()[patchi];
    int patchSize = patchMut_sct.size();
    memcpy(boundary_mutsct + offset, &patchMut_sct[0], patchSize*sizeof(double));
    offset += patchSize;

    // debug
    // const fvsPatchScalarField& pw = mesh.surfaceInterpolation::weights().boundaryField()[patchi];
    // Field<scalar> valueInternalCoeffs = Y[5].boundaryField()[patchi].valueInternalCoeffs(pw);
    // Field<scalar> valueBoundaryCoeffs = Y[5].boundaryField()[patchi].valueBoundaryCoeffs(pw);
    // Field<scalar> gradientInternalCoeffs = Y[5].boundaryField()[patchi].gradientInternalCoeffs();
    // Field<scalar> gradientBoundaryCoeffs = Y[5].boundaryField()[patchi].gradientBoundaryCoeffs();
    // Info << "valueInternalCoeffs\n" << valueInternalCoeffs << endl;
    // Info << "valueBoundaryCoeffs\n" << valueBoundaryCoeffs << endl;
    // Info << "gradientInternalCoeffs\n" << gradientInternalCoeffs << endl;
    // Info << "gradientBoundaryCoeffs\n" << gradientBoundaryCoeffs << endl;
}
end1 = std::clock();
time_monitor_YEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
time_monitor_YEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);
time_monitor_YEqn_mtxAssembly_CPU_prepare += double(end1 - start1) / double(CLOCKS_PER_SEC);
//fprintf(stderr, "time_monitor_YEqn_mtxAssembly_CPU_prepare: %lf\n", time_monitor_YEqn_mtxAssembly_CPU_prepare);

start1 = std::clock();
YEqn_GPU.initializeTimeStep();
YEqn_GPU.upwindWeight();
YEqn_GPU.fvm_laplacian_and_sumYDiffError_diffAlphaD_hDiffCorrFlux(Y_old, boundary_Y,
        hai, boundary_hai, rhoD, boundary_rhoD, &mut_sct[0], boundary_mutsct, &thermo.alpha()[0]);
YEqn_GPU.fvm_ddt();
YEqn_GPU.fvm_div_phi();
YEqn_GPU.fvm_div_phiUc();
YEqn_GPU.sync();
// YEqn_GPU.checkValue(true, "of_output_H2.txt");
end1 = std::clock();
time_monitor_YEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
time_monitor_YEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);
time_monitor_YEqn_mtxAssembly_GPU_run += double(end1 - start1) / double(CLOCKS_PER_SEC);
//fprintf(stderr, "time_monitor_YEqn_mtxAssembly_GPU_run: %lf\n", time_monitor_YEqn_mtxAssembly_GPU_run);

start1 = std::clock();
YEqn_GPU.solve();
end1 = std::clock();
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

    start1 = std::clock();
    forAll(Y, i)
    {
        volScalarField& Yi = Y[i];
        YEqn_GPU.updatePsi(&Yi[0], i);
        Yi.correctBoundaryConditions();
    }
    YEqn_GPU.correctBoundaryConditions();
    end1 = std::clock();
    time_monitor_YEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_YEqn_correctBC += double(end1 - start1) / double(CLOCKS_PER_SEC);
}
