Info << "Enter UEqn" << endl;
// Solve the Momentum equation
#ifdef GPUSolverNew_

#if defined DEBUG_
    // run CPU, for temp
    TICK_START;
    tmp<fvVectorMatrix> tUEqn
    (
        fvm::ddt(rho, U) 
        + 
        fvm::div(phi, U)
        +  
        turbulence->divDevRhoReff(U)
    );
    fvVectorMatrix& UEqn = tUEqn.ref();
    TICK_STOP(CPU assembly time);

    volTensorField gradU = fvc::grad(U);

    double *h_boundary_gradU = new double[dfDataBase.num_boundary_surfaces * 9];
    offset = 0;
    forAll(U.boundaryField(), patchi)
    {
        const fvPatchTensorField& patchGradU = gradU.boundaryField()[patchi];
        int patchsize = patchGradU.size();
        if (patchGradU.type() == "processor"
                || patchGradU.type() == "processorCyclic") {
            // print info
            if (dynamic_cast<const processorFvPatchField<tensor>&>(patchGradU).doTransform()) {
                Info << "gradU transform = true" << endl;
            } else {
                Info << "gradU transform = false" << endl;
            }
            Info << "rank = " << dynamic_cast<const processorFvPatchField<tensor>&>(patchGradU).rank() << endl;

            memcpy(h_boundary_gradU + 9*offset, &patchGradU[0][0], patchsize * 9 * sizeof(double));
            tensorField patchGradUInternal = 
                    dynamic_cast<const processorFvPatchField<tensor>&>(patchGradU).patchInternalField()();
            memcpy(h_boundary_gradU + 9*offset + patchsize * 9, &patchGradUInternal[0][0], patchsize * 9 * sizeof(double));
            offset += patchsize * 2;
        } else {
            memcpy(h_boundary_gradU + 9*offset, &patchGradU[0][0], patchsize * 9 * sizeof(double));
            offset += patchsize;
        }
    }
#endif

    // process
    TICK_START;
    UEqn_GPU.process();
    UEqn_GPU.sync();
    TICK_STOP(GPU process time);

    // postProcess
    // TICK_START;
    // UEqn_GPU.postProcess(h_u);
    // memcpy(&U[0][0], h_u, dfDataBase.cell_value_vec_bytes);
    // U.correctBoundaryConditions();
    // K = 0.5*magSqr(U);
    // DEBUG_TRACE;
    // TICK_STOP(post process time);

#if defined DEBUG_
    // UEqn.relax();
    TICK_START;
    solve(UEqn == -fvc::grad(p));
    K.oldTime();
    K = 0.5*magSqr(U);
    TICK_STOP(CPU solve time);
    // checkResult
    // TODO: for temp, now we compare ldu, finally we compare csr
    std::vector<double> h_internal_coeffs(dfDataBase.num_boundary_surfaces * 3);
    std::vector<double> h_boundary_coeffs(dfDataBase.num_boundary_surfaces * 3);

    offset = 0;
    for (int patchi = 0; patchi < dfDataBase.num_patches; patchi++)
    {
        const fvPatchVectorField& patchU = U.boundaryField()[patchi];
        int patchsize = dfDataBase.patch_size[patchi];
        const double* internal_coeff_ptr = &UEqn.internalCoeffs()[patchi][0][0];
        const double* boundary_coeff_ptr = &UEqn.boundaryCoeffs()[patchi][0][0];
        memcpy(h_internal_coeffs.data() + offset * 3, internal_coeff_ptr, patchsize * 3 * sizeof(double));
        memcpy(h_boundary_coeffs.data() + offset * 3, boundary_coeff_ptr, patchsize * 3 * sizeof(double));
        if (patchU.type() == "processor" || patchU.type() == "processorCyclic") offset += 2 * patchsize;
        else offset += patchsize;
    }

    double *h_boundary_u_tmp = new double[dfDataBase.num_boundary_surfaces * 3];
    offset = 0;
    forAll(U.boundaryField(), patchi)
    {
        const fvPatchVectorField& patchU = U.boundaryField()[patchi];
        int patchsize = dfDataBase.patch_size[patchi];

        if (patchU.type() == "processor"
            || patchU.type() == "processorCyclic") {
            memcpy(h_boundary_u_tmp + 3*offset, &patchU[0][0], 3*patchsize * sizeof(double));
            vectorField patchUInternal = 
                    dynamic_cast<const processorFvPatchField<vector>&>(patchU).patchInternalField()();
            memcpy(h_boundary_u_tmp + 3*offset + 3*patchsize, &patchUInternal[0][0], 3*patchsize * sizeof(double));
            offset += 2 * patchsize;
        } else {
            memcpy(h_boundary_u_tmp + 3*offset, &patchU[0][0], 3*patchsize * sizeof(double));
            offset += patchsize;
        }
    }

    bool printFlag = false;

    int rank = -1;
    if (mpi_init_flag) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
    
    if (!mpi_init_flag || rank == 0) {
        // UEqn_GPU.compareResult(&UEqn.lower()[0], &UEqn.upper()[0], &UEqn.diag()[0], &UEqn.source()[0][0],
        //     h_internal_coeffs.data(), h_boundary_coeffs.data(), 
        //     // &gradU[0][0], h_boundary_gradU,
        //     printFlag);
        // UEqn_GPU.compareU(&U[0][0], h_boundary_u_tmp, printFlag);
    }
    DEBUG_TRACE;
#endif

#else
    start1 = MPI_Wtime();
    tmp<fvVectorMatrix> tUEqn
    (
        fvm::ddt(rho, U) + fvm::div(phi, U)
    + turbulence->divDevRhoReff(U) 
    );
    fvVectorMatrix& UEqn = tUEqn.ref();
    end1 = MPI_Wtime();
    time_monitor_UEqn += double(end1 - start1);
    time_monitor_UEqn_mtxAssembly += double(end1 - start1);

    // UEqn.relax();
    start1 = MPI_Wtime();
    if (pimple.momentumPredictor())
    {   
        // solve(UEqn == -fvc::grad(p));
        auto tUEqnSolve = (UEqn == -fvc::grad(p));
        auto& UEqnSolve = tUEqnSolve.ref();
        // tmp<fvVectorMatrix> tUEqnSolve
        // (
        //     UEqn == -fvc::grad(p)
        // );
        // fvVectorMatrix& UEqnSolve = tUEqnSolve.ref();
#ifdef USE_DF_MATRIX
        dfMatrix df_UEqnSolve(tUEqnSolve.ref(), regionPtr);
        auto& psi = const_cast<volVectorField&>(UEqnSolve.psi());
        auto& source = UEqnSolve.source();
        auto& internalCoeffs = UEqnSolve.internalCoeffs();
        auto& boundaryCoeffs = UEqnSolve.boundaryCoeffs();
        df_UEqnSolve.solve(psi, source, internalCoeffs, boundaryCoeffs);
#else
        // UEqnSolve.solve();
        UEqnSolve.solve();
#endif
        K.oldTime();
        K = 0.5*magSqr(U);
    }
    end1 = MPI_Wtime();
    time_monitor_UEqn += double(end1 - start1);
    time_monitor_UEqn_solve += double(end1 - start1);
#endif

Info << "Exit UEqn" << endl;