/* FOR DEEPFLAME-ACADEMIC */
#ifdef GPUSolverNew_
    process_equation(MATRIX_EQUATION::UEqn);
#endif

// Solve the Momentum equation
#ifdef GPUSolverNew_

// tmp<fvVectorMatrix> tUEqn0
// (
//     fvm::ddt(rho, U) 
//     + 
//     fvm::div(phi, U)
//     +  
//     turbulence->divDevRhoReff(U)
//     == -fvc::grad(p)
// );
// fvVectorMatrix& UEqn0 = tUEqn0.ref();
// csrMatrix csr0(UEqn0);
// auto& psi0 = const_cast<volVectorField&>(UEqn0.psi());
// auto& source0 = UEqn0.source();
// auto& internalCoeffs0 = UEqn0.internalCoeffs();
// auto& boundaryCoeffs0 = UEqn0.boundaryCoeffs();
// csr0.solve(psi0, source0, internalCoeffs0, boundaryCoeffs0);

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

    // add for test
    double *diag = &UEqn.diag()[0];
    double *lower = &UEqn.lower()[0];
    double *upper = &UEqn.upper()[0];

    double *source_newOrder = new double[mesh_paras.num_cells*3]();
    permuteVector(source_newOrder, &UEqn.source()[0][0], mesh_paras.num_cells);

    writeDoubleArrayToFile(diag, mesh_paras.num_cells, "U_diag.host");
    writeDoubleArrayToFile(lower, mesh_paras.num_surfaces, "U_lower.host");
    writeDoubleArrayToFile(upper, mesh_paras.num_surfaces, "U_upper.host");
    writeDoubleArrayToFile(source_newOrder, mesh_paras.num_cells * 3, "U_source.host");


    volTensorField gradU = fvc::grad(U);

    double *gradU_newOrder = new double[mesh_paras.num_cells*9]();
    permuteTensor(gradU_newOrder, &gradU[0][0], mesh_paras.num_cells);
    writeDoubleArrayToFile(gradU_newOrder, mesh_paras.num_cells * 9, "U_grad_U.host");

    double *h_boundary_gradU = new double[mesh_paras.num_boundary_surfaces * 9];
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

    double *boundary_gradU_newOrder = new double[mesh_paras.num_boundary_surfaces * 9]();
    permuteTensor(boundary_gradU_newOrder, &h_boundary_gradU[0], mesh_paras.num_boundary_surfaces);
    writeDoubleArrayToFile(boundary_gradU_newOrder, mesh_paras.num_boundary_surfaces * 9, "U_boundary_grad_u.host");

#endif

    // process
    TICK_START;
    // UEqn_GPU.process();
    // UEqn_GPU.sync();
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
    std::vector<double> h_internal_coeffs(mesh_paras.num_boundary_surfaces * 3);
    std::vector<double> h_boundary_coeffs(mesh_paras.num_boundary_surfaces * 3);

    offset = 0;
    for (int patchi = 0; patchi < mesh_paras.num_patches; patchi++)
    {
        const fvPatchVectorField& patchU = U.boundaryField()[patchi];
        int patchsize = mesh_paras.patch_size[patchi];
        const double* internal_coeff_ptr = &UEqn.internalCoeffs()[patchi][0][0];
        const double* boundary_coeff_ptr = &UEqn.boundaryCoeffs()[patchi][0][0];
        memcpy(h_internal_coeffs.data() + offset * 3, internal_coeff_ptr, patchsize * 3 * sizeof(double));
        memcpy(h_boundary_coeffs.data() + offset * 3, boundary_coeff_ptr, patchsize * 3 * sizeof(double));
        if (patchU.type() == "processor" || patchU.type() == "processorCyclic") offset += 2 * patchsize;
        else offset += patchsize;
    }

    double *h_internal_coeffs_newOrder = new double[mesh_paras.num_boundary_surfaces * 3]();
    double *h_boundary_coeffs_newOrder = new double[mesh_paras.num_boundary_surfaces * 3]();
    permuteVector(h_internal_coeffs_newOrder, &h_internal_coeffs.data()[0], mesh_paras.num_boundary_surfaces);
    permuteVector(h_boundary_coeffs_newOrder, &h_boundary_coeffs.data()[0], mesh_paras.num_boundary_surfaces);
    writeDoubleArrayToFile(h_internal_coeffs_newOrder, mesh_paras.num_boundary_surfaces * 3, "U_internalCoeffs.host");
    writeDoubleArrayToFile(h_boundary_coeffs_newOrder, mesh_paras.num_boundary_surfaces * 3, "U_boundaryCoeffs.host");

    double *h_boundary_u_tmp = new double[mesh_paras.num_boundary_surfaces * 3];
    offset = 0;
    forAll(U.boundaryField(), patchi)
    {
        const fvPatchVectorField& patchU = U.boundaryField()[patchi];
        int patchsize = mesh_paras.patch_size[patchi];

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

    double *U_newOrder = new double[mesh_paras.num_cells * 3]();
    double *boundary_u_newOrder = new double[mesh_paras.num_boundary_surfaces * 3]();
    permuteVector(U_newOrder, &U[0][0], mesh_paras.num_cells);
    permuteVector(boundary_u_newOrder, &h_boundary_u_tmp[0], mesh_paras.num_boundary_surfaces);
    writeDoubleArrayToFile(U_newOrder, mesh_paras.num_cells * 3, "U_solved_u.host");
    writeDoubleArrayToFile(boundary_u_newOrder, mesh_paras.num_boundary_surfaces * 3, "U_solved_boundary_u.host");
    writeDoubleArrayToFile(&K[0], mesh_paras.num_cells, "U_solved_k.host");

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
    start1 = std::clock();
    tmp<fvVectorMatrix> tUEqn
    (
        fvm::ddt(rho, U) + fvm::div(phi, U)
    + turbulence->divDevRhoReff(U) 
    );
    fvVectorMatrix& UEqn = tUEqn.ref();
    end1 = std::clock();
    time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_UEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);

    // UEqn.relax();
    start1 = std::clock();
    if (pimple.momentumPredictor())
    {
        solve(UEqn == -fvc::grad(p));
        K.oldTime();
        K = 0.5*magSqr(U);
    }
    end1 = std::clock();
    time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_UEqn_solve += double(end1 - start1) / double(CLOCKS_PER_SEC);
#endif
