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

    writeDoubleArrayToFile(&turbulence->k()()[0], mesh_paras.num_cells, "turb_k.host", compareCPUResults);
    writeDoubleArrayToFile(&turbulence->epsilon()()[0], mesh_paras.num_cells, "turb_epsilon.host", compareCPUResults);
    writeDoubleArrayToFile(&turbulence->nut()()[0], mesh_paras.num_cells, "turb_nut.host", compareCPUResults);
    writeDoubleArrayToFile(&turbulence->alphat()()[0], mesh_paras.num_cells, "turb_alphat.host", compareCPUResults);
    writeDoubleArrayToFile(&turbulence->mut()()[0], mesh_paras.num_cells, "turb_mut.host", compareCPUResults);

    volScalarField& nutField = const_cast<volScalarField&>(turbulence->nut()());
    nutField.correctBoundaryConditions();

    volScalarField& alphatField = const_cast<volScalarField&>(turbulence->alphat()());
    alphatField.correctBoundaryConditions();

    int offset = 0;
    double *h_boundary_nut = new double[mesh_paras.num_boundary_surfaces]();
    double *h_boundary_alphat = new double[mesh_paras.num_boundary_surfaces]();
    double *h_boundary_mut = new double[mesh_paras.num_boundary_surfaces]();
    forAll(turbulence->nut()().boundaryField(), patchi)
    {
        Info << "patchi: " << patchi << ", nut type: " << turbulence->nut()().boundaryField()[patchi].type() << endl;
        Info << "patchi: " << patchi << ", alphat type: " << turbulence->alphat()().boundaryField()[patchi].type() << endl;
        Info << "patchi: " << patchi << ", mut type: " << turbulence->mut()().boundaryField()[patchi].type() << endl;

        const fvsPatchScalarField& patchFlux = phi.boundaryField()[patchi];
        int patchsize = mesh_paras.patch_size[patchi];

        Field<scalar> bouNut = turbulence->nut()().boundaryField()[patchi];
        // Field<scalar> bouNut = nutField.boundaryField()[patchi];
        Field<scalar> bouAlphat = turbulence->alphat()().boundaryField()[patchi]; 
        // Field<scalar> bouAlphat = alphatField.boundaryField()[patchi]; 
        Field<scalar> bouMut = turbulence->mut()().boundaryField()[patchi]; 

        memcpy(h_boundary_nut + offset, bouNut.data(), patchsize * sizeof(double));
        memcpy(h_boundary_alphat + offset, bouAlphat.data(), patchsize * sizeof(double));
        memcpy(h_boundary_mut + offset, bouMut.data(), patchsize * sizeof(double));

        if (patchFlux.type() == "processor" || patchFlux.type() == "processorCyclic") offset += 2 * patchsize;
        else offset += patchsize;

    }
    writeDoubleArrayToFile(h_boundary_nut, mesh_paras.num_boundary_surfaces, "boundary_nut.host", compareCPUResults);
    writeDoubleArrayToFile(h_boundary_alphat, mesh_paras.num_boundary_surfaces, "boundary_alphat.host", compareCPUResults);
    writeDoubleArrayToFile(h_boundary_mut, mesh_paras.num_boundary_surfaces, "boundary_mut.host", compareCPUResults);

    double *h_value_internal_coeffs = new double[mesh_paras.num_boundary_surfaces * 3]();
    double *h_value_boundary_coeffs = new double[mesh_paras.num_boundary_surfaces * 3]();
    double *h_gradient_internal_coeffs = new double[mesh_paras.num_boundary_surfaces * 3]();
    double *h_gradient_boundary_coeffs = new double[mesh_paras.num_boundary_surfaces * 3]();

    offset = 0;

    forAll(U.boundaryField(), patchi)
    {
        Info << "patchi: " << patchi << ", offset: " << offset << endl;
        // if (U.boundaryField()[patchi].type() == "wedge")
        // {
        //     vectorField snGradTransformDiag_ = dynamic_cast<const wedgeFvPatchField<vector>&>(U.boundaryField()[patchi]).snGradTransformDiag();
        //     Info << "snGradTransformDiag_: " << snGradTransformDiag_ << endl;

        //     vectorField snGrad_ = dynamic_cast<const wedgeFvPatchField<vector>&>(U.boundaryField()[patchi]).snGrad();
        //     Info << "snGrad_: " << snGrad_ << endl;
        // }

        const fvsPatchScalarField& patchFlux = phi.boundaryField()[patchi];
        const fvsPatchScalarField& pw = mesh.surfaceInterpolation::weights().boundaryField()[patchi];
        const scalarField& pDeltaCoeffs = mesh.deltaCoeffs().boundaryField()[patchi];
        int patchsize = mesh_paras.patch_size[patchi];

        // Field<vector> ueqn_internalCoeffs_vec = patchFlux*U.boundaryField()[patchi].valueInternalCoeffs(pw);
        // Field<vector> ueqn_boundaryCoeffs_vec = -patchFlux*U.boundaryField()[patchi].valueBoundaryCoeffs(pw); 

        Field<vector> ueqn_internalCoeffs_vec = U.boundaryField()[patchi].valueInternalCoeffs(pw);
        Field<vector> ueqn_boundaryCoeffs_vec = U.boundaryField()[patchi].valueBoundaryCoeffs(pw); 
        Field<vector> ueqn_laplac_internalCoeffs_vec = U.boundaryField()[patchi].gradientInternalCoeffs();
        Field<vector> ueqn_laplac_boundaryCoeffs_vec = U.boundaryField()[patchi].gradientBoundaryCoeffs();

        memcpy(h_value_internal_coeffs + offset * 3, ueqn_internalCoeffs_vec.data(), patchsize * 3 * sizeof(double));
        memcpy(h_value_boundary_coeffs + offset * 3, ueqn_boundaryCoeffs_vec.data(), patchsize * 3 * sizeof(double));
        memcpy(h_gradient_internal_coeffs + offset * 3, ueqn_laplac_internalCoeffs_vec.data(), patchsize * 3 * sizeof(double));
        memcpy(h_gradient_boundary_coeffs + offset * 3, ueqn_laplac_boundaryCoeffs_vec.data(), patchsize * 3 * sizeof(double));

        if (patchFlux.type() == "processor" || patchFlux.type() == "processorCyclic") offset += 2 * patchsize;
        else offset += patchsize;
    }

    double *d_value_internal_coeffs = new double[mesh_paras.num_boundary_surfaces * 3]();
    double *d_value_boundary_coeffs = new double[mesh_paras.num_boundary_surfaces * 3]();
    double *d_gradient_internal_coeffs = new double[mesh_paras.num_boundary_surfaces * 3]();
    double *d_gradient_boundary_coeffs = new double[mesh_paras.num_boundary_surfaces * 3]();
    permuteVector(d_value_internal_coeffs, &h_value_internal_coeffs[0], mesh_paras.num_boundary_surfaces);
    permuteVector(d_value_boundary_coeffs, &h_value_boundary_coeffs[0], mesh_paras.num_boundary_surfaces);
    permuteVector(d_gradient_internal_coeffs, &h_gradient_internal_coeffs[0], mesh_paras.num_boundary_surfaces);
    permuteVector(d_gradient_boundary_coeffs, &h_gradient_boundary_coeffs[0], mesh_paras.num_boundary_surfaces);

    writeDoubleArrayToFile(d_value_internal_coeffs, mesh_paras.num_boundary_surfaces * 3, "d_value_internal_coeffs.host", compareCPUResults);
    writeDoubleArrayToFile(d_value_boundary_coeffs, mesh_paras.num_boundary_surfaces * 3, "d_value_boundary_coeffs.host", compareCPUResults);
    writeDoubleArrayToFile(d_gradient_internal_coeffs, mesh_paras.num_boundary_surfaces * 3, "d_gradient_internal_coeffs.host", compareCPUResults);
    writeDoubleArrayToFile(d_gradient_boundary_coeffs, mesh_paras.num_boundary_surfaces * 3, "d_gradient_boundary_coeffs.host", compareCPUResults);

    // add for test
    double *diag = &UEqn.diag()[0];
    double *lower = &UEqn.lower()[0];
    double *upper = &UEqn.upper()[0];

    double *source_newOrder = new double[mesh_paras.num_cells*3]();
    permuteVector(source_newOrder, &UEqn.source()[0][0], mesh_paras.num_cells);

    writeDoubleArrayToFile(diag, mesh_paras.num_cells, "U_diag.host", compareCPUResults);
    writeDoubleArrayToFile(lower, mesh_paras.num_surfaces, "U_lower.host", compareCPUResults);
    writeDoubleArrayToFile(upper, mesh_paras.num_surfaces, "U_upper.host", compareCPUResults);
    writeDoubleArrayToFile(source_newOrder, mesh_paras.num_cells * 3, "U_source.host", compareCPUResults);


    volTensorField gradU = fvc::grad(U);

    // volTensorField gradU = gaussGradSchemeGrad(U);

    double *gradU_newOrder = new double[mesh_paras.num_cells*9]();
    permuteTensor(gradU_newOrder, &gradU[0][0], mesh_paras.num_cells);
    writeDoubleArrayToFile(gradU_newOrder, mesh_paras.num_cells * 9, "U_grad_U.host", compareCPUResults);

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
    writeDoubleArrayToFile(boundary_gradU_newOrder, mesh_paras.num_boundary_surfaces * 9, "U_boundary_grad_u.host", compareCPUResults);

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
    U.correctBoundaryConditions();
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
    writeDoubleArrayToFile(h_internal_coeffs_newOrder, mesh_paras.num_boundary_surfaces * 3, "U_internalCoeffs.host", compareCPUResults);
    writeDoubleArrayToFile(h_boundary_coeffs_newOrder, mesh_paras.num_boundary_surfaces * 3, "U_boundaryCoeffs.host", compareCPUResults);

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
    writeDoubleArrayToFile(U_newOrder, mesh_paras.num_cells * 3, "U_solved_u.host", compareCPUResults);
    writeDoubleArrayToFile(boundary_u_newOrder, mesh_paras.num_boundary_surfaces * 3, "U_solved_boundary_u.host", compareCPUResults);
    writeDoubleArrayToFile(&K[0], mesh_paras.num_cells, "U_solved_k.host", compareCPUResults);

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
