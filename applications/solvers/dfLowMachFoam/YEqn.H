/* FOR DEEPFLAME-ACADEMIC */
#ifdef GPUSolverNew_
    process_equation(MATRIX_EQUATION::YEqn);
#endif

#ifdef GPUSolverNew_
#if defined DEBUG_
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
    //randomInitField<volScalarField>(const_cast<volScalarField&>(chemistry->hai(0)));
    //randomInitField<volScalarField>(const_cast<volScalarField&>(chemistry->hai(1)));
    //randomInitField<volScalarField>(const_cast<volScalarField&>(chemistry->hai(2)));
    //randomInitField<volScalarField>(const_cast<volScalarField&>(chemistry->hai(3)));
    //randomInitField<volScalarField>(const_cast<volScalarField&>(chemistry->hai(4)));
    //randomInitField<volScalarField>(const_cast<volScalarField&>(chemistry->hai(5)));
    //randomInitField<volScalarField>(const_cast<volScalarField&>(chemistry->hai(6)));
    
    auto& mgcs = dynamic_cast<fv::multivariateGaussConvectionScheme<scalar>&>(mvConvection.ref());
    tmp<surfaceInterpolationScheme<scalar>> tinterpScheme_ = mgcs.interpolationScheme()()(Y[0]);
    tmp<surfaceScalarField> tweights = tinterpScheme_().weights(Y[0]);
    const surfaceScalarField& weights = tweights();
    // Info << "CPU weights\n" << weights << endl;


    // tmp<fv::convectionScheme<scalar>> cs = 
    //     fv::convectionScheme<scalar>::New(mesh, phi, mesh.divScheme("div(phi,Yi_h)"));
    // fv::gaussConvectionScheme<scalar>& gcs = dynamic_cast<fv::gaussConvectionScheme<scalar>&>(cs.ref());
    // tmp<surfaceScalarField> tweights = gcs.interpScheme().weights(Y[0]);
    // const surfaceScalarField& weights = tweights();


    writeDoubleArrayToFile(&weights[0], mesh_paras.num_surfaces, "weights.host", true);
    writeDoubleArrayToFile(&phi[0], mesh_paras.num_surfaces, "phi0.host", true);
    
    // auto& limitedScheme_ = dynamic_cast<const limitedSurfaceInterpolationScheme<scalar>&>(tinterpScheme_());
    // Info << "CPU limiter\n" << limitedScheme_.limiter(Y[0]) << endl;

    forAll(Y, i)
    {
        sumYDiffError += chemistry->rhoD(i)*fvc::grad(Y[i]);
    }
    const surfaceScalarField phiUc = linearInterpolate(sumYDiffError) & mesh.Sf();
#endif
#else // should only for CPUSolver
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
#endif

//MPI_Barrier(PstreamGlobals::MPI_COMM_FOAM);
label flag_mpi_init;
MPI_Initialized(&flag_mpi_init);
if(flag_mpi_init) MPI_Barrier(PstreamGlobals::MPI_COMM_FOAM);

{

#ifdef GPUSolverNew_
#if defined DEBUG_
    // run CPU
    volScalarField Yt(0.0*Y[0]);

    int specie_index = 0;

    // should compute grad_yi before YiEqn.solve()
    const volVectorField grad_yi = fvc::grad(Y[0]);
    double *grad_y_newOrder = new double[mesh_paras.num_cells * 3]();
    permuteVector(grad_y_newOrder, &grad_yi[0][0], mesh_paras.num_cells);
    writeDoubleArrayToFile(&grad_y_newOrder[0], mesh_paras.num_cells * 3, "Y_grad_y.host", compareCPUResults);

    int speciesIndex = 0;
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
            // YiEqn.relax();

            start1 = std::clock();
            YiEqn.solve("Yi");
            end1 = std::clock();
            time_monitor_YEqn_solve += double(end1 - start1) / double(CLOCKS_PER_SEC);

            Yi.max(0.0);
            Yt += Yi;
            ++speciesIndex;
        }
    }
    Y[inertIndex] = scalar(1) - Yt;
    Y[inertIndex].max(0.0);

    tmp<volScalarField> DEff = chemistry->rhoD(specie_index) + turbulence->mut()/Sct;
    fvScalarMatrix YiEqn
        (
         fvm::ddt(rho, Y[specie_index])
         + mvConvection->fvmDiv(phi, Y[specie_index])
         + mvConvection->fvmDiv(phiUc, Y[specie_index])
         ==
         fvm::laplacian(DEff(), Y[specie_index])
        );
    // YiEqn.relax();
    // YiEqn.solve("Yi");
    // Y[specie_index].max(0.0);
#endif

    // // process
    // YEqn_GPU.process();
    // YEqn_GPU.sync();

#if defined DEBUG_
    std::vector<double> h_boundary_diffAlphaD;
    std::vector<double> h_boundary_grad_yi;
    std::vector<double> h_boundary_sumYDiffError;
    std::vector<double> h_boundary_hDiffCorrFlux;
    std::vector<double> h_boundary_phiUc;
    h_boundary_diffAlphaD.resize(mesh_paras.num_boundary_surfaces);
    h_boundary_grad_yi.resize(mesh_paras.num_boundary_surfaces * 3);
    h_boundary_sumYDiffError.resize(mesh_paras.num_boundary_surfaces * 3);
    h_boundary_hDiffCorrFlux.resize(mesh_paras.num_boundary_surfaces * 3);
    h_boundary_phiUc.resize(mesh_paras.num_boundary_surfaces);
    offset = 0;
    forAll(diffAlphaD.boundaryField(), patchi)
    {
        //const scalarField& patchdiffAlphaD = diffAlphaD.boundaryField()[patchi];
        const fvPatchScalarField& patchdiffAlphaD = diffAlphaD.boundaryField()[patchi];
        const fvPatchVectorField& patchgradyi = grad_yi.boundaryField()[patchi];
        const fvPatchVectorField& patchsumYDiffError = sumYDiffError.boundaryField()[patchi];
        const fvPatchVectorField& patchhDiffCorrFlux = hDiffCorrFlux.boundaryField()[patchi];
        const fvsPatchScalarField& patchphiUc = phiUc.boundaryField()[patchi];
        int patchSize = patchdiffAlphaD.size();
        if (patchdiffAlphaD.type() == "processor"
            || patchdiffAlphaD.type() == "processorCyclic") {
            scalarField patchdiffAlphaDInternal = dynamic_cast<const processorFvPatchField<scalar>&>(patchdiffAlphaD).patchInternalField()();
            vectorField patchgradyiInternal = dynamic_cast<const processorFvPatchField<vector>&>(patchgradyi).patchInternalField()();
            vectorField patchsumYDiffErrorInternal = dynamic_cast<const processorFvPatchField<vector>&>(patchsumYDiffError).patchInternalField()();
            vectorField patchhDiffCorrFluxInternal = dynamic_cast<const processorFvPatchField<vector>&>(patchhDiffCorrFlux).patchInternalField()();
            memcpy(h_boundary_diffAlphaD.data() + offset, &patchdiffAlphaD[0], patchSize*sizeof(double));
            memcpy(h_boundary_diffAlphaD.data() + offset + patchSize, &patchdiffAlphaDInternal[0], patchSize*sizeof(double));
            memcpy(h_boundary_grad_yi.data() + offset * 3, &patchgradyi[0][0], patchSize * 3 *sizeof(double));
            memcpy(h_boundary_grad_yi.data() + (offset + patchSize) * 3, &patchgradyiInternal[0][0], patchSize * 3 *sizeof(double));
            memcpy(h_boundary_sumYDiffError.data() + offset * 3, &patchsumYDiffError[0][0], patchSize * 3 *sizeof(double));
            memcpy(h_boundary_sumYDiffError.data() + (offset + patchSize) * 3, &patchsumYDiffErrorInternal[0][0], patchSize * 3 *sizeof(double));
            memcpy(h_boundary_hDiffCorrFlux.data() + offset * 3, &patchhDiffCorrFlux[0][0], patchSize * 3 *sizeof(double));
            memcpy(h_boundary_hDiffCorrFlux.data() + (offset + patchSize) * 3, &patchhDiffCorrFluxInternal[0][0], patchSize * 3 *sizeof(double));
            memcpy(h_boundary_phiUc.data() + offset, &patchphiUc[0], patchSize*sizeof(double));
            memcpy(h_boundary_phiUc.data() + offset, &patchphiUc[0], patchSize*sizeof(double));
            offset += patchSize * 2;
        } else {
            memcpy(h_boundary_diffAlphaD.data() + offset, &patchdiffAlphaD[0], patchSize*sizeof(double));
            memcpy(h_boundary_grad_yi.data() + offset * 3, &patchgradyi[0][0], patchSize * 3 *sizeof(double));
            memcpy(h_boundary_sumYDiffError.data() + offset * 3, &patchsumYDiffError[0][0], patchSize * 3 *sizeof(double));
            memcpy(h_boundary_hDiffCorrFlux.data() + offset * 3, &patchhDiffCorrFlux[0][0], patchSize * 3 *sizeof(double));
            memcpy(h_boundary_phiUc.data() + offset, &patchphiUc[0], patchSize*sizeof(double));
            offset += patchSize;
        }
    }
    DEBUG_TRACE;
    // YEqn_GPU.comparediffAlphaD(&diffAlphaD[0], h_boundary_diffAlphaD.data(), false);
    // YEqn_GPU.comparegradyi(&grad_yi[0][0], h_boundary_grad_yi.data(), specie_index, false);
    // YEqn_GPU.comparesumYDiffError(&sumYDiffError[0][0], h_boundary_sumYDiffError.data(), false);
    // YEqn_GPU.comparehDiffCorrFlux(&hDiffCorrFlux[0][0], h_boundary_hDiffCorrFlux.data(), false);
    // YEqn_GPU.comparephiUc(&phiUc[0], h_boundary_phiUc.data(), false);
    DEBUG_TRACE;

    // checkResult
    // TODO: for temp, now we compare ldu, finally we compare csr
    std::vector<double> yeqn_h_internal_coeffs(mesh_paras.num_boundary_surfaces);
    std::vector<double> yeqn_h_boundary_coeffs(mesh_paras.num_boundary_surfaces);

    offset = 0;
    forAll(Y[specie_index].boundaryField(), patchi)
    {
        const fvPatchScalarField& patchYi = Y[specie_index].boundaryField()[patchi];
        int patchsize = patchYi.size();
        const double* internal_coeff_ptr = &YiEqn.internalCoeffs()[patchi][0];
        const double* boundary_coeff_ptr = &YiEqn.boundaryCoeffs()[patchi][0];
        if (patchYi.type() == "processor"
            || patchYi.type() == "processorCyclic") {
            memcpy(yeqn_h_internal_coeffs.data() + offset, internal_coeff_ptr, patchsize * sizeof(double));
            memset(yeqn_h_internal_coeffs.data() + offset + patchsize, 0, patchsize * sizeof(double));
            memcpy(yeqn_h_boundary_coeffs.data() + offset, boundary_coeff_ptr, patchsize * sizeof(double));
            memset(yeqn_h_boundary_coeffs.data() + offset + patchsize, 0, patchsize * sizeof(double));
            offset += patchsize * 2;
        } else {
            memcpy(yeqn_h_internal_coeffs.data() + offset, internal_coeff_ptr, patchsize * sizeof(double));
            memcpy(yeqn_h_boundary_coeffs.data() + offset, boundary_coeff_ptr, patchsize * sizeof(double));
            offset += patchsize;
        }
    }
    // NOTE: ldu and yi can't be compared at the same time
    // to compare ldu data, you should open both DEBUG_ and DEBUG_CHECK_LDU in src_gpu
    // to compare yi, you should only open DEBUG_ in src_gpu.
    // Besides, if you compare ldu data, be patient to keep specie_index in YEqn.H and dfYEqn.cu the same.
    //DEBUG_TRACE;
    bool printFlag = false;
    int rank = -1;
    if (mpi_init_flag) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
    if (!mpi_init_flag || rank == 0) {
        // YEqn_GPU.compareResult(&YiEqn.lower()[0], &YiEqn.upper()[0], &YiEqn.diag()[0], &YiEqn.source()[0],
        //         yeqn_h_internal_coeffs.data(), yeqn_h_boundary_coeffs.data(), printFlag);
    }
    // double diffAlphaD_newOrder[mesh_paras.num_cells * 3];
    // double hDiffCorrFlux_newOrder[mesh_paras.num_cells * 3];
    // double sumYDiffError_newOrder[mesh_paras.num_cells * 3];
    // permuteVector(diffAlphaD_newOrder, &diffAlphaD[0], mesh_paras.num_cells);
    // permuteVector(hDiffCorrFlux_newOrder, &hDiffCorrFlux[0][0], mesh_paras.num_cells);
    // permuteVector(sumYDiffError_newOrder, &sumYDiffError[0][0], mesh_paras.num_cells);

    // writeDoubleArrayToFile(&diffAlphaD_newOrder[0], mesh_paras.num_cells * 3, "Y_diffAlphaD.host", compareCPUResults);
    // writeDoubleArrayToFile(&hDiffCorrFlux_newOrder[0], mesh_paras.num_cells * 3, "Y_hDiffCorrFlux.host", compareCPUResults);
    // writeDoubleArrayToFile(&sumYDiffError_newOrder[0], mesh_paras.num_cells * 3, "Y_sumYDiffError.host", compareCPUResults);
    // writeDoubleArrayToFile(&phiUc[0], mesh_paras.num_surfaces, "Y_phiUc.host", compareCPUResults);

    writeDoubleArrayToFile(&YiEqn.lower()[0], mesh_paras.num_surfaces, "Y_lower0.host", compareCPUResults);
    writeDoubleArrayToFile(&YiEqn.upper()[0], mesh_paras.num_surfaces, "Y_upper0.host", compareCPUResults);
    writeDoubleArrayToFile(&YiEqn.diag()[0], mesh_paras.num_cells, "Y_diag0.host", compareCPUResults);
    writeDoubleArrayToFile(&YiEqn.source()[0], mesh_paras.num_cells, "Y_source0.host", compareCPUResults);
    writeDoubleArrayToFile(yeqn_h_internal_coeffs.data(), mesh_paras.num_boundary_surfaces, "Y_internalCoeffs0.host", compareCPUResults);
    writeDoubleArrayToFile(yeqn_h_boundary_coeffs.data(), mesh_paras.num_boundary_surfaces, "Y_boundaryCoeffs0.host", compareCPUResults);

    writeDoubleArrayToFile(&Y[0][0], mesh_paras.num_cells, "Y_solved_y.host", compareCPUResults);
    // writeDoubleArrayToFile(&Y[3][0], mesh_paras.num_cells, "Y_solved_y.host", compareCPUResults);

    DEBUG_TRACE;
    // YEqn_GPU.compareYi(&Y[specie_index][0], specie_index, false);
    // DEBUG_TRACE;
#endif

    // // postProcess
    // double *h_y = mesh_paras.getFieldPointer("y", location::cpu, position::internal);
    // double *h_boundary_y = mesh_paras.getFieldPointer("y", location::cpu, position::boundary);
    // // YEqn_GPU.postProcess(h_y, h_boundary_y);
    // DEBUG_TRACE;

    // copy h_y to Y(cpu)
    // offset = 0;
    // forAll(Y, i)
    // {
    //     volScalarField& Yi = Y[i];
    //     memcpy(&Yi[0], h_y + offset, Yi.size() * sizeof(double));
    //     offset += Yi.size();
    //     Yi.correctBoundaryConditions();
    // }
    DEBUG_TRACE;

    fflush(stderr);
#else
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
    int speciesIndex = 0;
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
            // YiEqn.relax();

            start1 = std::clock();
            YiEqn.solve("Yi");
            end1 = std::clock();
            time_monitor_YEqn_solve += double(end1 - start1) / double(CLOCKS_PER_SEC);

            Yi.max(0.0);
            Yt += Yi;
            ++speciesIndex;
        }
    }

    Y[inertIndex] = scalar(1) - Yt;
    Y[inertIndex].max(0.0);
    end2 = std::clock();
    time_monitor_YEqn += double(end2 - start2) / double(CLOCKS_PER_SEC);
#endif
}
