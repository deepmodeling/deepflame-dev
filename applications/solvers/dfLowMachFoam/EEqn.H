/* FOR DEEPFLAME-ACADEMIC */
#ifdef GPUSolverNew_
    process_equation(MATRIX_EQUATION::EEqn);
#endif

{
    volScalarField& he = thermo.he();

    tmp<fv::convectionScheme<scalar>> cs = 
        fv::convectionScheme<scalar>::New(mesh, phi, mesh.divScheme("div(phi,k)"));
    fv::gaussConvectionScheme<scalar>& gcs = dynamic_cast<fv::gaussConvectionScheme<scalar>&>(cs.ref());

    tmp<surfaceScalarField> tweights = gcs.interpScheme().weights(he);
    const surfaceScalarField& weights = tweights();
    // Info << "ha_weights: " << weights << endl;
    writeDoubleArrayToFile(&weights[0], mesh_paras.num_surfaces, "weights-thermohe.host", compareCPUResults);
#if defined GPUSolverNew_
    // double *h_he = dfDataBase.getFieldPointer("he", location::cpu, position::internal);
    // double *h_boundary_he = dfDataBase.getFieldPointer("he", location::cpu, position::boundary);

    // EEqn_GPU.process();
    // EEqn_GPU.sync();
    // EEqn_GPU.postProcess(h_he, h_boundary_he);

    // copy h_he to he(cpu)
    // memcpy(&he[0], h_he, dfDataBase.cell_value_bytes);

    //DEBUG_TRACE;
    //he.correctBoundaryConditions();
    //DEBUG_TRACE;

#if defined DEBUG_
    fvScalarMatrix EEqn
    (

        fvm::ddt(rho, he) + mvConvection->fvmDiv(phi, he)
    +   fvc::ddt(rho, K) + fvc::div(phi, K)
    -   dpdt
    ==
        (
            turbName == "laminar"
            ?
            (
                fvm::laplacian(turbulence->alpha(), he)
            -   diffAlphaD
            +   fvc::div(hDiffCorrFlux)
            )
            :
            (
                fvm::laplacian(turbulence->alphaEff(), he)
            )
        )
    );
    // EEqn.relax();
    EEqn.solve("ha");
    // checkResult
    // TODO: for temp, now we compare ldu, finally we compare csr
    std::vector<double> h_internal_coeffs(mesh_paras.num_boundary_surfaces);
    std::vector<double> h_boundary_coeffs(mesh_paras.num_boundary_surfaces);

    offset = 0;
    forAll(he.boundaryField(), patchi)
    {
        const fvPatchScalarField& patchHe = he.boundaryField()[patchi];
        int patchSize = patchHe.size();
        const double* internal_coeff_ptr = &EEqn.internalCoeffs()[patchi][0];
        const double* boundary_coeff_ptr = &EEqn.boundaryCoeffs()[patchi][0];
        if (patchHe.type() == "processor"
            || patchHe.type() == "processorCyclic") {
            memcpy(h_internal_coeffs.data() + offset, internal_coeff_ptr, patchSize * sizeof(double));
            memset(h_internal_coeffs.data() + offset + patchSize, 0, patchSize * sizeof(double));
            memcpy(h_boundary_coeffs.data() + offset, boundary_coeff_ptr, patchSize * sizeof(double));
            memset(h_boundary_coeffs.data() + offset + patchSize, 0, patchSize * sizeof(double));
            offset += patchSize * 2;
        } else {
            memcpy(h_internal_coeffs.data() + offset, internal_coeff_ptr, patchSize * sizeof(double));
            memcpy(h_boundary_coeffs.data() + offset, boundary_coeff_ptr, patchSize * sizeof(double));
            offset += patchSize;
        }
    }

    double *h_boundary_he_tmp = new double[mesh_paras.num_boundary_surfaces];
    offset = 0;
    forAll(he.boundaryField(), patchi)
    {
        const fvPatchScalarField& patchHe = he.boundaryField()[patchi];
        int patchSize = patchHe.size();
        if (patchHe.type() == "processor"
            || patchHe.type() == "processorCyclic") {
            const scalarField& patchHeInternal = dynamic_cast<const processorFvPatchField<scalar>&>(patchHe).patchInternalField()();
            memcpy(h_boundary_he_tmp + offset, &patchHe[0], patchSize * sizeof(double));
            memcpy(h_boundary_he_tmp + offset + patchSize, &patchHeInternal[0], patchSize * sizeof(double));
            offset += patchSize * 2;
        } else {
            memcpy(h_boundary_he_tmp + offset, &patchHe[0], patchSize * sizeof(double));
            offset += patchSize;
        }
    }

    bool printFlag = false;
    int rank = -1;
    if (mpi_init_flag) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
    if (!mpi_init_flag || rank == 0) {
		// DEBUG_TRACE;
		// EEqn_GPU.compareResult(&EEqn.lower()[0], &EEqn.upper()[0], &EEqn.diag()[0], &EEqn.source()[0],
		// 		h_internal_coeffs.data(), h_boundary_coeffs.data(), printFlag);
		// DEBUG_TRACE;
        // EEqn_GPU.compareHe(&he[0], h_boundary_he_tmp, printFlag);
    }

    writeDoubleArrayToFile(&EEqn.lower()[0], mesh_paras.num_surfaces, "E_lower.host", compareCPUResults);
    writeDoubleArrayToFile(&EEqn.upper()[0], mesh_paras.num_surfaces, "E_upper.host", compareCPUResults);
    writeDoubleArrayToFile(&EEqn.diag()[0], mesh_paras.num_cells, "E_diag.host", compareCPUResults);
    writeDoubleArrayToFile(&EEqn.source()[0], mesh_paras.num_cells, "E_source.host", compareCPUResults);
    writeDoubleArrayToFile(h_internal_coeffs.data(), mesh_paras.num_boundary_surfaces, "E_internalCoeffs.host", compareCPUResults);
    writeDoubleArrayToFile(h_boundary_coeffs.data(), mesh_paras.num_boundary_surfaces, "E_boundaryCoeffs.host", compareCPUResults);

    writeDoubleArrayToFile(&he[0], mesh_paras.num_cells, "E_he_solved.host", compareCPUResults);

    delete h_boundary_he_tmp;

#endif

#else
    start1 = std::clock();
    fvScalarMatrix EEqn
    (

        fvm::ddt(rho, he) + mvConvection->fvmDiv(phi, he)
    +   fvc::ddt(rho, K) + fvc::div(phi, K)
    -   dpdt
    ==
        (
            turbName == "laminar"
            ?
            (
                fvm::laplacian(turbulence->alpha(), he)
            -   diffAlphaD
            +   fvc::div(hDiffCorrFlux)
            )
            :
            (
                fvm::laplacian(turbulence->alphaEff(), he)
            )
        )
    );
    end1 = std::clock();
    time_monitor_EEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_EEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);

    // EEqn.relax();
    start1 = std::clock();
    EEqn.solve("ha");
    end1 = std::clock();
    time_monitor_EEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_EEqn_solve += double(end1 - start1) / double(CLOCKS_PER_SEC);
#endif
}
