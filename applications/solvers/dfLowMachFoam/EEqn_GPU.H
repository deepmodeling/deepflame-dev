{
    volScalarField& he = thermo.he();
    start1 = std::clock();
    UEqn_GPU.updatePsi(&U[0][0]);
    UEqn_GPU.correctBoundaryConditions();
    U.correctBoundaryConditions();
    K = 0.5*magSqr(U);
    end1 = std::clock();
    time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_UEqn_correctBC += double(end1 - start1) / double(CLOCKS_PER_SEC);

    // prepare data on CPU
    start1 = std::clock();
    start2 = std::clock();
    // const tmp<volScalarField> alphaEff_tmp(thermo.alpha());
    // const volScalarField& alphaEff = alphaEff_tmp();
    double *alphaEff = nullptr; // tmp
    end2 = std::clock();
    int eeqn_offset = 0;
    int patchNum = 0;

    forAll(he.boundaryField(), patchi)
    {
        patchNum++;
        const fvsPatchScalarField& pw = mesh.surfaceInterpolation::weights().boundaryField()[patchi];
        int patchSize = pw.size();

        // construct gradient manually
        const fvPatchScalarField& hew = he.boundaryField()[patchi];
        const basicThermo& bThermo = basicThermo::lookupThermo(hew);
        const scalarField& ppw = bThermo.p().boundaryField()[patchi];
        fvPatchScalarField& Tw =
            const_cast<fvPatchScalarField&>(bThermo.T().boundaryField()[patchi]);
        scalarField& Tw_v = Tw;

        Tw.evaluate();
        const scalarField& patchDeltaCoeff = mesh.boundary()[patchi].deltaCoeffs();
        const scalarField heInternal = bThermo.he(ppw, Tw, patchi)();
        const scalarField heBoundary = bThermo.he(ppw, Tw, mesh.boundary()[patchi].faceCells())();
        const scalarField patchGradMau = patchDeltaCoeff * (heInternal - heBoundary);

        const scalarField& patchK = K.boundaryField()[patchi];
        // const scalarField& patchAlphaEff = alphaEff.boundaryField()[patchi]; // not H2Dcopy when use UnityLewis
        // const scalarField& patchGrad = he.boundaryField()[patchi].gradientBoundaryCoeffs(); // gradient_

        // const DimensionedField<scalar, volMesh>& patchHa_ = he.boundaryField()[patchi];
        // const gradientEnergyFvPatchScalarField patchHa(mesh.boundary()[patchi], patchHa_);
        // const scalarField& patchGrad = patchHa.gradient(); // gradient_
        memcpy(boundary_K + eeqn_offset, &patchK[0], patchSize*sizeof(double));
        // memcpy(boundary_alphaEff + eeqn_offset, &patchAlphaEff[0], patchSize*sizeof(double));
        memcpy(boundary_gradient + eeqn_offset, &patchGradMau[0], patchSize*sizeof(double));

        eeqn_offset += patchSize;
    }
    end1 = std::clock();
    time_monitor_EEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_EEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_EEqn_mtxAssembly_CPU_prepare += double(end1 - start1) / double(CLOCKS_PER_SEC);
    fprintf(stderr, "time_monitor_EEqn_mtxAssembly_CPU_prepare: %lf, build alphaEff time: %lf, patchNum: %d\n",
            time_monitor_EEqn_mtxAssembly_CPU_prepare,
            double(end2 - start2) / double(CLOCKS_PER_SEC), patchNum);

    // prepare data on GPU
    start1 = std::clock();
    he.oldTime();
    K.oldTime();
    EEqn_GPU.prepare_data(&he.oldTime()[0], &K[0], &K.oldTime()[0], alphaEff,
            &dpdt[0], boundary_K, boundary_alphaEff, boundary_gradient);
    EEqn_GPU.sync();
    end1 = std::clock();
    time_monitor_EEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_EEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_EEqn_mtxAssembly_GPU_prepare += double(end1 - start1) / double(CLOCKS_PER_SEC);

    start1 = std::clock();
    EEqn_GPU.initializeTimeStep();
    EEqn_GPU.fvm_ddt();
    EEqn_GPU.fvm_div();
    EEqn_GPU.fvm_laplacian();
    EEqn_GPU.fvc_ddt();
    EEqn_GPU.fvc_div_phi_scalar();
    EEqn_GPU.fvc_div_vector();
    EEqn_GPU.add_to_source();
    EEqn_GPU.sync();
    end1 = std::clock();
    time_monitor_EEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_EEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_EEqn_mtxAssembly_GPU_run += double(end1 - start1) / double(CLOCKS_PER_SEC);

    // check value of mtxAssembly, no time monitor
    // EEqn_GPU.checkValue(true);

    start1 = std::clock();
    EEqn_GPU.solve();
    end1 = std::clock();
    time_monitor_EEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_EEqn_solve += double(end1 - start1) / double(CLOCKS_PER_SEC);

    start1 = std::clock();
    EEqn_GPU.updatePsi(&he[0]);
    he.correctBoundaryConditions();
    he.write();
    end1 = std::clock();
    time_monitor_EEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_EEqn_correctBC += double(end1 - start1) / double(CLOCKS_PER_SEC);
}
