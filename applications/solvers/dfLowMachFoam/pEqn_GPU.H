/* FOR DEEPFLAME-ACADEMIC */
#ifdef GPUSolverNew_

    // thermo_GPU.updateRho();
    // thermo_GPU.psip0();
    // UEqn_GPU.getHbyA();
    
    process_equation(MATRIX_EQUATION::pEqn); 

    if (doCorrectBCsCPU) correctVelBCsHost(mesh_paras, U);

    // thermo_GPU.correctPsipRho();

#endif

// Thermodynamic density needs to be updated by psi*d(p) after the
// pressure solution

#include <mpi.h>
#include <iostream>
#include <vector>
#include <fstream>

#if defined DEBUG_

    if (doCorrectBCsCPU)
    {
        memcpy(&U[0][0], &U.oldTime()[0][0], mesh_paras.num_cells * 3 * sizeof(double));
        U.correctBoundaryConditions();
    }

    if (!pimple.simpleRho())
    {
        rho = thermo.rho();
    }

    const volScalarField psip0(psi*p);

    volScalarField rAU(1/UEqn.A());
    surfaceScalarField rhorAUf("rhorAUf", fvc::interpolate(rho*rAU));
    volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U, p));

    volVectorField UEqn_H(UEqn.H());
    double *UEqn_H_newOrder = new double[mesh_paras.num_cells*3]();
    permuteVector(UEqn_H_newOrder, &UEqn_H[0][0], mesh_paras.num_cells);
    writeDoubleArrayToFile(UEqn_H_newOrder, mesh_paras.num_cells * 3, "d_UEqn_H.host", compareCPUResults);

    double *HbyA_newOrder = new double[mesh_paras.num_cells*3]();
    permuteVector(HbyA_newOrder, &HbyA[0][0], mesh_paras.num_cells);
    writeDoubleArrayToFile(&rho[0], mesh_paras.num_cells, "d_rho_thermo.host", compareCPUResults);
    writeDoubleArrayToFile(&rAU[0], mesh_paras.num_cells, "d_rAU.host", compareCPUResults);
    writeDoubleArrayToFile(HbyA_newOrder, mesh_paras.num_cells * 3, "d_HbyA.host", compareCPUResults);
    writeDoubleArrayToFile(&rhorAUf[0], mesh_paras.num_surfaces, "d_rhorAUf.host", compareCPUResults);

    writeDoubleArrayToFile(&psip0[0], mesh_paras.num_cells, "thermo_d_psip0.host", compareCPUResults);

    double *h_boundary_rhoT = new double[mesh_paras.num_boundary_surfaces];
    double *h_boundary_psip0 = new double[mesh_paras.num_boundary_surfaces];
    double *h_boundary_rAU = new double[mesh_paras.num_boundary_surfaces];
    double *h_boundary_rhorAUf = new double[mesh_paras.num_boundary_surfaces];
    double *h_boundary_HbyA = new double[3 * mesh_paras.num_boundary_surfaces];
    offset = 0;
    forAll(rAU.boundaryField(), patchi)
    {
        const fvPatchScalarField& patchpsip0 = psip0.boundaryField()[patchi];
        const fvPatchScalarField& patchrho = rho.boundaryField()[patchi];
        const fvPatchScalarField& patchrAU = rAU.boundaryField()[patchi];
        const fvPatchVectorField& patchHbyA = HbyA.boundaryField()[patchi];
        const fvsPatchScalarField& patchrhorAUf = rhorAUf.boundaryField()[patchi];
        int patchSize = patchrAU.size();

        if (patchrAU.type() == "processor"
            || patchrAU.type() == "processorCyclic") {
            memcpy(h_boundary_rAU + offset, &patchrAU[0], patchSize*sizeof(double));
            scalarField patchrAUInternal = 
                    dynamic_cast<const processorFvPatchField<scalar>&>(patchrAU).patchInternalField()();
            memcpy(h_boundary_rAU + offset + patchSize, &patchrAUInternal[0], patchSize*sizeof(double));

            memcpy(h_boundary_rhorAUf + offset, &patchrhorAUf[0], patchSize*sizeof(double));
            memcpy(h_boundary_rhoT + offset, &patchrho[0], patchSize*sizeof(double));
            scalarField patchrhoTInternal = dynamic_cast<const processorFvPatchField<scalar>&>(patchrho).patchInternalField()();
            memcpy(h_boundary_rhoT + offset + patchSize, &patchrhoTInternal[0], patchSize*sizeof(double));

            memcpy(h_boundary_psip0 + offset, &patchpsip0[0], patchSize*sizeof(double));

            memcpy(h_boundary_HbyA + offset * 3, &patchHbyA[0][0], patchSize*3*sizeof(double));
            vectorField patchHbyAInternal = 
                    dynamic_cast<const processorFvPatchField<vector>&>(patchHbyA).patchInternalField()();
            memcpy(h_boundary_HbyA + offset * 3 + patchSize * 3, &patchHbyAInternal[0][0], patchSize*3*sizeof(double));

            offset += patchSize * 2;
        } else {
            memcpy(h_boundary_psip0 + offset, &patchpsip0[0], patchSize*sizeof(double));
            memcpy(h_boundary_rhoT + offset, &patchrho[0], patchSize*sizeof(double));
            memcpy(h_boundary_rAU + offset, &patchrAU[0], patchSize*sizeof(double));
            memcpy(h_boundary_rhorAUf + offset, &patchrhorAUf[0], patchSize*sizeof(double));
            memcpy(h_boundary_HbyA + offset * 3, &patchHbyA[0][0], patchSize*3*sizeof(double));
            offset += patchSize;
        }
    }
    double *boundary_HbyA_newOrder = new double[mesh_paras.num_boundary_surfaces*3]();
    permuteVector(boundary_HbyA_newOrder, &h_boundary_HbyA[0], mesh_paras.num_boundary_surfaces);
    writeDoubleArrayToFile(boundary_HbyA_newOrder, mesh_paras.num_boundary_surfaces * 3, "d_HbyA_boundary.host", compareCPUResults);
    writeDoubleArrayToFile(h_boundary_rhoT, mesh_paras.num_boundary_surfaces, "d_rho_thermo_boundary.host", compareCPUResults);
    writeDoubleArrayToFile(h_boundary_rAU, mesh_paras.num_boundary_surfaces, "d_rAU_boundary.host", compareCPUResults);
    writeDoubleArrayToFile(h_boundary_rhorAUf, mesh_paras.num_boundary_surfaces, "d_rhorAUf_boundary.host", compareCPUResults);
    writeDoubleArrayToFile(h_boundary_psip0, mesh_paras.num_boundary_surfaces, "thermo_d_boundary_psip0.host", compareCPUResults);
    if (!mpi_init_flag || rank == 0) {
        // UEqn_GPU.compareHbyA(&HbyA[0][0], h_boundary_HbyA, false);
        // UEqn_GPU.comparerAU(&rAU[0], h_boundary_rAU, false);
        // pEqn_GPU.comparerhorAUf(&rhorAUf[0], h_boundary_rhorAUf, false);
    }
    delete h_boundary_rhoT;
    delete h_boundary_rAU;
    delete h_boundary_rhorAUf;
    delete h_boundary_HbyA;
    delete h_boundary_psip0;
#endif

#if defined DEBUG_
    surfaceScalarField phiHbyA
    (
        "phiHbyA",
        fvc::interpolate(rho)*fvc::flux(HbyA)
    + rhorAUf*fvc::ddtCorr(rho, U, phi, rhoUf)
    );


    // EulerDdtSchemeFvcDdtCorr(rho, U, phi, rhoUf);

    double *h_boundary_phiHbyA = (double*)calloc(mesh_paras.num_boundary_surfaces, sizeof(double));
    offset = 0;
    forAll(phiHbyA.boundaryField(), patchi)
    {
        const fvsPatchScalarField& patchphiHbyA = phiHbyA.boundaryField()[patchi];
        int patchSize = patchphiHbyA.size();
        if (patchphiHbyA.type() == "processor"
            || patchphiHbyA.type() == "processorCyclic") {
            memcpy(h_boundary_phiHbyA + offset, &patchphiHbyA[0], patchSize*sizeof(double));
            offset += 2 * patchSize;
        } else {
            memcpy(h_boundary_phiHbyA + offset, &patchphiHbyA[0], patchSize*sizeof(double));
            offset += patchSize;
        }
    }
    if (!mpi_init_flag || rank == 0) {
        // pEqn_GPU.comparephiHbyA(&phiHbyA[0], h_boundary_phiHbyA, false);
    }
    writeDoubleArrayToFile(&phiHbyA[0], mesh_paras.num_surfaces, "d_phiHbyA.host", compareCPUResults);
    writeDoubleArrayToFile(&h_boundary_phiHbyA[0], mesh_paras.num_boundary_surfaces, "d_phiHbyA_boundary.host", compareCPUResults);
    delete h_boundary_phiHbyA;
#endif

#if defined DEBUG_
    fvScalarMatrix pDDtEqn
    (
        fvc::ddt(rho) + 
        psi*correction(fvm::ddt(p))
        + fvc::div(phiHbyA)
    );
    while (pimple.correctNonOrthogonal())
    {
        fvScalarMatrix pEqn(pDDtEqn - fvm::laplacian(rhorAUf, p));

        // csrMatrix csr(pEqn);
        // auto& p_psi = const_cast<GeometricField<scalar, fvPatchField, volMesh>&>(pEqn.psi());
        // auto& source = pEqn.source();
        // auto& internalCoeffs = pEqn.internalCoeffs();
        // auto& boundaryCoeffs = pEqn.boundaryCoeffs();

        // writeDoubleArrayToFile(&pEqn.lower()[0], mesh_paras.num_surfaces, "p_lower.host", compareCPUResults);
        writeDoubleArrayToFile(&pEqn.upper()[0], mesh_paras.num_surfaces, "p_upper.host", compareCPUResults);
        writeDoubleArrayToFile(&pEqn.diag()[0], mesh_paras.num_cells, "p_diag.host", compareCPUResults);
        writeDoubleArrayToFile(&pEqn.source()[0], mesh_paras.num_cells, "p_source.host", compareCPUResults);

        writeDoubleArrayToFile(&psi[0], mesh_paras.num_cells, "d_thermo_psi.host", compareCPUResults);

        // double *h_value_internal_coeffs = new double[mesh_paras.num_boundary_surfaces]();
        // double *h_value_boundary_coeffs = new double[mesh_paras.num_boundary_surfaces]();
        // double *h_gradient_internal_coeffs = new double[mesh_paras.num_boundary_surfaces]();
        // double *h_gradient_boundary_coeffs = new double[mesh_paras.num_boundary_surfaces]();

        // offset = 0;
        // forAll(U.boundaryField(), patchi)
        // {
        //     const fvsPatchScalarField& patchFlux = phi.boundaryField()[patchi];
        //     const fvsPatchScalarField& pw = mesh.surfaceInterpolation::weights().boundaryField()[patchi];
        //     const scalarField& pDeltaCoeffs = mesh.deltaCoeffs().boundaryField()[patchi];
        //     int patchsize = mesh_paras.patch_size[patchi];

        //     Field<scalar> peqn_internalCoeffs = p.boundaryField()[patchi].valueInternalCoeffs(pw);
        //     Field<scalar> peqn_boundaryCoeffs = p.boundaryField()[patchi].valueBoundaryCoeffs(pw); 
        //     Field<scalar> peqn_laplac_internalCoeffs = p.boundaryField()[patchi].gradientInternalCoeffs();
        //     Field<scalar> peqn_laplac_boundaryCoeffs = p.boundaryField()[patchi].gradientBoundaryCoeffs();

        //     memcpy(h_value_internal_coeffs + offset, peqn_internalCoeffs.data(), patchsize * sizeof(double));
        //     memcpy(h_value_boundary_coeffs + offset, peqn_boundaryCoeffs.data(), patchsize * sizeof(double));
        //     memcpy(h_gradient_internal_coeffs + offset, peqn_laplac_internalCoeffs.data(), patchsize * sizeof(double));
        //     memcpy(h_gradient_boundary_coeffs + offset, peqn_laplac_boundaryCoeffs.data(), patchsize * sizeof(double));

        //     if (patchFlux.type() == "processor" || patchFlux.type() == "processorCyclic") offset += 2 * patchsize;
        //     else offset += patchsize;
        // }

        // writeDoubleArrayToFile(h_value_internal_coeffs, mesh_paras.num_boundary_surfaces, "p_value_internal_coeffs.host", compareCPUResults);
        // writeDoubleArrayToFile(h_value_boundary_coeffs, mesh_paras.num_boundary_surfaces, "p_value_boundary_coeffs.host", compareCPUResults);
        // writeDoubleArrayToFile(h_gradient_internal_coeffs, mesh_paras.num_boundary_surfaces, "p_gradient_internal_coeffs.host", compareCPUResults);
        // writeDoubleArrayToFile(h_gradient_boundary_coeffs, mesh_paras.num_boundary_surfaces, "p_gradient_boundary_coeffs.host", compareCPUResults);
        
        // std::cout << "=== start call csr.solve " << std::endl;

        // int mpirank;
        // MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
        // std::string c = std::to_string(mpirank);        
        // std::string a1 = "diag";
        // std::string b1 = ".bin";
        // std::string filename1 = a1 + c + b1;
        // // saveArrayToBinaryFile(filename1, &csr.diag()[0], mesh_paras.num_cells);
        // std::ofstream outFile1(filename1, std::ios::binary);
        // outFile1.write(reinterpret_cast<const char*>(&csr.diag()[0]), mesh_paras.num_cells * sizeof(double));
        // outFile1.close();

        // std::string a2 = "lower";
        // std::string b2 = ".bin";
        // std::string filename2 = a2 + c + b2;
        // // saveArrayToBinaryFile(filename2, &csr.ldu().lower()[0], mesh_paras.num_surfaces);
        // std::ofstream outFile2(filename2, std::ios::binary);
        // outFile2.write(reinterpret_cast<const char*>(&csr.ldu().lower()[0]), mesh_paras.num_surfaces * sizeof(double));
        // outFile2.close();

        // std::string a3 = "upper";
        // std::string b3 = ".bin";
        // std::string filename3 = a3 + c + b3;
        // // saveArrayToBinaryFile(filename3, &csr.ldu().upper()[0], mesh_paras.num_surfaces);
        // std::ofstream outFile3(filename3, std::ios::binary);
        // outFile3.write(reinterpret_cast<const char*>(&csr.ldu().upper()[0]), mesh_paras.num_surfaces * sizeof(double));
        // outFile3.close();

        // int toffset = 0;
        // std::vector<double> vecInt;
        // std::vector<double> vecBou;
        // forAll(p.boundaryField(), patchi)
        // {
        //     const fvPatchScalarField& patchPi = p.boundaryField()[patchi];
        //     int patchsize = patchPi.size();
        //     const double* internal_coeff_ptr = &pEqn.internalCoeffs()[patchi][0];
        //     const double* boundary_coeff_ptr = &pEqn.boundaryCoeffs()[patchi][0];
        //     if (mesh_paras.patch_type_extropolated.data()[patchi] == boundaryConditions::processor
        //         || mesh_paras.patch_type_extropolated.data()[patchi] == boundaryConditions::processorCyclic) {
        //         vecInt.insert(vecInt.end(), internal_coeff_ptr, internal_coeff_ptr + patchsize);
        //         vecInt.insert(vecInt.end(), internal_coeff_ptr, internal_coeff_ptr + patchsize);
        //         vecBou.insert(vecBou.end(), boundary_coeff_ptr, boundary_coeff_ptr + patchsize);
        //         vecBou.insert(vecBou.end(), boundary_coeff_ptr, boundary_coeff_ptr + patchsize);
        //         toffset += patchsize * 2;
        //     } else {
        //         vecInt.insert(vecInt.end(), internal_coeff_ptr, internal_coeff_ptr + patchsize);
        //         vecBou.insert(vecBou.end(), boundary_coeff_ptr, boundary_coeff_ptr + patchsize);
        //         toffset += patchsize;
        //     }
        // }

        // std::string a1 = "IntCoef";
        // std::string b1 = ".bin";
        // std::string filename1 = a1 + c + b1;
        // // saveArrayToBinaryFile(filename1, &csr.diag()[0], mesh_paras.num_cells);
        // std::ofstream outFile1(filename1, std::ios::binary);
        // outFile1.write(reinterpret_cast<const char*>(&vecInt[0]), toffset * sizeof(double));
        // outFile1.close();

        // std::string a2 = "BouCoef";
        // std::string b2 = ".bin";
        // std::string filename2 = a2 + c + b2;
        // // saveArrayToBinaryFile(filename1, &csr.diag()[0], mesh_paras.num_cells);
        // std::ofstream outFile2(filename2, std::ios::binary);
        // outFile2.write(reinterpret_cast<const char*>(&vecBou[0]), toffset * sizeof(double));
        // outFile2.close();

        // MPI_Abort(MPI_COMM_WORLD, 1);

        // csr.solve(p_psi, source, internalCoeffs, boundaryCoeffs);
        // std::cout << "=== end call csr.solve " << std::endl;

        pEqn.solve();

        // p.correctBoundaryConditions(); //need check, update phi after correctBC
        
        if (pimple.finalNonOrthogonalIter())
        {
            phi = phiHbyA + pEqn.flux();
        }
        surfaceScalarField pEqn_flux(pEqn.flux());
        writeDoubleArrayToFile(&pEqn_flux[0], mesh_paras.num_surfaces, "d_pEqn_flux.host", compareCPUResults);
        // Info << "pEqn.flux(): " << pEqn.flux() << endl;

        thermo.correctRho(psi*p - psip0);

        bool limitedp = pressureControl.limit(p);
        // Info << "limitedp: " << limitedp << endl;
        if (limitedp)
        {
            rho = thermo.rho();
        }

        writeDoubleArrayToFile(&thermo.rho()()[0], mesh_paras.num_cells, "d_rho_thermo_correctRho.host", compareCPUResults);
        writeDoubleArrayToFile(&rho[0], mesh_paras.num_cells, "d_rho_thermo_correctRho0.host", compareCPUResults);

        double *h_boundary_rhoC = new double[mesh_paras.num_boundary_surfaces];

        offset = 0;
        forAll(rAU.boundaryField(), patchi)
        {
            const fvPatchScalarField& patchrho = thermo.rho()().boundaryField()[patchi];
            int patchSize = patchrho.size();

            if (patchrho.type() == "processor"
                || patchrho.type() == "processorCyclic") {
                memcpy(h_boundary_rhoC + offset, &patchrho[0], patchSize*sizeof(double));
                offset += patchSize * 2;
            } else {
                memcpy(h_boundary_rhoC + offset, &patchrho[0], patchSize*sizeof(double));
                offset += patchSize;
            }
        }

        writeDoubleArrayToFile(h_boundary_rhoC, mesh_paras.num_boundary_surfaces, "d_rho_thermo_correctRho_boundary.host", compareCPUResults);

        // compare pEqn
        std::vector<double> h_internal_coeffs(mesh_paras.num_boundary_surfaces);
        std::vector<double> h_boundary_coeffs(mesh_paras.num_boundary_surfaces);

        offset = 0;
        for (int patchi = 0; patchi < mesh_paras.num_patches; patchi++)
        {
            const fvPatchScalarField& patchP = p.boundaryField()[patchi];
            int patchsize = mesh_paras.patch_size[patchi];
            const double* internal_coeff_ptr = &pEqn.internalCoeffs()[patchi][0];
            const double* boundary_coeff_ptr = &pEqn.boundaryCoeffs()[patchi][0];
            memcpy(h_internal_coeffs.data() + offset, internal_coeff_ptr, patchsize * sizeof(double));
            memcpy(h_boundary_coeffs.data() + offset, boundary_coeff_ptr, patchsize * sizeof(double));
            if (patchP.type() == "processor" || patchP.type() == "processorCyclic") offset += 2 * patchsize;
            else offset += patchsize;
        }
        if (!mpi_init_flag || rank == 0) {
            // pEqn_GPU.compareResult(&pEqn.lower()[0], &pEqn.upper()[0], &pEqn.diag()[0], &pEqn.source()[0], 
            //         h_internal_coeffs.data(), h_boundary_coeffs.data(), false);
        }

        writeDoubleArrayToFile(h_internal_coeffs.data(), mesh_paras.num_boundary_surfaces, "p_internalCoeffs.host", compareCPUResults);
        writeDoubleArrayToFile(h_boundary_coeffs.data(), mesh_paras.num_boundary_surfaces, "p_boundaryCoeffs.host", compareCPUResults);

        writeDoubleArrayToFile(&p[0], mesh_paras.num_cells, "p_p_solved.host", compareCPUResults);
        // abort();
    }

    // compare p
    double *h_boundary_p = new double[mesh_paras.num_boundary_surfaces];
    offset = 0;
    forAll(p.boundaryField(), patchi)
    {
        const fvPatchScalarField& patchP = p.boundaryField()[patchi];
        int patchsize = patchP.size();
        if (patchP.type() == "processor"
            || patchP.type() == "processorCyclic") {
            memcpy(h_boundary_p + offset, &patchP[0], patchsize * sizeof(double));
            scalarField patchPInternal = 
                    dynamic_cast<const processorFvPatchField<scalar>&>(patchP).patchInternalField()();
            memcpy(h_boundary_p + offset + patchsize, &patchPInternal[0], patchsize * sizeof(double));
            offset += patchsize * 2;
        } else {
            memcpy(h_boundary_p + offset, &patchP[0], patchsize * sizeof(double));
            offset += patchsize;
        }
    }
    // pEqn_GPU.correctP(&p[0], h_boundary_p);
    if (!mpi_init_flag || rank == 0) {
        //pEqn_GPU.comparep(&p[0], h_boundary_p, false);
    }
    writeDoubleArrayToFile(&h_boundary_p[0], mesh_paras.num_boundary_surfaces, "p_boundary_p_solved.host", compareCPUResults);
    delete h_boundary_p;

    // compare phi
    double *h_boundary_phi = new double[mesh_paras.num_boundary_surfaces];
    offset = 0;
    forAll(phi.boundaryField(), patchi)
    {
        const fvsPatchScalarField& patchFlux = phi.boundaryField()[patchi];
        int patchSize = patchFlux.size();
        memcpy(h_boundary_phi + offset, &patchFlux[0], patchSize*sizeof(double));
        if (patchFlux.type() == "processor"
            || patchFlux.type() == "processorCyclic") {
            memset(h_boundary_phi + offset + patchSize, 0, patchSize*sizeof(double));
            offset += 2 * patchSize;
        } else {
            offset += patchSize;
        }
    }
    if (!mpi_init_flag || rank == 0) {
        // pEqn_GPU.comparephi(&phi[0], h_boundary_phi, false);
    }
    writeDoubleArrayToFile(&phi[0], mesh_paras.num_surfaces, "p_phi_correct.host", compareCPUResults);
    writeDoubleArrayToFile(h_boundary_phi, mesh_paras.num_boundary_surfaces, "p_phi_correct_boundary.host", compareCPUResults);    
    delete h_boundary_phi;
#endif

// // thermo_GPU.correctPsipRho();

/* FOR DEEPFLAME-ACADEMIC */
#include "rhoEqn.H"
// #include "compressibleContinuityErrs.H" // TODO: implement this func in future

#if defined DEBUG_

    U = HbyA - rAU*fvc::grad(p);
    U.correctBoundaryConditions();
    K = 0.5*magSqr(U);

    // check U
    double *h_boundary_u_tmp_inp = new double[mesh_paras.num_boundary_surfaces * 3];
    offset = 0;
    forAll(U.boundaryField(), patchi)
    {
        const fvPatchVectorField& patchU = U.boundaryField()[patchi];
        int patchSize = patchU.size();

        if (patchU.type() == "processor"
            || patchU.type() == "processorCyclic") {
            memcpy(h_boundary_u_tmp_inp + 3*offset, &patchU[0][0], patchSize*sizeof(double)*3);
            vectorField patchUInternal = 
                    dynamic_cast<const processorFvPatchField<vector>&>(patchU).patchInternalField()();
            memcpy(h_boundary_u_tmp_inp + 3*offset + 3*patchSize, &patchUInternal[0][0], patchSize*sizeof(double)*3);
            offset += patchSize * 2;
        } else {
            memcpy(h_boundary_u_tmp_inp + 3*offset, &patchU[0][0], patchSize*sizeof(double)*3);
            offset += patchSize;
        }
    }
    double *U_newOrder = new double[mesh_paras.num_cells * 3]();
    permuteVector(U_newOrder, &U[0][0], mesh_paras.num_cells);
    writeDoubleArrayToFile(U_newOrder, mesh_paras.num_cells * 3, "p_correct_u.host", compareCPUResults);

    double *boundary_u_newOrder = new double[mesh_paras.num_boundary_surfaces * 3]();
    permuteVector(boundary_u_newOrder, &h_boundary_u_tmp_inp[0], mesh_paras.num_boundary_surfaces);
    writeDoubleArrayToFile(boundary_u_newOrder, mesh_paras.num_boundary_surfaces * 3, "p_correct_boundary_u.host", compareCPUResults);
    
    if (!mpi_init_flag || rank == 0) {
        // pEqn_GPU.compareU(&U[0][0], h_boundary_u_tmp_inp, false);
    }
    delete h_boundary_u_tmp_inp;
#endif

#if defined DEBUG_ 
    dpdt = fvc::ddt(p);
    writeDoubleArrayToFile(&dpdt[0], mesh_paras.num_cells, "p_dpdt.host", compareCPUResults);
    // pEqn_GPU.comparedpdt(&dpdt[0], false);
#endif

// #undef CPUSolver_
// #define GPUSolverNew_
