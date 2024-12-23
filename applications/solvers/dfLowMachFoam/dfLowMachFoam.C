/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2019 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    rhoPimpleFoam

Description
    Transient solver for turbulent flow of compressible fluids for HVAC and
    similar applications, with optional mesh motion and mesh topology changes.

    Uses the flexible PIMPLE (PISO-SIMPLE) solution for time-resolved and
    pseudo-transient simulations.

\*---------------------------------------------------------------------------*/
#include "stdlib.h"
#include "dfChemistryModel.H"
#include "CanteraMixture.H"
// #include "hePsiThermo.H"
#include "heRhoThermo.H"
#include "csrMatrix.H"
#include "ellMatrix.H"
// #include "GPUTestRef.H"

#ifdef USE_PYTORCH
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> //used to convert
#endif

#ifdef USE_LIBTORCH
#include <torch/script.h>
#include "DNNInferencer.H"
#endif

#include "fvCFD.H"
#include "fluidThermo.H"
#include "turbulentFluidThermoModel.H"
#include "pimpleControl.H"
#include "pressureControl.H"
#include "localEulerDdtScheme.H"
#include "fvcSmooth.H"
#include "PstreamGlobals.H"
#include "basicThermo.H"
#include "CombustionModel.H"

/* For DeepFlame_Academic */
#include "dfgamg.h"
#include "dfacademic.h"

#define GPUSolverNew_
// #define TIME
// #define DEBUG_
// #define SHOW_MEMINFO
// #define OPENCC

#define iscsr // true -> csr, false -> ell
#define DEBUG_TRACE fprintf(stderr, "myRank[%d] %s %d\n", myRank, __FILE__, __LINE__);


#ifdef GPUSolverNew_
    #include "GenFvMatrix.H"
    // #include "AmgXSolver.H"
    // #include "dfNcclBase.H"
    // #include "dfThermo.H"
    // #include "dfChemistrySolver.H"
    #include <cuda_runtime.h>
    #include <thread>

    #include "processorFvPatchField.H"
    #include "cyclicFvPatchField.H"
    #include "processorCyclicFvPatchField.H"
    #include "totalPressureFvPatchScalarField.H"
    #include "epsilonWallFunctionFvPatchScalarField.H"
    #include "wedgeFvPatch.H"
    #include "wedgeFvPatchField.H"
    #include "createGPUSolver.H"

    #include "upwind.H"
    #include "CanteraMixture.H"
    #include "multivariateGaussConvectionScheme.H"
    #include "limitedSurfaceInterpolationScheme.H"
    #include "nearWallDist.H"
#else
    #include "processorFvPatchField.H"
    #include "cyclicFvPatchField.H"
    #include "multivariateGaussConvectionScheme.H"
    #include "limitedSurfaceInterpolationScheme.H"
    int myRank = -1;
    int mpi_init_flag = 0;
#endif

#ifdef OPENCC
    #include "opencc.h"
#endif

int offset;

#ifdef TIME
    #define TICK_START \
        start_new = std::clock(); 
    #define TICK_STOP(prefix) \
        stop_new = std::clock(); \
        Foam::Info << #prefix << " time = " << double(stop_new - start_new) / double(CLOCKS_PER_SEC) << " s" << Foam::endl;
#else
    #define TICK_START
    #define TICK_STOP(prefix)
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
#ifdef USE_PYTORCH
    pybind11::scoped_interpreter guard{};//start python interpreter
#endif
    #include "postProcess.H"

    // unsigned int flags = 0;
    // checkCudaErrors(cudaGetDeviceFlags(&flags));
    // flags |= cudaDeviceScheduleYield;
    // checkCudaErrors(cudaSetDeviceFlags(flags));

    // #include "setRootCaseLists.H"
    #include "listOptions.H"
    #include "setRootCase2.H"
    #include "listOutput.H"

    #include "createTime.H"
    #include "createMesh.H"
    #include "createDyMControls.H"
    #include "initContinuityErrs.H"
    #include "createFields.H"
    #include "createRhoUfIfPresent.H"

    double time_monitor_init = 0;

    double time_monitor_other = 0;
    double time_monitor_rho = 0;
    double time_monitor_U = 0;
    double time_monitor_Y = 0;
    double time_monitor_E = 0;
    double time_monitor_p = 0;
    double time_monitor_chemistry_correctThermo = 0;
    double time_monitor_turbulence_correct = 0;
    double time_monitor_chem = 0; // combustion correct

    double time_monitor_rhoEqn = 0;
    double time_monitor_rhoEqn_mtxAssembly = 0;
    double time_monitor_rhoEqn_mtxAssembly_CPU_prepare = 0;
    double time_monitor_rhoEqn_mtxAssembly_GPU_run = 0;
    double time_monitor_rhoEqn_solve = 0;
    double time_monitor_rhoEqn_correctBC = 0;

    double time_monitor_UEqn = 0;
    double time_monitor_UEqn_mtxAssembly = 0;
    double time_monitor_UEqn_mtxAssembly_CPU_prepare = 0;
    double time_monitor_UEqn_mtxAssembly_GPU_run = 0;
    double time_monitor_UEqn_solve = 0;
    double time_monitor_UEqn_correctBC = 0;
    double time_monitor_UEqn_H = 0;
    double time_monitor_UEqn_H_GPU_run = 0;
    double time_monitor_UEqn_H_correctBC = 0;
    double time_monitor_UEqn_A = 0;
    double time_monitor_UEqn_A_GPU_run = 0;
    double time_monitor_UEqn_A_correctBC = 0;

    double time_monitor_YEqn = 0;
    double time_monitor_YEqn_mtxAssembly = 0;
    double time_monitor_YEqn_mtxAssembly_CPU_prepare = 0;
    double time_monitor_YEqn_mtxAssembly_GPU_run = 0;
    double time_monitor_YEqn_solve = 0;
    double time_monitor_YEqn_correctBC = 0;

    double time_monitor_EEqn = 0;
    double time_monitor_EEqn_mtxAssembly = 0;
    double time_monitor_EEqn_mtxAssembly_CPU_prepare = 0;
    double time_monitor_EEqn_mtxAssembly_GPU_prepare = 0;
    double time_monitor_EEqn_mtxAssembly_GPU_run = 0;
    double time_monitor_EEqn_solve = 0;
    double time_monitor_EEqn_correctBC = 0;

    double time_monitor_pEqn = 0;
    double time_monitor_pEqn_solve = 0;

    label timeIndex = 0;
    clock_t start, end, start1, end1, start2, end2;
    clock_t start_new, stop_new;
    double time_new = 0;

    turbulence->validate();

    if (!LTS)
    {
        #include "compressibleCourantNo.H"
        #include "setInitialDeltaT.H"
    }

#ifndef GPUSolverNew_   
    #ifdef OPENCC
        #include "createFields_GPU.H"
    #endif   
#endif 
    start1 = std::clock();

#ifdef GPUSolverNew_

    int mpi_init_flag;
    checkMpiErrors(MPI_Initialized(&mpi_init_flag));
    initNccl();

#ifdef OPENCC
    #include "createFields_GPU.H"
#endif  

    std::cout << "                                                          " << std::endl;
    std::cout << "==========================================================" << std::endl;
    std::cout << "              SETDATA FOR DEEPFLAME-ACADEMIC              " << std::endl;
    std::cout << "                                                          " << std::endl; 
    
    mesh_info_para mesh_paras;    
    init_data_para init_data;
    bool compareCPUResults = compareResults;
    bool doCorrectBCsCPU = true;

    // DF-A: CREATE MESHBASE 
    createGPUBaseInput(mesh_paras, init_data, CanteraTorchProperties, mesh, Y); 

    // DF-A: CREATE DFDATABASE 
    createGPUUEqn(mesh_paras, init_data, CanteraTorchProperties, U);
    createGPUYEqn(mesh_paras, init_data, CanteraTorchProperties, Y, inertIndex);
    createGPUEEqn(mesh_paras, init_data, CanteraTorchProperties, thermo.he(), K);
    createGPUpEqn(mesh_paras, init_data, CanteraTorchProperties, p, U);
    createGPURhoEqn(mesh_paras, init_data, rho, phi);
    createGPUThermo(mesh_paras, init_data, CanteraTorchProperties, T, thermo.he(), 
                    psi, thermo.alpha(), thermo.mu(), K, dpdt, chemistry);

    // DF-A: MALLOC & MEMCPY-H2D "MESHBASE" 
    set_mesh_info(mesh_paras, compareCPUResults);

    // DF-A: MALLOC & MEMCPY-H2D "DFDATABASE" 
    set_data_info(init_data);

    // DF-A: SET THERMO-CONSTANT-COEFFS
    string mechanismFile = CanteraTorchProperties.lookupOrDefault("CanteraMechanismFile", string(""));
    const char* mechanism_file = mechanismFile.c_str();
    set_thermo_const_coeffs(mechanism_file);
    
    // DF-A: SET SPARSE-FORMAT-MAPS
    createGPUSparseFormat(mesh_paras, mesh);

    IOdictionary fvSolutionDict
    (
        IOobject
        (
            "fvSolution",          // Dictionary name
            runTime.system(),      // Location within case
            Y[0].mesh(),           // Mesh reference
            IOobject::MUST_READ,   // Read if present
            IOobject::NO_WRITE     // Do not write to disk
        )
    );
    dictionary solversDict = fvSolutionDict.subDict("solvers");

    // DF-A: SET LINEAR SOLVER CONFIGS
    bool setRAS = (turbName == "RAS");
    createGPUSolver(solversDict, CanteraTorchProperties, rho, U, p, Y, thermo.he(), setRAS); 

    // DF-A: SET CHEMISTRY SOLVER CONFIGS
    if (chemistry->ifChemstry())
    {
        createChemistrySolver(4096, 500); // (batch_size, unReactT)
    }

    IOdictionary fvSchemesDict
    (
        IOobject
        (
            "fvSchemes",          // Dictionary name
            runTime.system(),      // Location within case
            Y[0].mesh(),           // Mesh reference
            IOobject::MUST_READ,   // Read if present
            IOobject::NO_WRITE     // Do not write to disk
        )
    );
    fvSchemes_para schemes_para;
    createGPUSchemesInput(fvSchemesDict, schemes_para);

    IOdictionary turbulenceDict
    (
        IOobject
        (
            "turbulenceProperties", // Dictionary name
            runTime.constant(),     // Location within case
            Y[0].mesh(),            // Mesh reference
            IOobject::MUST_READ,    // Read if present
            IOobject::NO_WRITE      // Do not write to disk
        )
    );
    createTurbulenceInput(turbulenceDict, mesh_paras, turbulence->nut(), turbulence->alphat(), turbulence->k(), turbulence->epsilon());

    IOdictionary combustionDict
    (
        IOobject
        (
            "combustionProperties", // Dictionary name
            runTime.constant(),     // Location within case
            Y[0].mesh(),            // Mesh reference
            IOobject::MUST_READ,    // Read if present
            IOobject::NO_WRITE      // Do not write to disk
        )
    );
    createCombustionInput(combustionDict, mesh_paras);


    std::cout << "                                                          " << std::endl;
    std::cout << "!!!  All data has been set done for deepflame academic.   " << std::endl; 
    std::cout << "==========================================================" << std::endl;

    DEBUG_TRACE;

#endif

    end1 = std::clock();
    time_monitor_init += double(end1 - start1) / double(CLOCKS_PER_SEC);

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nStarting time loop\n" << endl;

    while (runTime.run())
    {
        timeIndex ++;

        #include "readDyMControls.H"

        if (LTS)
        {
            #include "setRDeltaT.H"
        }
        else
        {
            #include "compressibleCourantNo.H"
            #include "setDeltaT.H"
        }

        runTime++;

        Info<< "Time = " << runTime.timeName() << nl << endl;
        
        // DF-A: STORE PRE-TIMESTPE FIELDS IN ACADEMIC
        #ifdef GPUSolverNew_
        double rDeltaT = 1.0/Y[0].mesh().time().deltaTValue();
        preTimeStep(rDeltaT);   
        #endif

        clock_t loop_start = std::clock();
        // --- Pressure-velocity PIMPLE corrector loop
        while (pimple.loop())
        {
            start = std::clock();
            if (splitting)
            {
                #include "YEqn_RR.H"
            }
            if (pimple.firstPimpleIter() || moveMeshOuterCorrectors)
            {
                // Store momentum to set rhoUf for introduced faces.
                autoPtr<volVectorField> rhoU;
                if (rhoUf.valid())
                {
                    rhoU = new volVectorField("rhoU", rho*U);
                }
            }
            end = std::clock();
            time_monitor_other += double(end - start) / double(CLOCKS_PER_SEC);

            start = std::clock();
            if (pimple.firstPimpleIter() && !pimple.simpleRho())
            {
                #include "rhoEqn.H"
            }
            end = std::clock();
            time_monitor_rho += double(end - start) / double(CLOCKS_PER_SEC);        

            start = std::clock();
            #include "UEqn.H"
            end = std::clock();
            time_monitor_U += double(end - start) / double(CLOCKS_PER_SEC);

            if(combModelName!="ESF" && combModelName!="flareFGM" && combModelName!="DeePFGM")
            {
                start = std::clock();
                #include "YEqn.H"
                end = std::clock();
                time_monitor_Y += double(end - start) / double(CLOCKS_PER_SEC);

                start = std::clock();
                #include "EEqn.H"
                end = std::clock();
                time_monitor_E += double(end - start) / double(CLOCKS_PER_SEC);

                start = std::clock();
                #ifdef GPUSolverNew_
                    #if defined DEBUG_
                    chemistry->correctThermo(); // reference debug
                    const volScalarField& mu = thermo.mu();
                    const volScalarField& alpha = thermo.alpha();
                    writeDoubleArrayToFile(&T[0], mesh_paras.num_cells, "thermo_d_T.host", compareCPUResults);
                    writeDoubleArrayToFile(&psi[0], mesh_paras.num_cells, "thermo_d_thermo_psi.host", compareCPUResults);
                    writeDoubleArrayToFile(&thermo.rho()()[0], mesh_paras.num_cells, "thermo_d_rho.host", compareCPUResults);
                    writeDoubleArrayToFile(&mu[0], mesh_paras.num_cells, "thermo_d_mu.host", compareCPUResults);
                    writeDoubleArrayToFile(&alpha[0], mesh_paras.num_cells, "thermo_d_thermo_alpha.host", compareCPUResults);
                    writeDoubleArrayToFile(&chemistry->rhoD(0)[0], mesh_paras.num_cells, "thermo_d_thermo_rhoD.host", compareCPUResults);  

                    double *h_boundary_T_tmp = new double[mesh_paras.num_boundary_surfaces];
                    double *h_boundary_psi_tmp = new double[mesh_paras.num_boundary_surfaces];
                    double *h_boundary_thermorho_tmp = new double[mesh_paras.num_boundary_surfaces];
                    double *h_boundary_mu_tmp = new double[mesh_paras.num_boundary_surfaces];
                    double *h_boundary_alpha_tmp = new double[mesh_paras.num_boundary_surfaces];
                    double *h_boundary_thermorhoD_tmp = new double[mesh_paras.num_boundary_surfaces];
                    offset = 0;
                    forAll(T.boundaryField(), patchi) {
                        fvPatchScalarField& patchT = const_cast<fvPatchScalarField&>(T.boundaryField()[patchi]);
                        fvPatchScalarField& patchpsi = const_cast<fvPatchScalarField&>(psi.boundaryField()[patchi]);
                        fvPatchScalarField& patchrho = const_cast<fvPatchScalarField&>(thermo.rho()().boundaryField()[patchi]);
                        fvPatchScalarField& patchmu = const_cast<fvPatchScalarField&>(mu.boundaryField()[patchi]);
                        fvPatchScalarField& patchalpha = const_cast<fvPatchScalarField&>(alpha.boundaryField()[patchi]);
                        fvPatchScalarField& patchrhoD = const_cast<fvPatchScalarField&>(chemistry->rhoD(0).boundaryField()[patchi]);

                        int patchsize = patchT.size();

                        memcpy(h_boundary_T_tmp + offset, &patchT[0], patchsize * sizeof(double));
                        memcpy(h_boundary_psi_tmp + offset, &patchpsi[0], patchsize * sizeof(double));
                        memcpy(h_boundary_thermorho_tmp + offset, &patchrho[0], patchsize * sizeof(double));
                        memcpy(h_boundary_mu_tmp + offset, &patchmu[0], patchsize * sizeof(double));
                        memcpy(h_boundary_alpha_tmp + offset, &patchalpha[0], patchsize * sizeof(double));
                        memcpy(h_boundary_thermorhoD_tmp + offset, &patchrhoD[0], patchsize * sizeof(double));

                        if (patchT.type() == "processor") {
                            offset += patchsize * 2;
                        } else {
                            offset += patchsize;
                        }
                    }
                    writeDoubleArrayToFile(h_boundary_T_tmp, mesh_paras.num_boundary_surfaces, "thermo_d_T_boundary.host", compareCPUResults);
                    writeDoubleArrayToFile(h_boundary_psi_tmp, mesh_paras.num_boundary_surfaces, "thermo_d_thermo_psi_boundary.host", compareCPUResults);
                    writeDoubleArrayToFile(h_boundary_thermorho_tmp, mesh_paras.num_boundary_surfaces, "thermo_d_rho_boundary.host", compareCPUResults);

                    writeDoubleArrayToFile(h_boundary_mu_tmp, mesh_paras.num_boundary_surfaces, "thermo_d_mu_boundary.host", compareCPUResults);
                    writeDoubleArrayToFile(h_boundary_alpha_tmp, mesh_paras.num_boundary_surfaces, "thermo_d_thermo_alpha_boundary.host", compareCPUResults);
                    writeDoubleArrayToFile(h_boundary_thermorhoD_tmp, mesh_paras.num_boundary_surfaces, "thermo_d_thermo_rhoD_boundary.host", compareCPUResults);

                    delete h_boundary_T_tmp;   
                    delete h_boundary_psi_tmp; 
                    delete h_boundary_thermorho_tmp; 
                    delete h_boundary_mu_tmp; 
                    delete h_boundary_alpha_tmp; 
                    delete h_boundary_thermorhoD_tmp; 

                    #endif
                #else
                    chemistry->correctThermo();
                #endif
                end = std::clock();
                time_monitor_chemistry_correctThermo += double(end - start) / double(CLOCKS_PER_SEC);
            }
            else
            {
                combustion->correct();
            }

            Info<< "min/max(T) = " << min(T).value() << ", " << max(T).value() << endl;

            // --- Pressure corrector loop

            start = std::clock();
            int num_pimple_loop = pimple.nCorrPimple();
            while (pimple.correct())
            {
                if (pimple.consistent())
                {
                    // #include "pcEqn.H"
                }
                else
                {

                #ifdef GPUSolverNew_
                    #include "pEqn_GPU.H"
                #else
                    #include "pEqn_CPU.H"
                #endif
                
                }
                num_pimple_loop --;
            }
            end = std::clock();
            time_monitor_p += double(end - start) / double(CLOCKS_PER_SEC);

            start = std::clock();
            if (pimple.turbCorr())
            {
                #ifdef GPUSolverNew_

                    correctTurbulence();

                    #if defined DEBUG_
                    turbulence->correct();

                    writeDoubleArrayToFile(&turbulence->k()()[0], mesh_paras.num_cells, "turb_k_correct.host", compareCPUResults);
                    writeDoubleArrayToFile(&turbulence->epsilon()()[0], mesh_paras.num_cells, "turb_epsilon_correct.host", compareCPUResults);

                    writeDoubleArrayToFile(&turbulence->nut()()[0], mesh_paras.num_cells, "turb_kEpsilon_nut.host", compareCPUResults);
                    writeDoubleArrayToFile(&turbulence->mut()()[0], mesh_paras.num_cells, "turb_kEpsilon_mut.host", compareCPUResults);
                    writeDoubleArrayToFile(&turbulence->alphat()()[0], mesh_paras.num_cells, "turb_kEpsilon_alphat.host", compareCPUResults);

                    int offset = 0;
                    double *h_boundary_nut = new double[mesh_paras.num_boundary_surfaces]();
                    double *h_boundary_alphat = new double[mesh_paras.num_boundary_surfaces]();
                    double *h_boundary_mut = new double[mesh_paras.num_boundary_surfaces]();
                    double *h_boundary_k = new double[mesh_paras.num_boundary_surfaces]();
                    double *h_boundary_epsilon = new double[mesh_paras.num_boundary_surfaces]();
                    forAll(turbulence->nut()().boundaryField(), patchi)
                    {
                        const fvsPatchScalarField& patchFlux = phi.boundaryField()[patchi];
                        int patchsize = mesh_paras.patch_size[patchi];

                        Field<scalar> bouNut = turbulence->nut()().boundaryField()[patchi];
                        Field<scalar> bouAlphat = turbulence->alphat()().boundaryField()[patchi]; 
                        Field<scalar> bouMut = turbulence->mut()().boundaryField()[patchi]; 
                        Field<scalar> bouK = turbulence->k()().boundaryField()[patchi]; 
                        Field<scalar> bouEps = turbulence->epsilon()().boundaryField()[patchi]; 

                        memcpy(h_boundary_nut + offset, bouNut.data(), patchsize * sizeof(double));
                        memcpy(h_boundary_alphat + offset, bouAlphat.data(), patchsize * sizeof(double));
                        memcpy(h_boundary_mut + offset, bouMut.data(), patchsize * sizeof(double));
                        memcpy(h_boundary_k + offset, bouK.data(), patchsize * sizeof(double));
                        memcpy(h_boundary_epsilon + offset, bouEps.data(), patchsize * sizeof(double));

                        if (patchFlux.type() == "processor" || patchFlux.type() == "processorCyclic") offset += 2 * patchsize;
                        else offset += patchsize;
                    }
                    writeDoubleArrayToFile(h_boundary_nut, mesh_paras.num_boundary_surfaces, "turb_kEpsilon_boundary_nut.host", compareCPUResults);
                    writeDoubleArrayToFile(h_boundary_alphat, mesh_paras.num_boundary_surfaces, "turb_kEpsilon_boundary_alphat.host", compareCPUResults);
                    writeDoubleArrayToFile(h_boundary_mut, mesh_paras.num_boundary_surfaces, "turb_kEpsilon_boundary_mut.host", compareCPUResults);
                    writeDoubleArrayToFile(h_boundary_k, mesh_paras.num_boundary_surfaces, "turb_kEpsilon_boundary_k.host", compareCPUResults);
                    writeDoubleArrayToFile(h_boundary_epsilon, mesh_paras.num_boundary_surfaces, "turb_kEpsilon_boundary_epsilon.host", compareCPUResults);

                    #endif

                #else
                    turbulence->correct();
                #endif
            }
            end = std::clock();
            time_monitor_turbulence_correct += double(end - start) / double(CLOCKS_PER_SEC);
        }
        clock_t loop_end = std::clock();
        double loop_time = double(loop_end - loop_start) / double(CLOCKS_PER_SEC);

        #ifdef GPUSolverNew_
            /* FOR DEEPFLAME-ACADEMIC */
            updateRho();  // thermo_GPU.updateRho();   
        #if defined DEBUG_
            rho = thermo.rho();
        #endif
        #else
            rho = thermo.rho();
        #endif

        #ifdef GPUSolverNew_
            copyGPUResults2Host(mesh_paras, U, T, Y, rho, phi);
        #endif

        runTime.write();

        Info<< "========Write Min/Max Info========"<< endl;
        Info<< "min/max(rho) = " << min(rho).value() << ", " << max(rho).value() << endl;
        Info<< "min/max(U) = " << min(U).value() << ", " << max(U).value() << endl;
        forAll(Y, i)
        {
            Info<< "min/max(Y) " << i << " = " << min(Y[i]).value() << ", " << max(Y[i]).value() << endl;
        } 
        Info<< "min/max(E) = " << min(thermo.he()).value() << ", " << max(thermo.he()).value() << endl;
        Info<< "min/max(T) = " << min(T).value() << ", " << max(T).value() << endl;
        Info<< "min/max(p) = " << min(p).value() << ", " << max(p).value() << endl;
        Info<< "min/max(k) = " << min(turbulence->k()()).value() << ", " << max(turbulence->k()()).value() << endl;
        Info<< "min/max(epsilon) = " << min(turbulence->epsilon()()).value() << ", " << max(turbulence->epsilon()()).value() << endl;

        Info<< "========Time Spent in diffenet parts========"<< endl;
        Info<< "loop Time                    = " << loop_time << " s" << endl;
        Info<< "other Time                   = " << time_monitor_other << " s" << endl;
        Info<< "rho Equations                = " << time_monitor_rho << " s" << endl;
        Info<< "U Equations                  = " << time_monitor_U << " s" << endl;
        Info<< "Y Equations                  = " << time_monitor_Y - time_monitor_chem << " s" << endl;
        Info<< "E Equations                  = " << time_monitor_E << " s" << endl;
        Info<< "p Equations                  = " << time_monitor_p << " s" << endl;
        Info<< "chemistry correctThermo      = " << time_monitor_chemistry_correctThermo << " s" << endl;
        Info<< "turbulence correct           = " << time_monitor_turbulence_correct << " s" << endl;
        Info<< "combustion correct(in Y)     = " << time_monitor_chem << " s" << endl;
        Info<< "percentage of chemistry      = " << time_monitor_chem / loop_time * 100 << " %" << endl;
        Info<< "percentage of rho/U/Y/E      = " << (time_monitor_E + time_monitor_Y + time_monitor_U + time_monitor_rho - time_monitor_chem) / loop_time * 100 << " %" << endl;


        Info<< "========Time details of each equation======="<< endl;

        Info<< "rhoEqn Time                  = " << time_monitor_rhoEqn << " s" << endl;
        Info<< "rhoEqn assamble              = " << time_monitor_rhoEqn_mtxAssembly << " s" << endl;
        Info<< "rhoEqn assamble(CPU prepare) = " << time_monitor_rhoEqn_mtxAssembly_CPU_prepare << " s" << endl;
        Info<< "rhoEqn assamble(GPU run)     = " << time_monitor_rhoEqn_mtxAssembly_GPU_run << " s" << endl;
        Info<< "rhoEqn solve                 = " << time_monitor_rhoEqn_solve << " s" << endl;
        Info<< "rhoEqn correct boundary      = " << time_monitor_rhoEqn_correctBC << " s" << endl;

        Info<< "UEqn Time                    = " << time_monitor_UEqn << " s" << endl;
        Info<< "UEqn assamble                = " << time_monitor_UEqn_mtxAssembly << " s" << endl;
        Info<< "UEqn assamble(CPU prepare)   = " << time_monitor_UEqn_mtxAssembly_CPU_prepare << " s" << endl;
        Info<< "UEqn assamble(GPU run)       = " << time_monitor_UEqn_mtxAssembly_GPU_run << " s" << endl;
        Info<< "UEqn solve                   = " << time_monitor_UEqn_solve << " s" << endl;
        Info<< "UEqn correct boundary        = " << time_monitor_UEqn_correctBC << " s" << endl;
        Info<< "UEqn H                       = " << time_monitor_UEqn_H << " s" << endl;
        Info<< "UEqn H(GPU run)              = " << time_monitor_UEqn_H_GPU_run << " s" << endl;
        Info<< "UEqn H(correct boundary)     = " << time_monitor_UEqn_H_correctBC << " s" << endl;
        Info<< "UEqn A                       = " << time_monitor_UEqn_A << " s" << endl;
        Info<< "UEqn A(GPU run)              = " << time_monitor_UEqn_A_GPU_run << " s" << endl;
        Info<< "UEqn A(correct boundary)     = " << time_monitor_UEqn_A_correctBC << " s" << endl;

        Info<< "YEqn Time                    = " << time_monitor_YEqn << " s" << endl;
        Info<< "YEqn assamble                = " << time_monitor_YEqn_mtxAssembly << " s" << endl;
        Info<< "YEqn assamble(CPU prepare)   = " << time_monitor_YEqn_mtxAssembly_CPU_prepare << " s" << endl;
        Info<< "YEqn assamble(GPU run)       = " << time_monitor_YEqn_mtxAssembly_GPU_run << " s" << endl;
        Info<< "YEqn solve                   = " << time_monitor_YEqn_solve << " s" << endl;
        Info<< "YEqn correct boundary        = " << time_monitor_YEqn_correctBC << " s" << endl;

        Info<< "EEqn Time                    = " << time_monitor_EEqn << " s" << endl;
        Info<< "EEqn assamble                = " << time_monitor_EEqn_mtxAssembly << " s" << endl;
        Info<< "EEqn assamble(CPU prepare)   = " << time_monitor_EEqn_mtxAssembly_CPU_prepare << " s" << endl;
        Info<< "EEqn assamble(GPU prepare)   = " << time_monitor_EEqn_mtxAssembly_GPU_prepare << " s" << endl;
        Info<< "EEqn assamble(GPU run)       = " << time_monitor_EEqn_mtxAssembly_GPU_run << " s" << endl;
        Info<< "EEqn solve                   = " << time_monitor_EEqn_solve << " s" << endl;
        Info<< "EEqn correct boundary        = " << time_monitor_EEqn_correctBC << " s" << endl;

        Info<< "pEqn Time                    = " << time_monitor_pEqn << " s" << endl;
        Info<< "pEqn Time solve              = " << time_monitor_pEqn_solve << " s" << endl;

        Info<< "============================================"<<nl<< endl;

        Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
            << "  ClockTime = " << runTime.elapsedClockTime() << " s" << endl;

#ifdef GPUSolverNew_
#ifdef SHOW_MEMINFO
	int rank = -1;
	if (mpi_init_flag) {
    	    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	}
	if (!mpi_init_flag || rank == 0) {
            fprintf(stderr, "show memory info...\n");
            //usleep(1 * 1000 * 1000);
	    system("nvidia-smi");
	}
#endif
#endif
        time_monitor_other = 0;
        time_monitor_rho = 0;
        time_monitor_U = 0;
        time_monitor_Y = 0;
        time_monitor_E = 0;
        time_monitor_p = 0;
        time_monitor_chemistry_correctThermo = 0;
        time_monitor_turbulence_correct = 0;
        time_monitor_chem = 0;

        time_monitor_rhoEqn = 0;
        time_monitor_rhoEqn_mtxAssembly = 0;
        time_monitor_rhoEqn_mtxAssembly_CPU_prepare = 0;
        time_monitor_rhoEqn_mtxAssembly_GPU_run = 0;
        time_monitor_rhoEqn_solve = 0;
        time_monitor_rhoEqn_correctBC = 0;

        time_monitor_UEqn = 0;
        time_monitor_UEqn_mtxAssembly = 0;
        time_monitor_UEqn_mtxAssembly_CPU_prepare = 0;
        time_monitor_UEqn_mtxAssembly_GPU_run = 0;
        time_monitor_UEqn_solve = 0;
        time_monitor_UEqn_correctBC = 0;
        time_monitor_UEqn_H = 0;
        time_monitor_UEqn_H_GPU_run = 0;
        time_monitor_UEqn_H_correctBC = 0;
        time_monitor_UEqn_A = 0;
        time_monitor_UEqn_A_GPU_run = 0;
        time_monitor_UEqn_A_correctBC = 0;

        time_monitor_YEqn = 0;
        time_monitor_YEqn_mtxAssembly = 0;
        time_monitor_YEqn_mtxAssembly_CPU_prepare = 0;
        time_monitor_YEqn_mtxAssembly_GPU_run = 0;
        time_monitor_YEqn_solve = 0;
        time_monitor_YEqn_correctBC = 0;

        time_monitor_EEqn = 0;
        time_monitor_EEqn_mtxAssembly = 0;
        time_monitor_EEqn_mtxAssembly_CPU_prepare = 0;
        time_monitor_EEqn_mtxAssembly_GPU_prepare = 0;
        time_monitor_EEqn_mtxAssembly_GPU_run = 0;
        time_monitor_EEqn_solve = 0;
        time_monitor_EEqn_correctBC = 0;

        time_monitor_pEqn = 0;
        time_monitor_pEqn_solve = 0;

#ifdef USE_PYTORCH
        if (log_ && torch_)
        {
            Info<< "    allsolveTime = " << chemistry->time_allsolve() << " s"
            << "    submasterTime = " << chemistry->time_submaster() << " s" << nl
            << "    sendProblemTime = " << chemistry->time_sendProblem() << " s"
            << "    recvProblemTime = " << chemistry->time_RecvProblem() << " s"
            << "    sendRecvSolutionTime = " << chemistry->time_sendRecvSolution() << " s" << nl
            << "    getDNNinputsTime = " << chemistry->time_getDNNinputs() << " s"
            << "    DNNinferenceTime = " << chemistry->time_DNNinference() << " s"
            << "    updateSolutionBufferTime = " << chemistry->time_updateSolutionBuffer() << " s" << nl
            << "    vec2ndarrayTime = " << chemistry->time_vec2ndarray() << " s"
            << "    pythonTime = " << chemistry->time_python() << " s"<< nl << endl;
        }
#endif
#ifdef USE_LIBTORCH
        if (log_ && torch_)
        {
            Info<< "    allsolveTime = " << chemistry->time_allsolve() << " s"
            << "    submasterTime = " << chemistry->time_submaster() << " s" << nl
            << "    sendProblemTime = " << chemistry->time_sendProblem() << " s"
            << "    recvProblemTime = " << chemistry->time_RecvProblem() << " s"
            << "    sendRecvSolutionTime = " << chemistry->time_sendRecvSolution() << " s" << nl
            << "    DNNinferenceTime = " << chemistry->time_DNNinference() << " s"
            << "    updateSolutionBufferTime = " << chemistry->time_updateSolutionBuffer() << " s" << nl;
        }
#endif
    }

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
