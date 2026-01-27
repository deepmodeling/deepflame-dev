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

#include "common.H"

#ifdef USE_PYTORCH
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> //used to convert
#endif

#ifdef USE_LIBTORCH
#include <torch/script.h>
#include "DNNInferencer.H"
#endif

#ifdef ODE_GPU_SOLVER
#include "opencc.h"
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

// renumber
#include "dynamicFvMesh.H"
#include "argList.H"
#include "IOobjectList.H"
#include "fvMesh.H"
#include "polyTopoChange.H"
#include "ReadFields.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "SortableList.H"
#include "decompositionMethod.H"
#include "renumberMethod.H"
#include "zeroGradientFvPatchFields.H"
#include "CuthillMcKeeRenumber.H"
#include "geomRenumber.H"
#include "fvMeshSubset.H"
#include "cellSet.H"
#include "faceSet.H"
#include "pointSet.H"
#include <omp.h>
#include <typeinfo>
#include "renumberMeshFuncs.H"

// #define GPUSolverNew_
// #define TIME
// #define DEBUG_
// #define SHOW_MEMINFO

#ifdef GPUSolverNew_
    #include "dfMatrixDataBase.H"
    #include "AmgXSolver.H"
    #include "dfUEqn.H"
    #include "dfYEqn.H"
    #include "dfRhoEqn.H"
    #include "dfEEqn.H"
    #include "dfpEqn.H"
    #include "dfMatrixOpBase.H"
    #include "dfNcclBase.H"
    #include "dfThermo.H"
    #include "dfChemistrySolver.H"
    #include <cuda_runtime.h>
    #include <thread>

    #include "processorFvPatchField.H"
    #include "cyclicFvPatchField.H"
    #include "processorCyclicFvPatchField.H"
    #include "createGPUSolver.H"

    #include "upwind.H"
    #include "CanteraMixture.H"
    #include "multivariateGaussConvectionScheme.H"
    #include "limitedSurfaceInterpolationScheme.H"
#else
    #include "processorFvPatchField.H"
    #include "cyclicFvPatchField.H"
    #include "multivariateGaussConvectionScheme.H"
    #include "limitedSurfaceInterpolationScheme.H"
    int myRank = -1;
    int mpi_init_flag = 0;
#endif

int offset;

#ifdef TIME
    #define TICK_START \
        start_new = MPI_Wtime(); 
    #define TICK_STOP(prefix) \
        stop_new = MPI_Wtime(); \
        Foam::Info << #prefix << " time = " << double(stop_new - start_new) << " s" << Foam::endl;
#else
    #define TICK_START
    #define TICK_STOP(prefix)
#endif

#include "env.H"
#include "csrPattern.H"
#include "BCSRPattern.H"
#include "dfMatrix.H"
#include "GenFvMatrix.H"
#include "MeshSchedule.H"

#define USE_DF_MATRIX
// #define OPT_GenMatrix_E
// #define OPT_GenMatrix_E_check

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
    // #include "createMesh.H"
    #include "createDynamicFvMesh.H"
    #include "createDyMControls.H"
    #include "initContinuityErrs.H"
    #include "createFields.H"
    #include "createRhoUfIfPresent.H"

    #ifdef ODE_GPU_SOLVER
    #include "createFields_GPU.H"
    #endif

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
    double start, end, start1, end1, start2, end2;
    double start_new, stop_new;
    double time_new = 0;

    turbulence->validate();

    if (!LTS)
    {
        #include "compressibleCourantNo.H"
        #include "setInitialDeltaT.H"
    }

    start1 = MPI_Wtime();
#ifdef GPUSolverNew_
    int mpi_init_flag;
    checkMpiErrors(MPI_Initialized(&mpi_init_flag));
    if(mpi_init_flag) {
        initNccl();
    }
    createGPUBase(CanteraTorchProperties, mesh, Y);
    DEBUG_TRACE;
#endif

#ifdef GPUSolverNew_
    createGPUUEqn(CanteraTorchProperties, U);
    createGPUYEqn(CanteraTorchProperties, Y, inertIndex);
    createGPUEEqn(CanteraTorchProperties, thermo.he(), K);
    createGPUpEqn(CanteraTorchProperties, p, U);
    createGPURhoEqn(rho, phi);

    const volScalarField& mu = thermo.mu();
    const volScalarField& alpha = thermo.alpha();
    createGPUThermo(CanteraTorchProperties, T, thermo.he(), psi, alpha, mu, K, dpdt, chemistry);
    if (chemistry->ifChemstry())
    {
        chemistrySolver_GPU.setConstantValue(dfDataBase.num_cells, dfDataBase.num_species, 4096);
    }
#endif

    end1 = MPI_Wtime();
    time_monitor_init += double(end1 - start1);

    env::show();
    
    // print cells per proc
    int mpirank = 0, mpisize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpisize);

    label nCell = mesh.nCells();
    std::vector<label> nCells(mpisize);
    MPI_Gather(&nCell, 1, MPI_LABEL, nCells.data(), 1, MPI_LABEL, 0, MPI_COMM_WORLD);
    label total_nCell = 0;
    label min_nCell = std::numeric_limits<label>::max();
    label max_nCell = 0;
    Info << "nCells : " << endl;
    for(int i = 0; i < mpisize; ++i){
        total_nCell += nCells[i];
        min_nCell = std::min(min_nCell, nCells[i]);
        max_nCell = std::max(max_nCell, nCells[i]);
        Info <<  "\t" << i << " : " << nCells[i] << endl;
    }
    Info << endl;
    Info << "total nCell : " << total_nCell << endl;
    Info << "avg nCell : " << static_cast<double>(total_nCell) / mpisize << endl;
    Info << "min nCell : " << min_nCell << endl;
    Info << "max nCell : " << max_nCell << endl;

    // mesh renumbering

    csrPattern pattern_before(mesh);
    if(mpirank == 0 || mpirank == 1){
        pattern_before.write_mtx("pattern_before");
    }
    
    #include "renumberMesh.H"

    csrPattern pattern_after(mesh);
    if(mpirank == 0 || mpirank == 1){
        pattern_after.write_mtx("pattern_after");
    }

    if (nBlocks > 1){
        BlockPattern bcsr(pattern_after, regionPtr);
    }else if(nBlocks == 1){
        regionPtr.resize(17);
        // partition nCells into 16 regions
        for(label i = 0; i < 16; ++i){
            label regionStart = nCell * i / label(16);
            regionPtr[i] = regionStart;
        }
        regionPtr[16] = nCell;
        BlockPattern bcsr(pattern_after, regionPtr);
    }

#if defined(OPT_GenMatrix_E)
    init_const_coeff_ptr(Y);
    MeshSchedule::buildMeshSchedule(mesh);
#endif
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
        
        // store old time fields
#ifdef GPUSolverNew_
        dfDataBase.preTimeStep();
#endif
        double loop_start = MPI_Wtime();
        // --- Pressure-velocity PIMPLE corrector loop
        while (pimple.loop())
        {
            start = MPI_Wtime();
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
            end = MPI_Wtime();
            time_monitor_other += double(end - start);

            start = MPI_Wtime();
            if (pimple.firstPimpleIter() && !pimple.simpleRho())
            {
                #include "rhoEqn.H"
            }
            end = MPI_Wtime();
            time_monitor_rho += double(end - start);
            
            start = MPI_Wtime();
            #ifdef GPUSolver_
            #include "UEqn_GPU.H"
            #else
            #include "UEqn.H"
            #endif
            end = MPI_Wtime();
            time_monitor_U += double(end - start);

            if(combModelName!="ESF" && combModelName!="flareFGM" && combModelName!="DeePFGM")
            {
                start = MPI_Wtime();
                #ifdef GPUSolver_
                #include "YEqn_GPU.H"
                #else
                #include "YEqn.H"
                #endif
                end = MPI_Wtime();
                time_monitor_Y += double(end - start);

                start = MPI_Wtime();
                #ifdef GPUSolver_
                #include "EEqn_GPU.H"
                #else
                #include "EEqn.H"
                #endif
                end = MPI_Wtime();
                time_monitor_E += double(end - start);

            start = MPI_Wtime();
            #ifdef GPUSolverNew_
                thermo_GPU.correctThermo();
                thermo_GPU.sync();
            #if defined DEBUG_
                // check correctThermo
                int speciesIndex = 6;
                chemistry->correctThermo(); // reference
                double *h_boundary_T_tmp = new double[dfDataBase.num_boundary_surfaces];
                double *h_boundary_he_tmp = new double[dfDataBase.num_boundary_surfaces];
                double *h_boundary_mu_tmp = new double[dfDataBase.num_boundary_surfaces];
                double *h_boundary_rho_tmp = new double[dfDataBase.num_boundary_surfaces];
                double *h_boundary_thermo_alpha_tmp = new double[dfDataBase.num_boundary_surfaces];    
                double *h_boundary_thermo_psi_tmp = new double[dfDataBase.num_boundary_surfaces];
                double *h_boundary_thermo_rhoD_tmp = new double[dfDataBase.num_boundary_surfaces];
                offset = 0;
                forAll(T.boundaryField(), patchi)
                {
                    const fvPatchScalarField& patchHe = thermo.he().boundaryField()[patchi];
                    const fvPatchScalarField& patchMu = mu.boundaryField()[patchi];
                    const fvPatchScalarField& patchPsi = psi.boundaryField()[patchi];
                    const fvPatchScalarField& patchAlpha = alpha.boundaryField()[patchi];
                    const fvPatchScalarField& patchRho = thermo.rho()().boundaryField()[patchi];
                    const fvPatchScalarField& patchT = T.boundaryField()[patchi];
                    const fvPatchScalarField& patchRhoD = chemistry->rhoD(speciesIndex).boundaryField()[patchi];

                    int patchsize = patchT.size();
                    if (patchT.type() == "processor") {
                        memcpy(h_boundary_T_tmp + offset, &patchT[0], patchsize * sizeof(double));
                        scalarField patchTInternal = 
                                dynamic_cast<const processorFvPatchField<scalar>&>(patchT).patchInternalField()();
                        memcpy(h_boundary_T_tmp + offset + patchsize, &patchTInternal[0], patchsize * sizeof(double));

                        memcpy(h_boundary_he_tmp + offset, &patchHe[0], patchsize * sizeof(double));
                        scalarField patchHeInternal = 
                                dynamic_cast<const processorFvPatchField<scalar>&>(patchHe).patchInternalField()();
                        memcpy(h_boundary_he_tmp + offset + patchsize, &patchHeInternal[0], patchsize * sizeof(double));

                        memcpy(h_boundary_thermo_psi_tmp + offset, &patchPsi[0], patchsize * sizeof(double));
                        scalarField patchPsiInternal = 
                                dynamic_cast<const processorFvPatchField<scalar>&>(patchPsi).patchInternalField()();
                        memcpy(h_boundary_thermo_psi_tmp + offset + patchsize, &patchPsiInternal[0], patchsize * sizeof(double));

                        memcpy(h_boundary_thermo_alpha_tmp + offset, &patchAlpha[0], patchsize * sizeof(double));
                        scalarField patchAlphaInternal = 
                                dynamic_cast<const processorFvPatchField<scalar>&>(patchAlpha).patchInternalField()();
                        memcpy(h_boundary_thermo_alpha_tmp + offset + patchsize, &patchAlphaInternal[0], patchsize * sizeof(double));

                        memcpy(h_boundary_mu_tmp + offset, &patchMu[0], patchsize * sizeof(double));
                        scalarField patchMuInternal = 
                                dynamic_cast<const processorFvPatchField<scalar>&>(patchMu).patchInternalField()();
                        memcpy(h_boundary_mu_tmp + offset + patchsize, &patchMuInternal[0], patchsize * sizeof(double));

                        memcpy(h_boundary_rho_tmp + offset, &patchRho[0], patchsize * sizeof(double));
                        scalarField patchRhoInternal = 
                                dynamic_cast<const processorFvPatchField<scalar>&>(patchRho).patchInternalField()();
                        memcpy(h_boundary_rho_tmp + offset + patchsize, &patchRhoInternal[0], patchsize * sizeof(double));

                        memcpy(h_boundary_thermo_rhoD_tmp + offset, &patchRhoD[0], patchsize * sizeof(double));
                        scalarField patchRhoDInternal = 
                                dynamic_cast<const processorFvPatchField<scalar>&>(patchRhoD).patchInternalField()();
                        memcpy(h_boundary_thermo_rhoD_tmp + offset + patchsize, &patchRhoDInternal[0], patchsize * sizeof(double));

                        offset += patchsize * 2;
                    } else {
                        memcpy(h_boundary_T_tmp + offset, &patchT[0], patchsize * sizeof(double));
                        memcpy(h_boundary_he_tmp + offset, &patchHe[0], patchsize * sizeof(double));
                        memcpy(h_boundary_thermo_psi_tmp + offset, &patchPsi[0], patchsize * sizeof(double));
                        memcpy(h_boundary_thermo_alpha_tmp + offset, &patchAlpha[0], patchsize * sizeof(double));
                        memcpy(h_boundary_mu_tmp + offset, &patchMu[0], patchsize * sizeof(double));
                        memcpy(h_boundary_rho_tmp + offset, &patchRho[0], patchsize * sizeof(double));
                        memcpy(h_boundary_thermo_rhoD_tmp + offset, &patchRhoD[0], patchsize * sizeof(double));

                        offset += patchsize;
                    }
                }
                bool printFlag = false;
                int rank = -1;
                if (mpi_init_flag) {
                    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                }
                if (!mpi_init_flag || rank == 0) {
                    // thermo_GPU.compareT(&T[0], h_boundary_T_tmp, printFlag);
                    // thermo_GPU.compareHe(&thermo.he()[0], h_boundary_he_tmp, printFlag);
                    // thermo_GPU.comparePsi(&psi[0], h_boundary_thermo_psi_tmp, printFlag);
                    // thermo_GPU.compareAlpha(&alpha[0], h_boundary_thermo_alpha_tmp, printFlag);
                    // thermo_GPU.compareMu(&mu[0], h_boundary_mu_tmp, printFlag);
                    // thermo_GPU.compareRho(&thermo.rho()()[0], h_boundary_rho_tmp, printFlag);
                    // thermo_GPU.compareRhoD(&chemistry->rhoD(speciesIndex)[0], h_boundary_thermo_rhoD_tmp, speciesIndex, printFlag);
                }

                delete h_boundary_T_tmp;
                delete h_boundary_he_tmp;
                delete h_boundary_thermo_psi_tmp;
                delete h_boundary_thermo_alpha_tmp;
                delete h_boundary_mu_tmp;
                delete h_boundary_rho_tmp;
            #endif
            #else
                chemistry->correctThermo();
            #endif
            end = MPI_Wtime();
            time_monitor_chemistry_correctThermo += double(end - start);
            }
            else
            {
                combustion->correct();
            }
            // update T for debug
            #ifdef GPUSolverNew_
            double *h_T = dfDataBase.getFieldPointer("T", location::cpu, position::internal);
            double *h_boundary_T_tmp = new double[dfDataBase.num_boundary_surfaces];
            thermo_GPU.updateCPUT(h_T, h_boundary_T_tmp);
            memcpy(&T[0], h_T, T.size() * sizeof(double));
            offset = 0;
            forAll(T.boundaryField(), patchi) {
                const fvPatchScalarField& const_patchT = T.boundaryField()[patchi];
                fvPatchScalarField& patchT = const_cast<fvPatchScalarField&>(const_patchT);
                int patchsize = patchT.size();
                if (patchT.type() == "processor") {
                    memcpy(&patchT[0], h_boundary_T_tmp + offset, patchsize * sizeof(double));
                    offset += patchsize * 2;
                } else {
                    memcpy(&patchT[0], h_boundary_T_tmp + offset, patchsize * sizeof(double));
                    offset += patchsize;
                }
            }
            delete h_boundary_T_tmp;
            #endif

            Info<< "min/max(T) = " << min(T).value() << ", " << max(T).value() << endl;

            // --- Pressure corrector loop

            start = MPI_Wtime();
            int num_pimple_loop = pimple.nCorrPimple();
            while (pimple.correct())
            {
                if (pimple.consistent())
                {
                    // #include "pcEqn.H"
                }
                else
                {

                #if defined GPUSolverNew_
                    #include "pEqn_GPU.H"
                #else
                    #include "pEqn_CPU.H"
                #endif
                
                }
                num_pimple_loop --;
            }
            end = MPI_Wtime();
            time_monitor_p += double(end - start);

            start = MPI_Wtime();
            if (pimple.turbCorr())
            {
                turbulence->correct();
            }
            end = MPI_Wtime();
            time_monitor_turbulence_correct += double(end - start);
        }
        double loop_end = MPI_Wtime();
        double loop_time = double(loop_end - loop_start);

#ifdef GPUSolverNew_
        thermo_GPU.updateRho();
        dfDataBase.postTimeStep();
#if defined DEBUG_
        rho = thermo.rho();
#endif
#else
        rho = thermo.rho();
#endif

#ifdef GPUSolverNew_
        // write U
        UEqn_GPU.postProcess();
        memcpy(&U[0][0], dfDataBase.h_u, dfDataBase.cell_value_vec_bytes);
        U.correctBoundaryConditions();
#endif

        runTime.write();
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

        // Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
        //     << "  ClockTime = " << runTime.elapsedClockTime() << " s" << endl;

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

#ifdef GPUSolverNew_
    // clean cuda resources before main() exit.
    // the destruct order should be reversed from the creation order
    pEqn_GPU.cleanCudaResources();
    EEqn_GPU.cleanCudaResources();
    YEqn_GPU.cleanCudaResources();
    UEqn_GPU.cleanCudaResources();
    rhoEqn_GPU.cleanCudaResources();
    thermo_GPU.cleanCudaResources();
    dfDataBase.resetAmgxSolvers();
    dfDataBase.cleanCudaResources();
#endif

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
