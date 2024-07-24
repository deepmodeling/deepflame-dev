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
#define DEBUG_
// #define SHOW_MEMINFO

#define iscsr // true -> csr, false -> ell


#ifdef GPUSolverNew_
    #include "AmgXSolver.H"
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

    start1 = std::clock();
#ifdef GPUSolverNew_

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
    createGPUSolver(solversDict, rho, U, p, Y, thermo.he());

    int mpi_init_flag;
    checkMpiErrors(MPI_Initialized(&mpi_init_flag));
    if(mpi_init_flag) {
        initNccl();
    }

    /* For DeepFlame_Academic */
    mesh_info_para mesh_paras;    
    init_data_para init_data;

    createGPUBaseInput(mesh_paras, init_data, CanteraTorchProperties, mesh, Y); 

    createGPUUEqn(mesh_paras, init_data, CanteraTorchProperties, U);
    createGPUYEqn(mesh_paras, init_data, CanteraTorchProperties, Y, inertIndex);

    createGPUEEqn(mesh_paras, init_data, CanteraTorchProperties, thermo.he(), K);
    createGPUpEqn(mesh_paras, init_data, CanteraTorchProperties, p, U);
    createGPURhoEqn(mesh_paras, init_data, rho, phi);

    createGPUThermo(mesh_paras, init_data, CanteraTorchProperties, T, thermo.he(), 
                    psi, thermo.alpha(), thermo.mu(), K, dpdt, chemistry);

    set_mesh_info(mesh_paras);
    set_data_info(init_data);
    DEBUG_TRACE;

    // if (chemistry->ifChemstry())
    // {
    //     chemistrySolver_GPU.setConstantValue(mesh_paras.num_cells, init_data.num_species, 4096);
    // }
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
        
        // store old time fields
        #ifdef GPUSolverNew_
        preTimeStep();
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
                #ifdef GPUSolverNew_
                process_equation(MATRIX_EQUATION::rhoEqn);
                #endif
            }
            end = std::clock();
            time_monitor_rho += double(end - start) / double(CLOCKS_PER_SEC);        

            start = std::clock();
            #include "UEqn.H"
            #ifdef GPUSolverNew_
            process_equation(MATRIX_EQUATION::UEqn);
            #endif
            end = std::clock();
            time_monitor_U += double(end - start) / double(CLOCKS_PER_SEC);

            if(combModelName!="ESF" && combModelName!="flareFGM" && combModelName!="DeePFGM")
            {
                start = std::clock();
                // #include "YEqn.H"
                #ifdef GPUSolverNew_
                process_equation(MATRIX_EQUATION::YEqn);
                #endif
                end = std::clock();
                time_monitor_Y += double(end - start) / double(CLOCKS_PER_SEC);

                start = std::clock();
                // #include "EEqn.H"
                #ifdef GPUSolverNew_
                process_equation(MATRIX_EQUATION::EEqn);
                #endif
                end = std::clock();
                time_monitor_E += double(end - start) / double(CLOCKS_PER_SEC);

            start = std::clock();
            #ifdef GPUSolverNew_
                // thermo_GPU.correctThermo();
                // thermo_GPU.sync();
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
            // // update T for debug
            // #ifdef GPUSolverNew_
            // double *h_T = dfDataBase.getFieldPointer("T", location::cpu, position::internal);
            // double *h_boundary_T_tmp = new double[dfDataBase.num_boundary_surfaces];
            // thermo_GPU.updateCPUT(h_T, h_boundary_T_tmp);
            // memcpy(&T[0], h_T, T.size() * sizeof(double));
            // offset = 0;
            // forAll(T.boundaryField(), patchi) {
            //     const fvPatchScalarField& const_patchT = T.boundaryField()[patchi];
            //     fvPatchScalarField& patchT = const_cast<fvPatchScalarField&>(const_patchT);
            //     int patchsize = patchT.size();
            //     if (patchT.type() == "processor") {
            //         memcpy(&patchT[0], h_boundary_T_tmp + offset, patchsize * sizeof(double));
            //         offset += patchsize * 2;
            //     } else {
            //         memcpy(&patchT[0], h_boundary_T_tmp + offset, patchsize * sizeof(double));
            //         offset += patchsize;
            //     }
            // }
            // delete h_boundary_T_tmp;
            // #endif

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
                    process_equation(MATRIX_EQUATION::pEqn); //include all "pEqn_GPU.H"

                    // thermo_GPU.updateRho();

                    // // Thermodynamic density needs to be updated by psi*d(p) after the
                    // // pressure solution
                    // thermo_GPU.psip0();

                    // UEqn_GPU.getHbyA();
                    // pEqn_GPU.process();
                    // UEqn_GPU.sync();
                    // #include "pEqn_GPU.H"
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
                turbulence->correct();
            }
            end = std::clock();
            time_monitor_turbulence_correct += double(end - start) / double(CLOCKS_PER_SEC);
        }
        clock_t loop_end = std::clock();
        double loop_time = double(loop_end - loop_start) / double(CLOCKS_PER_SEC);

// #ifdef GPUSolverNew_
//         thermo_GPU.updateRho();
//         dfDataBase.postTimeStep();
// #if defined DEBUG_
//         rho = thermo.rho();
// #endif
// #else
//         rho = thermo.rho();
// #endif

// #ifdef GPUSolverNew_
//         // write U
//         UEqn_GPU.postProcess();
//         memcpy(&U[0][0], dfDataBase.h_u, dfDataBase.cell_value_vec_bytes);
//         U.correctBoundaryConditions();
// #endif

        std::cout << "*************************************************" << std::endl;
        std::cout << "NOTICE:" << std::endl;
        std::cout << "  The subsequent code has not been modified yet. " << std::endl;
        std::cout << "*************************************************" << std::endl;
        abort();    

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
