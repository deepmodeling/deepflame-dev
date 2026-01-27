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
#include "CorrectPhi.H"

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

// #define TIME
// #define DEBUG_
// #define SHOW_MEMINFO

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
#include "clockTime.H"

#define USE_DF_MATRIX
// #define OPT_GenMatrix_E
// #define OPT_GenMatrix_Y
// #define OPT_GenMatrix_U
// #define OPT_GenMatrix_p
// #define OPT_GenMatrix_Y_check
// #define OPT_GenMatrix_E_check
// #define OPT_GenMatrix_U_check
// #define OPT_GenMatrix_p_check

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
#ifdef USE_PYTORCH
    pybind11::scoped_interpreter guard{};//start python interpreter
#endif

    clockTime initClock;

    #include "postProcess.H"
    double postProcess_time = initClock.timeIncrement();
    #include "listOptions.H"
    double listOptions_time = initClock.timeIncrement();
    #include "setRootCase2.H"
    double setRootCase2_time = initClock.timeIncrement();
    Info << "postProcess_time = " << postProcess_time << " s" << endl;
    Info << "listOptions_time = " << listOptions_time << " s" << endl;
    Info << "setRootCase2_time = " << setRootCase2_time << " s" << endl;

    #include "listOutput.H"
    double listOutput_time = initClock.timeIncrement();
    Info << "listOutput_time = " << listOutput_time << " s" << endl;

    #include "createTime.H"
    double createTime_time = initClock.timeIncrement();
    Info << "createTime_time = " << createTime_time << " s" << endl;

    #include "createDynamicFvMesh.H"
    double createDynamicFvMesh_time = initClock.timeIncrement();
    Info << "createDynamicFvMesh_time = " << createDynamicFvMesh_time << " s" << endl;

    #include "createDyMControls.H"
    double createDyMControls_time = initClock.timeIncrement();
    Info << "createDyMControls_time = " << createDyMControls_time << " s" << endl;

    #include "initContinuityErrs.H"
    double initContinuityErrs_time = initClock.timeIncrement();
    Info << "initContinuityErrs_time = " << initContinuityErrs_time << " s" << endl;

    #include "createFields.H"
    double createFields_time = initClock.timeIncrement();
    Info << "createFields_time = " << createFields_time << " s" << endl;

    #include "createRhoUfIfPresent.H"
    double createRhoUfIfPresent_time = initClock.timeIncrement();
    Info << "createRhoUfIfPresent_time = " << createRhoUfIfPresent_time << " s" << endl;

    turbulence->validate();
    double turbulence_validate_time = initClock.timeIncrement();
    Info << "turbulence_validate_time = " << turbulence_validate_time << " s" << endl;
    
    if (!LTS)
    {
        #include "compressibleCourantNo.H"
        #include "setInitialDeltaT.H"
    }

    double LTS_time = initClock.timeIncrement();
    Info << "LTS_time = " << LTS_time << " s" << endl;

    env::show();
    
    // print cells per proc
    int mpirank = 0, mpisize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpisize);

    // nCell info per proc before renumbering
    {
        label nCell = mesh.nCells();
        label total_nCell = 0;
        label min_nCell = 0;
        label max_nCell = 0;
        MPI_Allreduce(&nCell, &total_nCell, 1, MPI_LABEL, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&nCell, &min_nCell, 1, MPI_LABEL, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&nCell, &max_nCell, 1, MPI_LABEL, MPI_MAX, MPI_COMM_WORLD);
        Info << "total nCell : " << total_nCell << endl;
        Info << "avg nCell : " << static_cast<double>(total_nCell) / mpisize << endl;
        Info << "min nCell : " << min_nCell << endl;
        Info << "max nCell : " << max_nCell << endl;
    }

    double proc_info_gather_time = initClock.timeIncrement();
    Info << "proc_info_gather_time = " << proc_info_gather_time << " s" << endl;

    int refineLevel = CanteraTorchProperties.subDict("RefineSettings").lookupOrDefault("refineLevel", 0);

    Info << "refineLevel : " << refineLevel << endl; 
    
    double refine_time = 0.;

    while (runTime.run())
    {
        while (refineLevel)
        {
            runTime++;
            #include "Refine.H"
            double refine_once_time = initClock.timeIncrement();
            refine_time += refine_once_time;
            // csrPattern refine(mesh);
            // refine.write_mtx("refine_" + std::to_string(refine_count));
            Info << "refine once time : " << refine_once_time << endl;

        }
        break;
    }

    // nCell info per proc after renumbering
    {
        label nCell = mesh.nCells();
        label total_nCell = 0;
        label min_nCell = 0;
        label max_nCell = 0;
        MPI_Allreduce(&nCell, &total_nCell, 1, MPI_LABEL, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&nCell, &min_nCell, 1, MPI_LABEL, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&nCell, &max_nCell, 1, MPI_LABEL, MPI_MAX, MPI_COMM_WORLD);
        Info << "total nCell : " << total_nCell << endl;
        Info << "avg nCell : " << static_cast<double>(total_nCell) / mpisize << endl;
        Info << "min nCell : " << min_nCell << endl;
        Info << "max nCell : " << max_nCell << endl;
    }

    // #include "intializeFields.H"
    double intializeFields_time = initClock.timeIncrement();
    Info << "IntializeFields time : " << intializeFields_time << endl; 

    // mesh renumbering
    csrPattern pattern_before(mesh);
    if(mpirank == 0 || mpirank == 1){
        pattern_before.write_mtx("pattern_before");
    }

    double pattern_before_time = initClock.timeIncrement();
    Info << "pattern_before_time = " << pattern_before_time << " s" << endl;

    #include "renumberMesh.H"

    double renumber_time = initClock.timeIncrement();
    Info << "renumber_time = " << renumber_time << " s" << endl;

    csrPattern pattern_after(mesh);
    if(mpirank == 0 || mpirank == 1){
        pattern_after.write_mtx("pattern_after");
    }

    double pattern_after_time = initClock.timeIncrement();
    Info << "pattern_after_time = " << pattern_after_time << " s" << endl;

    if (nBlocks > 1){
        BlockPattern blockPattern(pattern_after, regionPtr);
    }else if (nBlocks == 1 && env::REGION_DECOMPOSE_NBLOCKS > 1){
        regionPtr.resize(env::REGION_DECOMPOSE_NBLOCKS + 1);
        // partition nCells into env::REGION_DECOMPOSE_NBLOCKS regions
        label nCell = mesh.nCells();
        for(label i = 0; i < env::REGION_DECOMPOSE_NBLOCKS; ++i){
            regionPtr[i] = nCell * i / env::REGION_DECOMPOSE_NBLOCKS;
        }
        regionPtr[env::REGION_DECOMPOSE_NBLOCKS] = nCell;
        Info << "RegionID : regionStart, regionSize" << endl;
        for(label regionI = 0; regionI < regionPtr.size() - 1; ++regionI){
            Info << regionI << " : " << regionPtr[regionI] << ", " << regionPtr[regionI+1] - regionPtr[regionI] << endl;
        }
        BlockPattern blockPattern(pattern_after, regionPtr);
    }

    double block_pattern_time = initClock.timeIncrement();
    Info << "block_pattern_time = " << block_pattern_time << " s" << endl;

#if defined(OPT_GenMatrix_E) || defined(OPT_GenMatrix_Y) || defined(OPT_GenMatrix_U) || defined(OPT_GenMatrix_p)
    init_const_coeff_ptr(Y);
    MeshSchedule::buildMeshSchedule(mesh);
    XYBlock1DColoringStructuredMeshSchedule::buildXYBlock1DColoringStructuredMeshSchedule(mesh);
#endif

    double buildMeshSchedule_time = initClock.timeIncrement();
    Info << "buildMeshSchedule_time = " << buildMeshSchedule_time << " s" << endl;

    std::unique_ptr<dfMatrix> matrixPtr;

    if(env::REGION_DECOMPOSE_NBLOCKS == 1){
        matrixPtr = std::make_unique<dfMatrix>(mesh);
    }else if(env::REGION_DECOMPOSE_NBLOCKS > 1){
        matrixPtr = std::make_unique<dfMatrix>(mesh, regionPtr);
    }else{
        Info << "Error: env::REGION_DECOMPOSE_NBLOCKS should be greater than 0" << endl;
        exit(1);
    }

    // print all init time:
    double total_init_time = initClock.elapsedTime();

    Info << endl;
    Info << "----------------------------------------------------------" << endl;
    Info << "Total init time : " << total_init_time << " s" << endl;
    Info << "postProcess_time = " << postProcess_time << " s " << postProcess_time * 100. / total_init_time << "%" << endl;
    Info << "listOptions_time = " << listOptions_time << " s " << listOptions_time * 100. / total_init_time << "%" << endl;
    Info << "setRootCase2_time = " << setRootCase2_time << " s " << setRootCase2_time * 100. / total_init_time << "%" << endl;
    Info << "listOutput_time = " << listOutput_time << " s " << listOutput_time * 100. / total_init_time << "%" << endl;
    Info << "createTime_time = " << createTime_time << " s " << createTime_time * 100. / total_init_time << "%" << endl;
    Info << "createDynamicFvMesh_time = " << createDynamicFvMesh_time << " s " << createDynamicFvMesh_time * 100. / total_init_time << "%" << endl;
    Info << "createDyMControls_time = " << createDyMControls_time << " s " << createDyMControls_time * 100. / total_init_time << "%" << endl;
    Info << "initContinuityErrs_time = " << initContinuityErrs_time << " s " << initContinuityErrs_time * 100. / total_init_time << "%" << endl;
    Info << "createFields_time = " << createFields_time << " s " << createFields_time * 100. / total_init_time << "%" << endl;
    Info << "createRhoUfIfPresent_time = " << createRhoUfIfPresent_time << " s " << createRhoUfIfPresent_time * 100. / total_init_time << "%" << endl;
    Info << "turbulence_validate_time = " << turbulence_validate_time << " s " << turbulence_validate_time * 100. / total_init_time << "%" << endl;
    Info << "LTS_time = " << LTS_time << " s " << LTS_time * 100. / total_init_time << "%" << endl;
    Info << "proc_info_gather_time = " << proc_info_gather_time << " s " << proc_info_gather_time * 100. / total_init_time << "%" << endl;
    Info << "IntializeFields_time : = " << intializeFields_time << " s " << intializeFields_time * 100. / total_init_time << "%" << endl;
    Info << "refine_time = " << refine_time << " s " << refine_time * 100. / total_init_time << "%" << endl;
    Info << "pattern_before_time = " << pattern_before_time << " s " << pattern_before_time * 100. / total_init_time << "%" << endl;
    Info << "renumber_time = " << renumber_time << " s " << renumber_time * 100. / total_init_time << "%" << endl;
    Info << "pattern_after_time = " << pattern_after_time << " s " << pattern_after_time * 100. / total_init_time << "%" << endl;
    Info << "block_pattern_time = " << block_pattern_time << " s " << block_pattern_time * 100. / total_init_time << "%" << endl;
    Info << "buildMeshSchedule_time = " << buildMeshSchedule_time << " s " << buildMeshSchedule_time * 100. / total_init_time << "%" << endl;
    Info << "----------------------------------------------------------" << endl;

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    double time_monitor_other = 0;
    double time_monitor_misc = 0;

    double time_monitor_rho = 0;
    double time_monitor_U = 0;
    double time_monitor_Y = 0;
    double time_monitor_E = 0;
    double time_monitor_p = 0;
    double time_monitor_chemistry_correctThermo = 0;
    double time_monitor_turbulence_correct = 0;
    double time_monitor_chem = 0; // combustion correct

    double time_monitor_rhoEqn = 0;
    double time_monitor_rhoEqn_pre = 0;
    double time_monitor_rhoEqn_convert = 0;
    double time_monitor_rhoEqn_solve = 0;

    double time_monitor_UEqn = 0;
    double time_monitor_UEqn_pre = 0;
    double time_monitor_UEqn_pre_check = 0;
    double time_monitor_UEqn_convert = 0;
    double time_monitor_UEqn_solve = 0;
    double time_monitor_UEqn_post = 0;

    double time_monitor_YEqn = 0;
    double time_monitor_YEqn_pre = 0;
    double time_monitor_YEqn_pre_check = 0;
    double time_monitor_YEqn_convert = 0;
    double time_monitor_YEqn_solve = 0;
    double time_monitor_YEqn_post = 0;

    double time_monitor_EEqn = 0;
    double time_monitor_EEqn_pre = 0;
    double time_monitor_EEqn_pre_check = 0;
    double time_monitor_EEqn_convert = 0;
    double time_monitor_EEqn_solve = 0;

    double time_monitor_pEqn = 0;
    double time_monitor_pEqn_pre = 0;
    double time_monitor_pEqn_pre_check = 0;
    double time_monitor_pEqn_convert = 0;
    double time_monitor_pEqn_solve = 0;
    double time_monitor_pEqn_post = 0;

    label timeIndex = 0;
    double start, end, start1, end1, start2, end2;
    double start_new, stop_new;
    double time_new = 0;

    Info<< "\nStarting time loop\n" << endl;

    while (runTime.run())
    {
        syncClockTime loopClock;

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

        time_monitor_misc += loopClock.timeIncrement();
        
        // store old time fields
        // --- Pressure-velocity PIMPLE corrector loop
        while (pimple.loop())
        {
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

            time_monitor_misc += loopClock.timeIncrement();

            if (pimple.firstPimpleIter() && !pimple.simpleRho())
            {
                #include "rhoEqn.H"
            }
            time_monitor_rho += loopClock.timeIncrement();
            
            #include "UEqn.H"
            time_monitor_U += loopClock.timeIncrement();

            if(combModelName!="ESF" && combModelName!="flareFGM" && combModelName!="DeePFGM"){
                #include "YEqn.H"
                time_monitor_Y += loopClock.timeIncrement();

                #include "EEqn.H"
                time_monitor_E += loopClock.timeIncrement();

                chemistry->correctThermo();
                time_monitor_chemistry_correctThermo += loopClock.timeIncrement();
            }else{
                combustion->correct();
            }

            Info<< "min/max(T) = " << min(T).value() << ", " << max(T).value() << endl;
            time_monitor_misc += loopClock.timeIncrement();

            // --- Pressure corrector loop
            int num_pimple_loop = pimple.nCorrPimple();
            while (pimple.correct()){
                if (pimple.consistent()){
                    // #include "pcEqn.H"
                }else{
                    #include "pEqn_CPU.H"
                }
                num_pimple_loop --;
            }
            time_monitor_p += loopClock.timeIncrement();

            if (pimple.turbCorr())
            {
                turbulence->correct();
            }
            time_monitor_turbulence_correct += loopClock.timeIncrement();
        }

        rho = thermo.rho();
        runTime.write();
        time_monitor_misc += loopClock.timeIncrement();


        double loop_time = loopClock.elapsedTime();
        time_monitor_other = loop_time - time_monitor_misc - time_monitor_rho - time_monitor_U - time_monitor_Y - time_monitor_E - time_monitor_p - time_monitor_chemistry_correctThermo - time_monitor_turbulence_correct;

        Info<< "========Time Spent in diffenet parts========"<< endl;
        Info<< "loop Time                    = " << loop_time << " s" << endl;
        Info<< "misc Time                    = " << time_monitor_misc << " s" << endl;
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
        Info<< "percentage of p              = " << (time_monitor_p) / loop_time * 100 << " %" << endl;

        Info<< "========Time details of each equation======="<< endl;

        Info<< "rhoEqn Time                  = " << time_monitor_rhoEqn << " s" << endl;
        Info<< "rhoEqn pre                   = " << time_monitor_rhoEqn_pre << " s" << endl;
        Info<< "rhoEqn convert               = " << time_monitor_rhoEqn_convert << " s" << endl;
        Info<< "rhoEqn solve                 = " << time_monitor_rhoEqn_solve << " s" << endl;
        
        Info<< "============================================"<< endl;
        
        Info<< "UEqn Time                    = " << time_monitor_UEqn << " s" << endl;
        Info<< "UEqn pre                     = " << time_monitor_UEqn_pre << " s" << endl;
        Info<< "UEqn pre check               = " << time_monitor_UEqn_pre_check << " s" << endl;
        Info<< "UEqn convert                 = " << time_monitor_UEqn_convert << " s" << endl;
        Info<< "UEqn solve                   = " << time_monitor_UEqn_solve << " s" << endl;
        Info<< "UEqn post                    = " << time_monitor_UEqn_post << " s" << endl;
        
        Info<< "============================================"<< endl;
        
        Info<< "YEqn Time                    = " << time_monitor_YEqn << " s" << endl;
        Info<< "YEqn pre                     = " << time_monitor_YEqn_pre << " s" << endl;
        Info<< "YEqn pre check               = " << time_monitor_YEqn_pre_check << " s" << endl;
        Info<< "YEqn convert                 = " << time_monitor_YEqn_convert << " s" << endl;
        Info<< "YEqn solve                   = " << time_monitor_YEqn_solve << " s" << endl;
        Info<< "YEqn post                    = " << time_monitor_YEqn_post << " s" << endl;
        
        Info<< "============================================"<< endl;
        
        Info<< "EEqn Time                    = " << time_monitor_EEqn << " s" << endl;
        Info<< "EEqn pre                     = " << time_monitor_EEqn_pre << " s" << endl;
        Info<< "EEqn pre check               = " << time_monitor_EEqn_pre_check << " s" << endl;
        Info<< "EEqn convert                 = " << time_monitor_EEqn_convert << " s" << endl;
        Info<< "EEqn solve                   = " << time_monitor_EEqn_solve << " s" << endl;

        Info<< "============================================"<< endl;

        Info<< "pEqn Time                    = " << time_monitor_pEqn << " s" << endl;
        Info<< "pEqn pre                     = " << time_monitor_pEqn_pre << " s" << endl;
        Info<< "pEqn pre check               = " << time_monitor_pEqn_pre_check << " s" << endl;
        Info<< "pEqn convert                 = " << time_monitor_pEqn_convert << " s" << endl;
        Info<< "pEqn solve                   = " << time_monitor_pEqn_solve << " s" << endl;
        Info<< "pEqn post                    = " << time_monitor_pEqn_post << " s" << endl;

        Info<< "============================================"<< endl;
        double total_construct_time = time_monitor_rhoEqn_pre + time_monitor_UEqn_pre + time_monitor_YEqn_pre + time_monitor_EEqn_pre + time_monitor_pEqn_pre +\
            time_monitor_UEqn_post + time_monitor_YEqn_post + time_monitor_EEqn_solve + time_monitor_pEqn_post;
        double total_convert_time = time_monitor_rhoEqn_convert + time_monitor_YEqn_convert + time_monitor_EEqn_convert + time_monitor_pEqn_convert;
        double total_solve_time = time_monitor_rhoEqn_solve + time_monitor_YEqn_solve + time_monitor_EEqn_solve + time_monitor_pEqn_solve;
        double total_eqn_time = time_monitor_rhoEqn + time_monitor_UEqn + time_monitor_YEqn + time_monitor_EEqn + time_monitor_pEqn;
        Info << "Total eqn time : " << total_eqn_time << " s" << endl; 
        Info << "Total constuct time : " << total_construct_time << " s " << total_construct_time * 100. / total_eqn_time << " %" << endl;
        Info << "Total convert time : " << total_convert_time << " s " << total_convert_time * 100. / total_eqn_time << " %" << endl;
        Info << "Total solve time : " << total_solve_time << " s " << total_solve_time * 100. / total_eqn_time << " %" << endl;
        Info<< "============================================"<< endl;

        // Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
        //     << "  ClockTime = " << runTime.elapsedClockTime() << " s" << endl;

        time_monitor_misc = 0;
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
        time_monitor_rhoEqn_pre = 0;
        time_monitor_rhoEqn_convert = 0;
        time_monitor_rhoEqn_solve = 0;

        time_monitor_UEqn = 0;
        time_monitor_UEqn_pre = 0;
        time_monitor_UEqn_pre_check = 0;
        time_monitor_UEqn_convert = 0;
        time_monitor_UEqn_solve = 0;
        time_monitor_UEqn_post = 0;

        time_monitor_YEqn = 0;
        time_monitor_YEqn_pre = 0;
        time_monitor_YEqn_pre_check = 0;
        time_monitor_YEqn_convert = 0;
        time_monitor_YEqn_solve = 0;
        time_monitor_YEqn_post = 0;

        time_monitor_EEqn = 0;
        time_monitor_EEqn_pre = 0;
        time_monitor_EEqn_pre_check = 0;
        time_monitor_EEqn_convert = 0;
        time_monitor_EEqn_solve = 0;

        time_monitor_pEqn = 0;
        time_monitor_pEqn_pre = 0;
        time_monitor_pEqn_pre_check = 0;
        time_monitor_pEqn_convert = 0;
        time_monitor_pEqn_solve = 0;
        time_monitor_pEqn_post = 0;

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
