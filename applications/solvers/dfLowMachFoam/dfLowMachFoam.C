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

#include "dfChemistryModel.H"
#include "CanteraMixture.H"
// #include "hePsiThermo.H"
#include "heRhoThermo.H"
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
#include "dynamicFvMesh.H"
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

#include "XYBlock1DColoringStructuredMeshSchedule.H"
#include "XBlock2DPartitionStructuredMeshSchedule.H"
#include <typeinfo>
#include "env.H"
#include "GenFvMatrix.H"
#include "multivariateGaussConvectionScheme.H"

// #define _CSR_
// #define _ELL_
#define _DIV_
// #define _LDU_
#define OPT_GenMatrix_Y
#define OPT_GenMatrix_E
#define OPT_GenMatrix_U
#define OPT_GenMatrix_p
// #define OPT_thermo
// #define OPT_GenMatrix_U_check
// #define OPT_GenMatrix_Y_check
// #define OPT_GenMatrix_E_check
// #define OPT_GenMatrix_p_check
// #define OPT_thermo_check

#ifdef _CSR_
#include "csrMatrix.H"
#endif
#ifdef _ELL_
#include "ellMatrix.H"
#endif
#ifdef _DIV_
#include "divMatrix.H"
#endif
#ifdef _LDU_
#include "LDUMatrix.H"
#endif
#include <assert.h>

// renumber
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

#include "csrPattern.H"

#ifdef FOAM_USE_ZOLTAN
    #include "zoltanRenumber.H"
#endif

#include "renumberMeshFuncs.H"

#define TIME
#ifdef TIME 
#define TICK0(prefix)\
    double prefix##_tick_0 = MPI_Wtime();

#define TICK(prefix,start,end)\
    double prefix##_tick_##end = MPI_Wtime();\
    Info << #prefix << "_time_" << #end << " : " << prefix##_tick_##end - prefix##_tick_##start << endl;

#else
#define TICK0(prefix) ;
#define TICK(prefix,start,end) ;
#endif

void benchmark(){
    Info << "benchmark start" << endl;
    bool b = false;
    scalar s = 1;
    label l = 1;
    int i = 1;
    clockTime clock;
    char buffer_1K[1024];
    char buffer_1M[1024 * 1024];
    string str_buffer_1K(1024, 'a');
    string str_buffer_1M(1024 * 1024, 'b');
    point p(1.,2.,3.);
    reduce(b, andOp<bool>());
    Info << "reduce bool and : " << clock.timeIncrement() << endl;
    reduce(b, orOp<bool>());
    Info << "reduce bool or : " << clock.timeIncrement() << endl;
    reduce(s, sumOp<scalar>());
    Info << "reduce scalar sum : " << clock.timeIncrement() << endl;
    reduce(s, maxOp<scalar>());
    Info << "reduce scalar max : " << clock.timeIncrement() << endl;
    reduce(s, minOp<scalar>());
    Info << "reduce scalar min : " << clock.timeIncrement() << endl;
    reduce(l, sumOp<label>());
    Info << "reduce label sum : " << clock.timeIncrement() << endl;
    reduce(l, maxOp<label>());
    Info << "reduce label max : " << clock.timeIncrement() << endl;
    reduce(l, minOp<label>());
    Info << "reduce label min : " << clock.timeIncrement() << endl;
    int flag_mpi_init;
    MPI_Initialized(&flag_mpi_init);
    if(flag_mpi_init){
        MPI_Bcast(&i, 1, MPI_INT, 0,  PstreamGlobals::MPI_COMM_FOAM);
    }
    Info << "MPI_Bcast int : " << clock.timeIncrement() << endl;
    Pstream::scatter(i);
    Info << "Pstream::scatter int : " << clock.timeIncrement() << endl;
    if(flag_mpi_init){
        MPI_Bcast(buffer_1K, 1024, MPI_CHAR, 0,  PstreamGlobals::MPI_COMM_FOAM);
    }
    Info << "MPI_Bcast char 1K : " << clock.timeIncrement() << endl;
    Pstream::scatter(str_buffer_1K);
    Info << "Pstream::scatter char 1K : " << clock.timeIncrement() << endl;
    if(flag_mpi_init){
        MPI_Bcast(buffer_1M, 1024 * 1024, MPI_CHAR, 0,  PstreamGlobals::MPI_COMM_FOAM);
    }
    Info << "MPI_Bcast char 1M : " << clock.timeIncrement() << endl;
    Pstream::scatter(str_buffer_1M);
    Info << "Pstream::scatter char 1M : " << clock.timeIncrement() << endl;
    
    reduce(p, sumOp<point>());
    Info << "reduce point sum : " << clock.timeIncrement() << endl;
    reduce(p, maxOp<point>());
    Info << "reduce point max : " << clock.timeIncrement() << endl;
    reduce(p, minOp<point>());
    Info << "reduce point min : " << clock.timeIncrement() << endl;
    if(flag_mpi_init){
        MPI_Barrier(PstreamGlobals::MPI_COMM_FOAM);
    }
    Info << "Pstream::barrier : " << clock.timeIncrement() << endl;
    Info << "benchmark end" << endl;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
#ifdef USE_PYTORCH
    pybind11::scoped_interpreter guard{};//start python interpreter
#endif

    #include "postProcess.H"
    // #include "setRootCaseLists.H"
    #include "listOptions.H"

    // MPI init here
    #include "setRootCase2.H"
    
    env_show();

    benchmark();

 
    clockTime initClock;

    #include "listOutput.H"
    double listOutput_time = initClock.timeIncrement();
    Info << "listOutput time : " << listOutput_time << endl;

    #include "createTime.H"
    double createTime_time = initClock.timeIncrement();
    Info << "createTime time : " << createTime_time << endl;

    // #include "createMesh.H"
    #include "createDynamicFvMesh.H"
    double createDynamicFvMesh_time = initClock.timeIncrement();
    Info << "createDynamicFvMesh time : " << createDynamicFvMesh_time << endl;

    #include "createDyMControls.H"
    double createDyMControls_time = initClock.timeIncrement();
    Info << "createDyMControls time : " << createDyMControls_time << endl;

    #include "initContinuityErrs.H"
    double initContinuityErrs_time = initClock.timeIncrement();
    Info << "initContinuityErrs time : " << initContinuityErrs_time << endl;

    #include "createFields.H"
    double createFields_time = initClock.timeIncrement();
    Info << "createFields time : " << createFields_time << endl;

    #include "createRhoUfIfPresent.H"
    double createRhoUfIfPresent_time = initClock.timeIncrement();
    Info << "createRhoUfIfPresent time : " << createRhoUfIfPresent_time << endl;

    label timeIndex = 0;

    turbulence->validate();
    double turbulenceValidate_time = initClock.timeIncrement();
    Info << "turbulenceValidate time : " << turbulenceValidate_time << endl;

    if (!LTS)
    {
        #include "compressibleCourantNo.H"
        #include "setInitialDeltaT.H"
    }
    int flag_mpi_init;
    MPI_Initialized(&flag_mpi_init);
    if(flag_mpi_init){
        MPI_Barrier(PstreamGlobals::MPI_COMM_FOAM);
    }
    double setInitialDeltaT_time = initClock.timeIncrement();
    Info << "setInitialDeltaT time : " << setInitialDeltaT_time << endl;

    double refine_time = 0.;

    while (runTime.run()){
        while (refineLevel)
        {
            runTime++;
            #include "Refine.H"
            double refine_once_time = initClock.timeIncrement();
            refine_time += refine_once_time;
            Info << "refine once time : " << refine_once_time << endl; 
        }
        break;
    }
    Info << "Refine time : " << refine_time << endl; 

    #include "intializeFields.H"
    double intializeFields_time = initClock.timeIncrement();
    Info << "IntializeFields time : " << intializeFields_time << endl; 

    #include "renumberMesh.H"
    double renumber_time = initClock.timeIncrement();
    Info << "Renumber time : " << renumber_time << endl; 

    init_const_coeff_ptr(fileName(CanteraTorchProperties.lookup("CanteraMechanismFile")).expand(), Y);

    surfaceScalarField upwindWeights
    (
        IOobject
        (
            "upwindWeights",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedScalar(dimensionSet(0,0,0,0,0,0,0), 0)
    );

    surfaceScalarField phiUc
    (
        IOobject
        (
            "phiUc",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedScalar(dimensionSet(1,0,-1,0,0,0,0), 0)
    );

    surfaceScalarField rhorAUf
    (
        IOobject
        (
            "rhorAUf",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedScalar(dimensionSet(0,0,1,0,0,0,0), 0)
    );

    volVectorField HbyA
    (
        IOobject
        (
            "HbyA",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedVector(dimensionSet(0,1,-1,0,0,0,0), Zero)
    );

    volScalarField rAU
    (
        IOobject
        (
            "rAU",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedScalar(dimensionSet(-1,3,1,0,0,0,0), 0)
    );

    surfaceScalarField phiHbyA
    (
        IOobject
        (
            "phiHbyA",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedScalar(dimensionSet(1,0,-1,0,0,0,0), 0)
    );

    #include "ScheduleSetup.H"
    double ScheduleSetup_time = initClock.timeIncrement();
    Info << "ScheduleSetup time : " << ScheduleSetup_time << endl; 
    
#ifdef _DIV_
    divMatrix div(mesh);
#endif

    double init_time = initClock.elapsedTime();
    Info << "Total Init time : " << init_time << endl;

    // // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    std::vector<double> step_timer;

    clockTime computeClock;

    Info<< "\nStarting time loop\n" << endl;

    while (runTime.run())
    {
        double time_readDyMControls=0.;
        double time_setDeltaT=0.;
        double time_monitor_rho=0.;
        double time_monitor_U=0.;
        double time_monitor_corrDiff=0.;
        double time_monitor_chem=0.;
        double time_monitor_Y=0.;
        double time_monitor_E=0.;
        double time_monitor_corrThermo=0.;
        double time_monitor_p=0.;
        double time_turbulenceCorrect=0.;
        double start, end;

        timeIndex ++;
        
        start = MPI_Wtime();
        #include "readDyMControls.H"
        end = MPI_Wtime();
        time_readDyMControls += end - start;

        start = MPI_Wtime();
        if (LTS)
        {
            #include "setRDeltaT.H"
        }
        else
        {
            #include "compressibleCourantNo.H"
            #include "setDeltaT.H"
        }
        end = MPI_Wtime();
        time_setDeltaT += end - start;
        
        runTime++;

        Info<< "Time = " << runTime.timeName() << nl << endl;

        // --- Pressure-velocity PIMPLE corrector loop
        int pimple_loop_count = 0;

        while (pimple.loop())
        {
            pimple_loop_count += 1;
            if (splitting)
            {
                Info<< "YEqn_RR.H start" << nl << endl;
                #include "YEqn_RR.H"
                Info<< "YEqn_RR.H end" << nl << endl;
            }
            
            if (pimple.firstPimpleIter() || moveMeshOuterCorrectors)
            {
                Info<< "new rhoU start" << nl << endl;
                start = MPI_Wtime();
                // Store momentum to set rhoUf for introduced faces.
                autoPtr<volVectorField> rhoU;
                if (rhoUf.valid())
                {
                    rhoU = new volVectorField("rhoU", rho*U);
                }
                end = MPI_Wtime();
                time_monitor_rho += end - start;
                Info<< "new rhoU end" << nl << endl;
            }

            if (pimple.firstPimpleIter() && !pimple.simpleRho())
            {
                Info<< "rhoEqn.H start" << nl << endl;
                start = MPI_Wtime();
                #include "rhoEqn.H"
                end = MPI_Wtime();
                time_monitor_rho += end - start;
                Info<< "rhoEqn.H end" << nl << endl;
            }

            Info<< "UEqn.H start" << nl << endl;
            start = MPI_Wtime();
            #include "UEqn.H"
            end = MPI_Wtime();
            Info<< "UEqn.H end" << nl << endl;
            time_monitor_U += end - start;

            if(combModelName!="ESF" && combModelName!="flareFGM" )
            {
                Info<< "YEqn.H start" << nl << endl;
                #include "YEqn.H"
                Info<< "YEqn.H end" << nl << endl;

                Info<< "EEqn.H start" << nl << endl;
                start = MPI_Wtime();
                #include "EEqn.H"
                end = MPI_Wtime();
                Info<< "EEqn.H end" << nl << endl;
                time_monitor_E += end - start;

                Info<< "chemistry->correctThermo start" << nl << endl;
                start = MPI_Wtime();
                #ifdef OPT_thermo
                volScalarField& mu = const_cast<volScalarField&>(thermo.mu()());
                volScalarField& alpha = const_cast<volScalarField&>(thermo.alpha());
                volScalarField& psi = const_cast<volScalarField&>(thermo.psi());
                correctThermo(Y, T, thermo.he(), rho, psi, alpha, mu, p, chemistry);
                #else
                chemistry->correctThermo();
                #endif
                end = MPI_Wtime();
                Info<< "chemistry->correctThermo end" << nl << endl;
                time_monitor_corrThermo += end - start;

                // when define OPT_thermo_check, do not define OPT_thermo
                #ifdef OPT_thermo_check
                volScalarField T_test = thermo.T();
                volScalarField rho_test = thermo.rho();
                volScalarField mu_test = thermo.mu();
                volScalarField alpha_test = thermo.alpha();
                volScalarField psi_test = thermo.psi();

                correctThermo(Y, T_test, thermo.he(), rho_test, psi_test, alpha_test, mu_test, p, chemistry);
                // check result
                check_field_boundary_equal(T, T_test);
                check_field_boundary_equal(psi, psi_test);
                check_field_boundary_equal(thermo.rho(), rho_test);
                check_field_boundary_equal(thermo.mu()(), mu_test);
                check_field_boundary_equal(thermo.alpha(), alpha_test);
                #endif
            }
            else
            {
                Info<< "chemistry->correct start" << nl << endl;
                combustion->correct();
                Info<< "chemistry->correct end" << nl << endl;
            }

            // Info << "min/max(T) = " << min(T).value() << ", " << max(T).value() << endl;

            // --- Pressure corrector loop

            while (pimple.correct())
            {
                if (pimple.consistent())
                {
                    Info<< "pcEqn.H start" << nl << endl;
                    #include "pcEqn.H"
                    Info<< "pcEqn.H end" << nl << endl;
                }
                else
                {
                    start = MPI_Wtime();
                    Info<< "pEqn.H start" << nl << endl;
                    #include "pEqn.H"
                    Info<< "pEqn.H end" << nl << endl;
                    end = MPI_Wtime();
                    time_monitor_p += end - start;
                }
            }

            // if (pimple.turbCorr())
            // {
            //     Info<< "turbulence->correct start" << nl << endl;
            //     start = MPI_Wtime();
            //     turbulence->correct();
            //     end = MPI_Wtime();
            //     time_turbulenceCorrect += end - start;
            //     Info<< "turbulence->correct end" << nl << endl;
            // }

        }
        start = MPI_Wtime();
        rho = thermo.rho();
        end = MPI_Wtime();
        time_monitor_rho += end - start;

        runTime.write();

        Info << "output time index " << runTime.timeIndex() << endl;

        double step_time = computeClock.timeIncrement();
        step_timer.push_back(step_time);

        Info<< "========Time Spent in diffenet parts========"<< endl;
        Info<< "Step time                  = " << step_time << endl;
        Info<< "Pimple loop count          = " << pimple_loop_count << endl;
        Info<< "readDyMControls            = " << time_readDyMControls << " s" << endl;
        Info<< "setDeltaT                  = " << time_setDeltaT << " s" << endl;
        Info<< "rho Equations              = " << time_monitor_rho << " s" << endl;
        Info<< "U Equations                = " << time_monitor_U << " s" << endl;
        Info<< "Diffusion Correction Time  = " << time_monitor_corrDiff << " s" << endl;
        Info<< "Chemical sources           = " << time_monitor_chem << " s" << endl;
        Info<< "Species Equations          = " << time_monitor_Y << " s" << endl;
        Info<< "Energy Equations           = " << time_monitor_E << " s" << endl;
        Info<< "thermo & Trans Properties  = " << time_monitor_corrThermo << " s" << endl;
        Info<< "p Equations                = " << time_monitor_p << " s" << endl;
        Info<< "turbulence Correction      = " << time_turbulenceCorrect << " s" << endl;
        Info<< "other                      = " << step_time \
            - time_readDyMControls - time_setDeltaT - time_monitor_rho \
            - time_monitor_U - time_monitor_corrDiff - time_monitor_chem - time_monitor_Y \
            - time_monitor_E - time_monitor_corrThermo - time_monitor_p - time_turbulenceCorrect \
            << endl;
        Info<< "============================================"<<nl<< endl;
    }

    double compute_time = computeClock.elapsedTime() - step_timer[0] - step_timer[1];

    Info << "Total Init time : " << init_time << endl;
    Info << "First time : " << step_timer[0] << endl;
    Info << "Second time : " << step_timer[1] << endl;
    Info << "Compute step : " << step_timer.size() - 2 << endl;
    Info << "Compute time : " << compute_time << endl;
    Info << "Total time : " << init_time + step_timer[0] + step_timer[1] + compute_time << endl;
    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
