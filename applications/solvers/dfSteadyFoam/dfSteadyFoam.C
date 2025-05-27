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

// #include "psiReactionThermo.H"
#include "basicThermo.H"
#include "CombustionModel.H"
#include "multivariateScheme.H"

#include "simpleControl.H"
#include "pressureControl.H"
#include "fvOptions.H"
#include "fvcSmooth.H"

// #include "PstreamGlobals.H"
// #include "CombustionModel.H"

// to be decided
// #include "basicSprayCloud.H"
// #include "SLGThermo.H"

//#define GPUSolver_
// #define TIME
// #define DEBUG_
// #define SHOW_MEMINFO

    #ifdef GPUSolver_
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
    // #include "processorFvPatchField.H"
    // #include "cyclicFvPatchField.H"
    // #include "multivariateGaussConvectionScheme.H"
    // #include "limitedSurfaceInterpolationScheme.H"
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
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createMesh.H"
    #include "createControl.H"
    #include "createFields.H"
    #include "initContinuityErrs.H"

    #ifdef ODE_GPU_SOLVER
    #include "createFields_GPU.H"
    #endif

    turbulence->validate();

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
    
    Info<< "\nStarting time loop\n" << endl;

    while (simple.loop(runTime))
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;

        // #include "rhoEqn.H"

        #include "UEqn.H"

        if(combModelName!="ESF" && combModelName!="flareFGM"  && combModelName!="DeePFGM" && combModelName!="FSD")
        {
            #include "YEqn.H"

            #include "EEqn.H"
            
            chemistry->correctThermo();

            Info<< "min/max(T) = "
                << min(T).value() << ", " << max(T).value() << endl;
        }
        else
        {
            combustion->correct();
        }

        // update T for debug

        if (simple.consistent())
        {
            #include "pcEqn.H"
        }
        else
        {
            #include "pEqn.H"
        }

        Info<< "min/max(p) = " << min(p).value() << ", " << max(p).value() << endl;

        turbulence->correct();
        // rho = thermo.rho();

        runTime.write();

        Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
            << "  ClockTime = " << runTime.elapsedClockTime() << " s"
            << nl << endl;
    }

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
