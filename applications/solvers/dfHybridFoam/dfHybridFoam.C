/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | DeepFlame: a]deep learning [empowered open-source
   \\    /   O peration     |            platform for reacting flow simulations
    \\  /    A nd           | Copyright (C) 2023-2025 DeepFlame Contributors
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of DeepFlame.

    DeepFlame is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    DeepFlame is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with DeepFlame.  If not, see <http://www.gnu.org/licenses/>.

Application
    dfHybridFoam

Copyright
    Author: Teng Zhang @ AISI
    Date: 2026-03-05

Description
    Multi-region hybrid solver combining dfLowMachFoam (pressure-based PIMPLE)
    for low-Mach regions (e.g. combustor without laval nozzle) and
    dfHighSpeedFoam (density-based central-upwind) for high-speed regions
    (e.g. Laval nozzle, plume flow region).

    Region coupling is handled via inter-region boundary conditions on the
    shared interface, transferring T, p, U, Yi between the two solver domains.
    See dfHybridBoundaryConditions for details on the implemented BCs.

    The multi-region architecture follows the chtMultiRegionFoam pattern from
    OpenFOAM-7:
      - regionProperties defines the region names and types
      - Separate meshes and fields are created for each region
      - A time loop iterates over all regions, solving each with its
        appropriate algorithm

\*---------------------------------------------------------------------------*/

#include "dfChemistryModel.H"
#include "CanteraMixture.H"
#include "heRhoThermo.H"

#ifdef USE_PYTORCH
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#endif

#ifdef USE_LIBTORCH
#include <torch/script.h>
#include "DNNInferencer.H"
#endif

#include "fvCFD.H"
#include "dynamicFvMesh.H"
#include "rhoThermo.H"
#include "fluidThermo.H"
#include "turbulentFluidThermoModel.H"
#include "CombustionModel.H"
#include "pimpleControl.H"
#include "pressureControl.H"
#include "localEulerDdtScheme.H"
#include "fvcSmooth.H"
#include "PstreamGlobals.H"
#include "basicThermo.H"

#include "basicSprayCloud.H"
#include "SLGThermo.H"

// For high-speed region (density-based)
#include "fixedRhoFvPatchScalarField.H"
#include "include/directionInterpolate.H"
#include "fluxScheme.H"

// For multi-region support
#include "regionProperties.H"
#include "fixedGradientFvPatchFields.H"

#include "dfSingleStepReactingMixture.H"


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
#ifdef USE_PYTORCH
    pybind11::scoped_interpreter guard{};
#endif

    #define NO_CONTROL
    #define CREATE_MESH createMeshesPostProcess.H
    #include "postProcess.H"

    #include "listOptions.H"
    #include "setRootCase2.H"
    #include "listOutput.H"

    #include "createTime.H"

    // ---- Multi-region mesh creation ----
    // Read regionProperties from constant/regionProperties
    // which defines "lowMach" and "highSpeed" region lists
    #include "createMeshes.H"

    // ---- Create fields for each region type ----
    #include "createFields.H"

    #include "initContinuityErrs.H"

    // ---- Read time controls ----
    #include "include/readMultiRegionTimeControls.H"

    // ---- Compute initial Courant numbers ----
    #include "include/computeMultiRegionCourantNo.H"

    // ---- Set initial deltaT ----
    #include "include/setInitialMultiRegionDeltaT.H"

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nStarting time loop\n" << endl;

    label timeIndex = 0;

    while (runTime.run())
    {
        timeIndex++;

        // ---- Read time controls ----
        #include "include/readMultiRegionTimeControls.H"

        // ---- Compute Courant numbers from all regions ----
        #include "include/computeMultiRegionCourantNo.H"

        // ---- Adjust deltaT based on both regions ----
        #include "include/setMultiRegionDeltaT.H"

        runTime++;

        Info<< "Time = " << runTime.timeName() << nl << endl;

        // =====================================================================
        // Solve low-Mach regions (pressure-based PIMPLE)
        // =====================================================================
        forAll(lowMachRegions, i)
        {
            Info<< "\nSolving for lowMach region "
                << lowMachRegions[i].name() << endl;

            #include "lowMach/setRegionLowMachFields.H"
            #include "lowMach/solveLowMach.H"
        }

        // =====================================================================
        // Solve high-speed regions (density-based central-upwind)
        // =====================================================================
        forAll(highSpeedRegions, i)
        {
            Info<< "\nSolving for highSpeed region "
                << highSpeedRegions[i].name() << endl;

            #include "highSpeed/setRegionHighSpeedFields.H"
            #include "highSpeed/solveHighSpeed.H"
        }

        runTime.write();

        Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
            << "  ClockTime = " << runTime.elapsedClockTime() << " s"
            << nl << endl;
    }

    Info<< "End\n" << endl;

    return 0;
}

// ************************************************************************* //
