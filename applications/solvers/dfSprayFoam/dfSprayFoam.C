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
    sprayFoam

Description
    Transient solver for compressible, turbulent flow with a spray particle
    cloud, with optional mesh motion and mesh topology changes.

\*---------------------------------------------------------------------------*/
#include "dfChemistryModel.H"
#include "CanteraMixture.H"
// #include "hePsiThermo.H"
#include "heRhoThermo.H"
#include "turbulentFluidThermoModel.H"

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
#include "turbulenceModel.H"
#include "basicSprayCloud.H"
//#include "radiationModel.H"
#include "SLGThermo.H"
#include "pimpleControl.H"
#include "CorrectPhi.H"
//#include "fvOptions.H"
#include "basicThermo.H"
#include "CombustionModel.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
#ifdef USE_PYTORCH
    pybind11::scoped_interpreter guard{};//start python interpreter
#endif
    #include "postProcess.H"

    // #include "setRootCaseLists.H"
    #include "listOptions.H"
    #include "setRootCase2.H"
    #include "listOutput.H"
    
    #include "createTime.H"
    #include "createDynamicFvMesh.H"
    #include "createDyMControls.H"
    #include "createFields.H"
    #include "compressibleCourantNo.H"
    #include "setInitialDeltaT.H"
    #include "initContinuityErrs.H"
    #include "createRhoUfIfPresent.H"
    
    double time_monitor_rho = 0;
    double time_monitor_U = 0;
    double time_monitor_Y = 0;
    double time_monitor_E = 0;
    double time_monitor_p = 0;
    double time_monitor_parcels=0;
    double time_monitor_chemistry_correctThermo=0;
    double time_monitor_turbulence_correct=0;
    clock_t start, end;

    turbulence->validate();

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nStarting time loop\n" << endl;

    while (runTime.run())
    {
        #include "readDyMControls.H"

        // Store divrhoU from the previous mesh so that it can be mapped
        // and used in correctPhi to ensure the corrected phi has the
        // same divergence
        autoPtr<volScalarField> divrhoU;
        if (correctPhi)
        {
            divrhoU = new volScalarField
            (
                "divrhoU",
                fvc::div(fvc::absolute(phi, rho, U))
            );
        }

        #include "compressibleCourantNo.H"
        #include "setDeltaT.H"

        runTime++;

        Info<< "Time = " << runTime.timeName() << nl << endl;

        // Store momentum to set rhoUf for introduced faces.
        autoPtr<volVectorField> rhoU;
        if (rhoUf.valid())
        {
            rhoU = new volVectorField("rhoU", rho*U);
        }

        start = std::clock();
        // Store the particle positions
        parcels.storeGlobalPositions();
        end = std::clock();
        time_monitor_parcels += double(end - start) / double(CLOCKS_PER_SEC);

        // Do any mesh changes
        mesh.update();

        if (mesh.changing())
        {
            MRF.update();

            if (correctPhi)
            {
                // Calculate absolute flux from the mapped surface velocity
                phi = mesh.Sf() & rhoUf();

                #include "correctPhi.H"

                // Make the fluxes relative to the mesh-motion
                fvc::makeRelative(phi, rho, U);
            }

            if (checkMeshCourantNo)
            {
                #include "meshCourantNo.H"
            }
        }

        start = std::clock();
        parcels.evolve();
        end = std::clock();
        time_monitor_parcels += double(end - start) / double(CLOCKS_PER_SEC);

        start = std::clock();
        #include "rhoEqn.H"
        end = std::clock();
        time_monitor_rho += double(end - start) / double(CLOCKS_PER_SEC);

        // --- Pressure-velocity PIMPLE corrector loop
        while (pimple.loop())
        {
            start = std::clock();
            #include "UEqn.H"
            end = std::clock();
            time_monitor_U += double(end - start) / double(CLOCKS_PER_SEC);

            if(combModelName!="ESF" && combModelName!="flareFGM"  && combModelName!="DeePFGM" && combModelName!="FSD")
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
                chemistry->correctThermo();
                end = std::clock();
                time_monitor_chemistry_correctThermo += double(end - start) / double(CLOCKS_PER_SEC);
            }
            else
            {
                combustion->correct();
            }
            Info<< "T gas min/max   " << min(T).value() << ", "
                << max(T).value() << endl;

            // --- Pressure corrector loop
            start = std::clock();
            while (pimple.correct())
            {
                #include "pEqn.H"
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

        rho = thermo.rho();

        runTime.write();

        Info<< "========Time Spent in diffenet parts========"<< endl;
        Info<< "rho Equations                = " << time_monitor_rho << " s" << endl;
        Info<< "U Equations                  = " << time_monitor_U << " s" << endl;
        Info<< "Y Equations                  = " << time_monitor_Y << " s" << endl;
        Info<< "E Equations                  = " << time_monitor_E << " s" << endl;
        Info<< "p Equations                  = " << time_monitor_p << " s" << endl;
        Info<< "calculate parcels            = " << time_monitor_parcels << " s" << endl;
        Info<< "chemistry correctThermo      = " << time_monitor_chemistry_correctThermo << " s" << endl;
        Info<< "turbulence correct           = " << time_monitor_turbulence_correct << " s" << endl;
        Info<< "============================================"<<nl<< endl;

        Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
            << "  ClockTime = " << runTime.elapsedClockTime() << " s"
            << nl << endl;
    }

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
