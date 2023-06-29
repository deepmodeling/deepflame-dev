/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2018 OpenFOAM Foundation
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
    rhoCentralFoam

Description
    Density-based compressible flow solver based on central-upwind schemes of
    Kurganov and Tadmor with support for mesh-motion and topology changes.

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
// #include "psiThermo.H"
#include "rhoThermo.H"
#include "turbulentFluidThermoModel.H"
#include "fixedRhoFvPatchScalarField.H"
#include "directionInterpolate.H"
#include "localEulerDdtScheme.H"
#include "fvcSmooth.H"
#include "PstreamGlobals.H"
#include "CombustionModel.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
#ifdef USE_PYTORCH
    pybind11::scoped_interpreter guard{};//start python interpreter
#endif
    #define NO_CONTROL

    #include "postProcess.H"

    // #include "setRootCaseLists.H"
    #include "listOptions.H"
    #include "setRootCase2.H"
    #include "listOutput.H"
    #include "createTime.H"
    #include "createDynamicFvMesh.H"
    #include "createFields.H"
    #include "createFields_rk2.H"
    #include "createTimeControls.H"

    double time_monitor_flow=0;
    double time_monitor_chem=0;
    double time_monitor_Y=0;
    double time_monitor_AMR=0;
    clock_t start, end;

    turbulence->validate();

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    #include "readFluxScheme.H"

    dimensionedScalar v_zero("v_zero", dimVolume/dimTime, 0.0);
    dimensionedScalar refCri("refCri", dimensionSet(1, -4, 0, 0, 0), 0.0);

    // Courant numbers used to adjust the time-step
    scalar CoNum = 0.0;
    scalar meanCoNum = 0.0;

    std::vector<double> rkcoe1(3);
    std::vector<double> rkcoe2(3);
    std::vector<double> rkcoe3(3);
    scalar rk;
    label nrk=0;

    if(ddtSchemes == "RK2SSP")
    {
        rkcoe1[0]=1.0; rkcoe2[0]=0.0; rkcoe3[0]=1.0;
        rkcoe1[1]=0.5; rkcoe2[1]=0.5; rkcoe3[1]=0.5;
        rkcoe1[2]=0.0; rkcoe2[2]=0.0; rkcoe3[2]=0.0;
        rk=2;
    }
    else if(ddtSchemes == "RK3SSP")
    {
        rkcoe1[0]=1.0; rkcoe2[0]=0.0; rkcoe3[0]=1.0;
        rkcoe1[1]=0.75; rkcoe2[1]=0.25; rkcoe3[1]=0.25;
        rkcoe1[2]=0.33; rkcoe2[2]=0.66; rkcoe3[2]=0.66;
        rk=3;
    }

    Info<< "\nStarting time loop\n" << endl;

    while (runTime.run())
    {
        #include "readTimeControls.H"

        //used for AMR
        refCri = max(mag(fvc::grad(rho)));
        tmp<volScalarField> tmagGradrho = mag(fvc::grad(rho));
        volScalarField normalisedGradrho
        (
            "normalisedGradrho",
            tmagGradrho()/refCri
        );
        normalisedGradrho.writeOpt() = IOobject::AUTO_WRITE;
        tmagGradrho.clear();

        if (!LTS)
        {
            #include "setDeltaT.H"
            runTime++;

            // Do any mesh changes
            start = std::clock();
            mesh.update();
            end = std::clock();
            time_monitor_AMR += double(end - start) / double(CLOCKS_PER_SEC);
            
        }

        volScalarField rho_rhs("rho_rhs",rho_save/runTime.deltaT());
        volVectorField rhoU_rhs("rhoU_rhs",rhoU_save/runTime.deltaT());
        volScalarField rhoYi_rhs("rhoYi_rhs",rhoYi_save[0]/runTime.deltaT());
        volScalarField rhoE_rhs("rhoE_rhs",rhoE_save/runTime.deltaT());

        if ((ddtSchemes == "RK2SSP") || (ddtSchemes == "RK3SSP"))
        {
            for( nrk=0 ; nrk<rk ; nrk++)
            {
                Info <<"into rk"<< nrk+1 << nl << endl;

                #include "preCal.H"
                #include "phiCal.H"

                Info <<"\n in rk"<< nrk+1 << " finish pre-calculation"<< nl << endl;

                if (nrk == 0)
                {
                    #include "centralCourantNo.H"
                    if (LTS)
                    {
                        #include "setRDeltaT.H"
                        runTime++;
                    }

                    Info<< "Time = " << runTime.timeName() << nl << endl;

                    #include "updateFieldsSave.H"
                }

                // --- Solve density
                #include "rhoEqn.H"

                start = std::clock();
                // --- Solve momentum
                #include "rhoUEqn.H"
                end = std::clock();
                time_monitor_flow += double(end - start) / double(CLOCKS_PER_SEC);

                // --- Solve species
                #include "rhoYEqn.H"

                // --- Solve energy
                #include "rhoEEqn.H"

                if ((nrk == rk-1) && (chemScheme == "RR"))
                {
                    #include "calculateR.H"
                }

            }
            
        }
        else
        {
            #include "preCal.H"
            #include "centralCourantNo.H"

            if (LTS)
            {
                #include "setRDeltaT.H"
                runTime++;
            }

            Info<< "Time = " << runTime.timeName() << nl << endl;

            #include "phiCal.H"

            // --- Solve density
            #include "rhoEqn.H"

            start = std::clock();
            // --- Solve momentum
            #include "rhoUEqn.H"
            end = std::clock();
            time_monitor_flow += double(end - start) / double(CLOCKS_PER_SEC);

            // --- Solve species
            #include "rhoYEqn.H"

            // --- Solve energy
            #include "rhoEEqn.H"
        }

        turbulence->correct();

        runTime.write();

        Info<< "MonitorTime_chem = " << time_monitor_chem << " s" << nl << endl;
        Info<< "MonitorTime_Y = " << time_monitor_Y << " s" << nl << endl;
        Info<< "MonitorTime_flow = " << time_monitor_flow << " s" << nl << endl;
        Info<< "MonitorTime_AMR = " << time_monitor_AMR << " s" << nl << endl;

        Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
            << "  ClockTime = " << runTime.elapsedClockTime() << " s"
            << nl << endl;
    }

    Info<< "End\n" << endl;

    return 0;
}

// ************************************************************************* //
