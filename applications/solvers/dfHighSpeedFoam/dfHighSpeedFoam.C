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
#include "basicSprayCloud.H" // for spray
#include "SLGThermo.H"       // for spray
// #include "psiThermo.H"
#include "rhoThermo.H"
#include "turbulentFluidThermoModel.H"
#include "fixedRhoFvPatchScalarField.H"
#include "directionInterpolate.H"
#include "localEulerDdtScheme.H"
#include "fvcSmooth.H"
#include "PstreamGlobals.H"
#include "CombustionModel.H"

#include "fluxScheme.H"

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
    #include "createFieldsSave.H"
    #include "createTimeControls.H"

    double time_monitor_flow=0;
    double time_monitor_chem=0;
    double time_monitor_Y=0;
    double time_monitor_AMR=0;
    double time_monitor_E=0;
    double time_monitor_corrDiff=0;
    double time_monitor_parcels=0;
    clock_t start, end;

    turbulence->validate();

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    dimensionedScalar v_zero("v_zero", dimVolume/dimTime, 0.0);
    dimensionedScalar refCri("refCri", dimensionSet(1, -4, 0, 0, 0), 0.0);

    // Courant numbers used to adjust the time-step
    scalar CoNum = 0.0;
    scalar meanCoNum = 0.0;

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

        start = std::clock();
        // Store the particle positions
        parcels.storeGlobalPositions();
        end = std::clock();
        time_monitor_parcels += double(end - start) / double(CLOCKS_PER_SEC);

        #include "centralCourantNo.H"

        if (!LTS)
        {
            #include "setDeltaT.H"
            runTime++;

            // Do any mesh changes
            start = std::clock();
            mesh.update();
            end = std::clock();
            time_monitor_AMR += double(end - start) / double(CLOCKS_PER_SEC);

            if(mesh.changing())
            {
                MRF.update();
            }
            
        }

        gradP = mag(fvc::grad(p));
        forAll(pMax,celli)
        {
            if( p[celli]>pMax[celli] )
            {
                pMax[celli] = p[celli];
            }
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

                // --- calculate flux
                volScalarField rPsi("rPsi", 1.0/psi);
                volScalarField c("c", sqrt(thermo.Cp()/thermo.Cv()*rPsi));

                fluxSchemeFields->update(rho,rhoYi,nspecies,U,ea,p,c,phi,rhoPhi,rhoPhiYi,rhoUPhi,rhoEPhi);

                volScalarField muEff("muEff", turbulence->muEff());
                volTensorField tauMC("tauMC", muEff*dev2(Foam::T(fvc::grad(U))));

                Info <<"\n in rk"<< nrk+1 << " finish pre-calculation"<< nl << endl;

                if (nrk == 0)
                {
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

                start = std::clock();
                // --- Solve energy
                #include "rhoEEqn.H"
                end = std::clock();
                time_monitor_E += double(end - start) / double(CLOCKS_PER_SEC);

                if ((nrk == rk-1) && (chemScheme == "ode"))
                {
                    #include "calculateR.H"
                }

            }
            
        }
        else
        {
            // --- calculate flux
            volScalarField rPsi("rPsi", 1.0/psi);
            volScalarField c("c", sqrt(thermo.Cp()/thermo.Cv()*rPsi));

            fluxSchemeFields->update(rho,rhoYi,nspecies,U,ea,p,c,phi,rhoPhi,rhoPhiYi,rhoUPhi,rhoEPhi);

            volScalarField muEff("muEff", turbulence->muEff());
            volTensorField tauMC("tauMC", muEff*dev2(Foam::T(fvc::grad(U))));

            if (LTS)
            {
                #include "setRDeltaT.H"
                runTime++;
            }

            Info<< "Time = " << runTime.timeName() << nl << endl;

            if (parcels.solution().active())
            {
                start = std::clock();
                parcels.evolve();
                end = std::clock();
                time_monitor_parcels += double(end - start) / double(CLOCKS_PER_SEC);
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

            start = std::clock();
            // --- Solve energy
            #include "rhoEEqn.H"
            end = std::clock();
            time_monitor_E += double(end - start) / double(CLOCKS_PER_SEC);
        }

        turbulence->correct();

        runTime.write();

        Info<< "========Time Spent in diffenet parts========"<< endl;
        Info<< "MonitorTime_AMR            = " << time_monitor_AMR << " s" << endl;
        Info<< "MonitorTime_flow           = " << time_monitor_flow << " s" << endl;
        Info<< "Diffusion Correction Time  = " << time_monitor_corrDiff << " s" << endl;
        Info<< "MonitorTime_chem           = " << time_monitor_chem << " s" << endl;
        Info<< "MonitorTime_Y              = " << time_monitor_Y << " s" << endl;
        Info<< "MonitorTime_E              = " << time_monitor_E << " s" << endl;
        if (parcels.solution().active())
        {
            Info<< "calculate parcels            = " << time_monitor_parcels << " s" << endl;
        }
        Info<< "============================================"<<nl<< endl;
        Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
            << "  ClockTime = " << runTime.elapsedClockTime() << " s"
            << nl << endl;
    }

    Info<< "End\n" << endl;

    return 0;
}

// ************************************************************************* //
