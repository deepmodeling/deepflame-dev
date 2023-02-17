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
#include "hePsiThermo.H"

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
    #include "createMesh.H"
    #include "createDyMControls.H"
    #include "initContinuityErrs.H"
    #include "createFields.H"
    #include "createRhoUfIfPresent.H"

    double time_monitor_flow,time_monitor_U,time_monitor_p,thermodensityu, rhoEqn,  constructM_p, updateU=0;
    double time_monitor_chem=0;
    double time_monitor_Y,yinside=0;
    double time_monitor_E, EE_relax, EE_cons=0;
    double time_monitor_corrThermo=0;
    double time_monitor_corrDiff=0;
    double Y_relax , p_solve, p_relax, EE_solve, U_solve, U_relax, U_construct, p_eq= 0;
    double Y_solve, Yeq, correctflux_t,laplaciansolve,constructM_Y= 0 ;
    label timeIndex = 0;
    clock_t start, end;

    turbulence->validate();

    if (!LTS)
    {
        #include "compressibleCourantNo.H"
        #include "setInitialDeltaT.H"
    }

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

            if (pimple.firstPimpleIter() && !pimple.simpleRho())
            {
                #include "rhoEqn.H"
            }

            start = std::clock();
            #include "UEqn.H"
            end = std::clock();
            time_monitor_U += double(end - start) / double(CLOCKS_PER_SEC);
            
            time_monitor_flow += double(end - start) / double(CLOCKS_PER_SEC);

            #include "YEqn.H"

            start = std::clock();
            #include "EEqn.H"
            end = std::clock();
            time_monitor_E += double(end - start) / double(CLOCKS_PER_SEC);

            start = std::clock();
            chemistry->correctThermo();
            end = std::clock();
            time_monitor_corrThermo += double(end - start) / double(CLOCKS_PER_SEC);

            Info<< "min/max(T) = " << min(T).value() << ", " << max(T).value() << endl;

            // --- Pressure corrector loop

            start = std::clock();
            while (pimple.correct())
            {
                if (pimple.consistent())
                {
                    
                    #include "pcEqn.H"
                }
                else
                {
                    #include "pEqn.H"
                }
            }
            end = std::clock();
            time_monitor_flow += double(end - start) / double(CLOCKS_PER_SEC);
            time_monitor_p += double(end - start) / double(CLOCKS_PER_SEC);


            if (pimple.turbCorr())
            {
                turbulence->correct();
            }
        }

        rho = thermo.rho();

        runTime.write();

        Info<< "========Time Spent in diffenet parts========"<< endl;
        Info<< "whole YEqn                 = " << Yeq << " s" << endl;
        Info<< "    Chemical sources       = " << time_monitor_chem << " s" << endl;
        Info<< "    Species Equations      = " << time_monitor_Y << " s" << endl;
        Info<< "        Y_relax            = " << Y_relax << "s"<< endl;
        Info<< "        Y_solve            = " << Y_solve << "s"<< endl;
        Info<< "        correctflux        = " << correctflux_t << "s"<< endl;
        Info<< "        laplacian solve    = " << laplaciansolve << "s"<< endl;
        Info<< "        construct matrix   = " << constructM_Y << "s"<< endl;        
        Info<< "    Diffusion Correction   = " << time_monitor_corrDiff << " s" << endl;       
        Info<< "U & p Equations            = " << time_monitor_flow << " s" << endl;
        Info<< "    p Equations            = " << time_monitor_p << " s" << endl;
        Info<< "      consturct matrix     = " << constructM_p << "s"<< endl;
        Info<< "      p_solve              = " << p_solve << "s"<< endl;
        Info<< "      p_relax              = " << p_relax << "s"<< endl;
        Info<< "      rho equation         = " << rhoEqn << "s"<< endl;
        Info<< "      density update       = " << thermodensityu << "s"<< endl;
        Info<< "      update U             = " << updateU << "s"<< endl;
        Info<< "    U Equations            = " << time_monitor_U << " s" << endl;
        Info<< "      U_construct          = " << U_construct << "s"<< endl;
        Info<< "      U_solve              = " << U_solve << "s"<< endl;
        Info<< "      U_relax              = " << U_relax << "s"<< endl;
        Info<< "Energy Equations           = " << time_monitor_E << " s" << endl;
        Info<< "      EE_construct         = " << EE_cons << "s"<< endl;
        Info<< "      EE_relax             = " << EE_relax<< "s"<< endl;
        Info<< "      EE_solve             = " << EE_solve << "s"<< endl;
        Info<< "thermo & Trans Properties  = " << time_monitor_corrThermo << " s" << endl;
        Info<< "sum Time                   = " << (time_monitor_chem + time_monitor_Y + time_monitor_flow + time_monitor_E + time_monitor_corrThermo + time_monitor_corrDiff) << " s" << endl;
        Info<< "============================================"<<nl<< endl;

        Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
            << "  ClockTime = " << runTime.elapsedClockTime() << " s" << endl;          
            
            
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
