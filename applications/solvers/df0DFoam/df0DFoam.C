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
#include "dynamicFvMesh.H"
#include "fluidThermo.H"
#include "turbulentFluidThermoModel.H"
#include "pimpleControl.H"
#include "pressureControl.H"
// #include "CorrectPhi.H"
//#include "fvOptions.H"
#include "localEulerDdtScheme.H"
#include "fvcSmooth.H"
#include "PstreamGlobals.H"

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
    #include "initContinuityErrs.H"
    #include "createFields.H"
    #include "createFieldRefs.H"
    #include "createRhoUfIfPresent.H"

    double time_monitor_chem=0;
    double time_monitor_Y=0;
    double time_monitor_corrThermo=0;
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
        #include "compressibleCourantNo.H"
        #include "setDeltaT.H"

        runTime++;

        Info<< "Time = " << runTime.timeName() << nl << endl;

        // --- Pressure-velocity PIMPLE corrector loop
        while (pimple.loop())
        {
            #include "YEqn.H"
            #include "EEqn.H"
            if (constProp == "volume") p[0] = rho[0] / psi[0];
        }

        rho = thermo.rho();

        runTime.write();

        Info<< "========Time Spent in diffenet parts========"<< endl;
        Info<< "Chemical sources           = " << time_monitor_chem << " s" << endl;
        Info<< "Species Equations          = " << time_monitor_Y << " s" << endl;
        Info<< "thermo & Trans Properties  = " << time_monitor_corrThermo << " s" << endl;
        Info<< "============================================"<<nl<< endl;

        Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
            << "  ClockTime = " << runTime.elapsedClockTime() << " s"<<endl;
#ifdef USE_PYTORCH
        if (log_ && torch_)
        {
            Info<< "    allsolveTime = " << chemistry.time_allsolve() << " s"
            << "    submasterTime = " << chemistry.time_submaster() << " s" << nl
            << "    sendProblemTime = " << chemistry.time_sendProblem() << " s"
            << "    recvProblemTime = " << chemistry.time_RecvProblem() << " s"
            << "    sendRecvSolutionTime = " << chemistry.time_sendRecvSolution() << " s" << nl
            << "    getDNNinputsTime = " << chemistry.time_getDNNinputs() << " s"
            << "    DNNinferenceTime = " << chemistry.time_DNNinference() << " s"
            << "    updateSolutionBufferTime = " << chemistry.time_updateSolutionBuffer() << " s" << nl
            << "    vec2ndarrayTime = " << chemistry.time_vec2ndarray() << " s"
            << "    pythonTime = " << chemistry.time_python() << " s"<< nl << endl;
        }
#endif
#ifdef USE_LIBTORCH
        if (log_ && torch_)
        {
            Info<< "    allsolveTime = " << chemistry.time_allsolve() << " s"
            << "    submasterTime = " << chemistry.time_submaster() << " s" << nl
            << "    sendProblemTime = " << chemistry.time_sendProblem() << " s"
            << "    recvProblemTime = " << chemistry.time_RecvProblem() << " s"
            << "    sendRecvSolutionTime = " << chemistry.time_sendRecvSolution() << " s" << nl
            << "    DNNinferenceTime = " << chemistry.time_DNNinference() << " s"
            << "    updateSolutionBufferTime = " << chemistry.time_updateSolutionBuffer() << " s" << nl;
        }
#endif
    }

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //