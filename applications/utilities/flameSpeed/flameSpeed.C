/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2021 OpenFOAM Foundation
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
    flameThickness
Description
\*---------------------------------------------------------------------------*/
#include "fvCFD.H"
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
int main(int argc, char *argv[])
{
    #include "setRootCase.H"
    #include "createTime.H"
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
    instantList timeDirs = timeSelector::select0(runTime, args);
    #include "createMesh.H"
    scalar flamePosition = 0.0211;
    forAll(timeDirs, timeI)
    {
        runTime.setTime(timeDirs[timeI], timeI);
        Info<< "Time = " << runTime.timeName() << endl;
        volScalarField T
        (
            IOobject
            (
                "T",
                runTime.timeName(),
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE
            ),
            mesh
        );
        volVectorField U
        (
            IOobject
            (
                "U",
                "0",
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE
            ),
            mesh
        );
        const auto gradT_ = fvc::grad(T)();
        scalarList gradT(mesh.nCells());
        forAll(gradT, cellI)
        {
            gradT[cellI] = gradT_[cellI].x();
        }
        const scalar flameThickness=  (max(T).value() - min(T).value())/max(gradT);
        Info<< "flameThickness = " << flameThickness << " m" << endl;
        Info<< "flamePoint.x (max T gradient) = " << mesh.C()[findMax(gradT)].x() << endl;
        Info<< "flamePropagationSpeed = " << (mesh.C()[findMax(gradT)].x() - flamePosition)/0.001 << " m/s" << endl;
        Info<< "flameSpeed = " << U[0][0] - (mesh.C()[findMax(gradT)].x() - flamePosition)/0.001 << " m/s" << endl;
        flamePosition = mesh.C()[findMax(gradT)].x();
    }
    Info<< nl << "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
        << "  ClockTime = " << runTime.elapsedClockTime() << " s"
        << nl << endl;
    Info<< "End\n" << endl;
    return 0;
}
// ************************************************************************* //
