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

Global
    rhoEqn

Description
    Solve the continuity for density.

\*---------------------------------------------------------------------------*/
Info << "Enter rhoEqn " << endl;

{
    clockTime rhoEqnClock;

    fvScalarMatrix rhoEqn
    (
        fvm::ddt(rho)
      + fvc::div(phi)
    );

    time_monitor_rhoEqn_pre += rhoEqnClock.timeIncrement();

// #ifdef USE_DF_MATRIX
//     dfMatrix df_rhoEqn(rhoEqn, regionPtr);
//     auto& psi = const_cast<volScalarField&>(rhoEqn.psi());
//     auto& source = rhoEqn.source();
//     auto& internalCoeffs = rhoEqn.internalCoeffs();
//     auto& boundaryCoeffs = rhoEqn.boundaryCoeffs();

//     time_monitor_rhoEqn_convert += rhoEqnClock.timeIncrement();

//     df_rhoEqn.solve(psi, source, internalCoeffs, boundaryCoeffs);
// #else
    rhoEqn.solve();
// #endif

    time_monitor_rhoEqn_solve += rhoEqnClock.timeIncrement();
    time_monitor_rhoEqn += rhoEqnClock.elapsedTime();
}
// ************************************************************************* //

Info << "Exit rhoEqn " << endl;