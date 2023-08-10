/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2018-2021 OpenCFD Ltd.
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

\*---------------------------------------------------------------------------*/

#include "basicHeterogeneousReactingCloud.H"

#include "makeReactingParcelCloudFunctionObjects.H"

// Kinematic
#include "makeThermoParcelForces.H" // thermo variant
#include "makeParcelDispersionModels.H"
#include "makeReactingParcelInjectionModels.H" // Reacting variant
#include "makeParcelPatchInteractionModels.H"
#include "makeParcelStochasticCollisionModels.H"
#include "makeReactingParcelSurfaceFilmModels.H" // Reacting variant
#include "makeHeterogeneousReactingParcelHeterogeneousReactingModels.H"

// Thermodynamic
#include "makeParcelHeatTransferModels.H"

// Reacting
#include "makeReactingMultiphaseParcelCompositionModels.H"
#include "makeReactingParcelPhaseChangeModels.H"

// MPPIC sub-models
#include "makeMPPICParcelDampingModels.H"
#include "makeMPPICParcelIsotropyModels.H"
#include "makeMPPICParcelPackingModels.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

makeReactingParcelCloudFunctionObjects(basicHeterogeneousReactingCloud);

// Kinematic sub-models
makeThermoParcelForces(basicHeterogeneousReactingCloud);
makeParcelDispersionModels(basicHeterogeneousReactingCloud);
makeReactingParcelInjectionModels(basicHeterogeneousReactingCloud);
makeParcelPatchInteractionModels(basicHeterogeneousReactingCloud);
makeParcelStochasticCollisionModels(basicHeterogeneousReactingCloud);
makeReactingParcelSurfaceFilmModels(basicHeterogeneousReactingCloud);

// Thermo sub-models
makeParcelHeatTransferModels(basicHeterogeneousReactingCloud);

// Reacting sub-models
makeReactingMultiphaseParcelCompositionModels(basicHeterogeneousReactingCloud);
makeReactingParcelPhaseChangeModels(basicHeterogeneousReactingCloud);
makeHeterogeneousReactingParcelHeterogeneousReactingModels
(
    basicHeterogeneousReactingCloud
);

// MPPIC sub-models
makeMPPICParcelDampingModels(basicHeterogeneousReactingCloud);
makeMPPICParcelIsotropyModels(basicHeterogeneousReactingCloud);
makeMPPICParcelPackingModels(basicHeterogeneousReactingCloud);

// ************************************************************************* //
