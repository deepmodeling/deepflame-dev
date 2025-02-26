/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2015-2018 OpenCFD Ltd.
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

#include "multiBandTransmissivity.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    namespace radiation
    {
        defineTypeNameAndDebug(multiBandTransmissivity, 0);

        addToRunTimeSelectionTable
        (
            wallTransmissivityModel,
            multiBandTransmissivity,
            dictionary
        );
    }
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::radiation::multiBandTransmissivity::multiBandTransmissivity
(
    const dictionary& dict,
    const polyPatch& pp
)
:
    wallTransmissivityModel(dict, pp),
    coeffsDict_(dict),
    tauCoeffs_(),
    nBands_(0)
{
    coeffsDict_.readEntry("transmissivity", tauCoeffs_);
    nBands_ = tauCoeffs_.size();
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::tmp<Foam::scalarField>
Foam::radiation::multiBandTransmissivity::t
(
    const label bandI,
    const vectorField* incomingDirection,
    const scalarField* T
) const
{
    return tmp<scalarField>::New(pp_.size(), tauCoeffs_[bandI]);
}


Foam::scalar Foam::radiation::multiBandTransmissivity::t
(
    const label faceI,
    const label bandI,
    const vector dir,
    const scalar T
) const
{
    return tauCoeffs_[bandI];
}


// ************************************************************************* //
