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

#include "multiBandAbsorption.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    namespace radiation
    {
        defineTypeNameAndDebug(multiBandAbsorption, 0);

        addToRunTimeSelectionTable
        (
            wallAbsorptionEmissionModel,
            multiBandAbsorption,
            dictionary
        );
    }
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::radiation::multiBandAbsorption::multiBandAbsorption
(
    const dictionary& dict,
    const polyPatch& pp
)
:
    wallAbsorptionEmissionModel(dict, pp),
    coeffsDict_(dict),
    aCoeffs_(),
    eCoeffs_(),
    nBands_(0)
{
    coeffsDict_.readEntry("absorptivity", aCoeffs_);
    coeffsDict_.readEntry("emissivity", eCoeffs_);
    nBands_ = aCoeffs_.size();
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::radiation::multiBandAbsorption::~multiBandAbsorption()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::tmp<Foam::scalarField>
Foam::radiation::multiBandAbsorption::a
(
    const label bandI,
    const vectorField* incomingDirection,
    const scalarField* T
) const
{
    return tmp<scalarField>::New(pp_.size(), aCoeffs_[bandI]);
}

Foam::scalar Foam::radiation::multiBandAbsorption::a
(
    const label faceI,
    const label bandI,
    const vector dir,
    const scalar T
) const
{
    return aCoeffs_[bandI];
}


Foam::tmp<Foam::scalarField> Foam::radiation::multiBandAbsorption::e
(
    const label bandI,
    const vectorField* incomingDirection,
    const scalarField* T
) const
{
    return tmp<scalarField>::New(pp_.size(), eCoeffs_[bandI]);
}


Foam::scalar Foam::radiation::multiBandAbsorption::e
(
    const label faceI,
    const label bandI,
    const vector dir,
    const scalar T
) const
{
    return eCoeffs_[bandI];
}


// ************************************************************************* //
