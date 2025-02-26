/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2018-2019 OpenCFD Ltd.
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

#include "lookup.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    namespace radiation
    {
        defineTypeNameAndDebug(lookup, 0);
        addToRunTimeSelectionTable
        (
            boundaryRadiationPropertiesPatch,
            lookup,
            dictionary
        );
    }
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::radiation::lookup::lookup
(
    const dictionary& dict,
    const polyPatch& pp
)
:
    boundaryRadiationPropertiesPatch(dict, pp),
    pp_(pp),
    dict_(dict)
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::tmp<Foam::scalarField> Foam::radiation::lookup::e
(
    const label bandI,
    const vectorField* dir,
    const scalarField* T
) const
{
    return tmp<scalarField>::New
    (
        pp_.size(),
        dict_.get<scalar>("emissivity")
    );
}


Foam::scalar Foam::radiation::lookup::e
(
    const label faceI,
    const label bandI,
    const vector& dir,
    const scalar T
) const
{
    return dict_.get<scalar>("emissivity");
}


Foam::tmp<Foam::scalarField> Foam::radiation::lookup::a
(
    const label bandI,
    const vectorField* dir,
    const scalarField* T
) const
{
    return tmp<scalarField>::New
    (
        pp_.size(),
        dict_.get<scalar>("absorptivity")
    );
}


Foam::scalar Foam::radiation::lookup::a
(
    const label faceI,
    const label bandI,
    const vector& dir,
    const scalar T
) const
{
     return dict_.get<scalar>("absorptivity");
}


Foam::tmp<Foam::scalarField> Foam::radiation::lookup::t
(
    const label bandI,
    const vectorField* dir,
    const scalarField* T
) const
{
    return tmp<scalarField>::New
    (
        pp_.size(),
        dict_.getOrDefault<scalar>("transmissivity", 0)
    );
}


Foam::scalar Foam::radiation::lookup::t
(
    const label faceI,
    const label bandI,
    const vector& dir,
    const scalar T
) const
{
    return dict_.getOrDefault<scalar>("transmissivity", 0);
}


Foam::tmp<Foam::scalarField> Foam::radiation::lookup::rSpec
(
    const label bandI,
    const vectorField* dir,
    const scalarField* T
) const
{
    return tmp<scalarField>::New(pp_.size(), Zero);
}


Foam::scalar Foam::radiation::lookup::rSpec
(
    const label faceI,
    const label bandI,
    const vector& dir,
    const scalar T
) const
{
    return Zero;
}


Foam::tmp<Foam::scalarField> Foam::radiation::lookup::rDiff
(
    const label bandI,
    const vectorField* dir,
    const scalarField* T
) const
{
    return tmp<scalarField>::New(pp_.size(), Zero);
}


Foam::scalar Foam::radiation::lookup::rDiff
(
    const label faceI,
    const label bandI,
    const vector& dir,
    const scalar T
) const
{
    return Zero;
}


bool Foam::radiation::lookup::isGrey() const
{
    return true;
}


Foam::label Foam::radiation::lookup::nBands() const
{
    return 1;
}


// ************************************************************************* //
