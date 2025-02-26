/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2015-2018, 2024 OpenCFD Ltd.
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

#include "constantAbsorption.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    namespace radiation
    {
        defineTypeNameAndDebug(constantAbsorption, 0);

        addToRunTimeSelectionTable
        (
            wallAbsorptionEmissionModel,
            constantAbsorption,
            dictionary
        );
    }
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::radiation::constantAbsorption::constantAbsorption
(
    const dictionary& dict,
     const polyPatch& pp
)
:
    wallAbsorptionEmissionModel(dict, pp),
    coeffsDict_(dict),
    a_(Function1<scalar>::New("absorptivity", coeffsDict_)),
    e_(Function1<scalar>::New("emissivity", coeffsDict_))
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::tmp<Foam::scalarField> Foam::radiation::constantAbsorption::a
(
    const label bandI,
    const vectorField* incomingDirection,
    const scalarField* T
) const
{
    if (a_->constant())
    {
        // Use arbitrary argument for a_
        return tmp<scalarField>::New(pp_.size(), a_->value(0));
    }

    if (T)
    {
        return a_->value(*T);
    }

    FatalErrorInFunction
        << "Attempted to set 'a' using a non-uniform function of Temperature, "
        << "but temperature field is unavailable"
        << abort(FatalError);

    return nullptr;
}


Foam::scalar Foam::radiation::constantAbsorption::a
(
    const label faceI,
    const label bandI,
    const vector dir,
    const scalar T
) const
{
    return a_->value(T);
}


Foam::tmp<Foam::scalarField> Foam::radiation::constantAbsorption::e
(
    const label bandI,
    const vectorField* incomingDirection,
    const scalarField* T
) const
{
    if (e_->constant())
    {
        // Use arbitrary argument for e_
        return tmp<scalarField>::New(pp_.size(), e_->value(0));
    }

    if (T)
    {
        return e_->value(*T);
    }

    FatalErrorInFunction
        << "Attempted to set 'e' using a non-uniform function of Temperature, "
        << "but temperature field is unavailable"
        << abort(FatalError);

    return nullptr;
}


Foam::scalar Foam::radiation::constantAbsorption::e
(
    const label faceI,
    const label bandI,
    const vector dir,
    const scalar T
) const
{
    return e_->value(T);
}


// ************************************************************************* //
