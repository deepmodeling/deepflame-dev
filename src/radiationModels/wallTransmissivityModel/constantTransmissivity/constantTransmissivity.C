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

#include "constantTransmissivity.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    namespace radiation
    {
        defineTypeNameAndDebug(constantTransmissivity, 0);

        addToRunTimeSelectionTable
        (
            wallTransmissivityModel,
            constantTransmissivity,
            dictionary
        );
    }
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::radiation::constantTransmissivity::constantTransmissivity
(
    const dictionary& dict,
    const polyPatch& pp
)
:
    wallTransmissivityModel(dict, pp),
    coeffsDict_(dict),
    tau_(Function1<scalar>::New("transmissivity", coeffsDict_))
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::tmp<Foam::scalarField> Foam::radiation::constantTransmissivity::t
(
    const label bandI,
    const vectorField* incomingDirection,
    const scalarField* T
) const
{
    if (tau_->constant())
    {
        // Use arbitrary argument for a_
        return tmp<scalarField>::New(pp_.size(), tau_->value(0));
    }

    if (T)
    {
        return tau_->value(*T);
    }

    FatalErrorInFunction
        << "Attempted to set 't' using a non-uniform function of Temperature, "
        << "but temperature field is unavailable"
        << abort(FatalError);

    return nullptr;
}


Foam::scalar Foam::radiation::constantTransmissivity::t
(
    const label faceI,
    const label bandI,
    const vector dir,
    const scalar T
) const
{
    return tau_->value(T);
}


// ************************************************************************* //
