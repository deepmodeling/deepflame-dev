/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2019-2021 OpenCFD Ltd.
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

#include "solidProperties.H"
#include "Switch.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

Foam::autoPtr<Foam::solidProperties> Foam::solidProperties::New
(
    const word& name
)
{
    DebugInFunction << "Constructing solidProperties" << endl;

    auto* ctorPtr = ConstructorTable(name);

    if (!ctorPtr)
    {
        FatalErrorInLookup
        (
            "solidProperties",
            name,
            *ConstructorTablePtr_
        ) << exit(FatalError);
    }

    return autoPtr<solidProperties>(ctorPtr());
}


Foam::autoPtr<Foam::solidProperties> Foam::solidProperties::New
(
    const dictionary& dict
)
{
    DebugInFunction << "Constructing solid" << endl;

    const word solidType(dict.dictName());

    if (dict.found("defaultCoeffs"))
    {
        // Backward-compatibility

        if (dict.get<bool>("defaultCoeffs"))
        {
            return New(solidType);
        }

        return autoPtr<solidProperties>::New
        (
            dict.optionalSubDict(solidType + "Coeffs")
        );
    }

    auto* ctorPtr = dictionaryConstructorTable(solidType);

    if (!ctorPtr)
    {
        FatalIOErrorInLookup
        (
            dict,
            "solidProperties",
            solidType,
            *dictionaryConstructorTablePtr_
        ) << exit(FatalIOError);
    }

    return autoPtr<solidProperties>(ctorPtr(dict));
}


// ************************************************************************* //
