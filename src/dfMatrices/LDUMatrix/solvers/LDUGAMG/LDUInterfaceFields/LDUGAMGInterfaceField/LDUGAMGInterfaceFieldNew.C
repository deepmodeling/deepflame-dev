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

\*---------------------------------------------------------------------------*/

#include "LDUGAMGInterfaceField.H"

// * * * * * * * * * * * * * * * * Selectors * * * * * * * * * * * * * * * * //

Foam::autoPtr<Foam::LDUGAMGInterfaceField> Foam::LDUGAMGInterfaceField::New
(
    const LDUGAMGInterface& GAMGCp,
    const lduInterfaceField& fineInterface
)
{
    const word coupleType(fineInterface.interfaceFieldType());

    lduInterfaceFieldConstructorTable::iterator cstrIter =
        lduInterfaceFieldConstructorTablePtr_->find(coupleType);

    if (cstrIter == lduInterfaceFieldConstructorTablePtr_->end())
    {
        FatalErrorInFunction
            << "Unknown LDUGAMGInterfaceField type "
            << coupleType << nl
            << "Valid LDUGAMGInterfaceField types are :"
            << lduInterfaceFieldConstructorTablePtr_->sortedToc()
            << exit(FatalError);
    }

    return autoPtr<LDUGAMGInterfaceField>(cstrIter()(GAMGCp, fineInterface));
}


Foam::autoPtr<Foam::LDUGAMGInterfaceField> Foam::LDUGAMGInterfaceField::New
(
    const LDUGAMGInterface& GAMGCp,
    const bool doTransform,
    const int rank
)
{
    const word coupleType(GAMGCp.type());

    lduInterfaceConstructorTable::iterator cstrIter =
        lduInterfaceConstructorTablePtr_->find(coupleType);

    if (cstrIter == lduInterfaceConstructorTablePtr_->end())
    {
        FatalErrorInFunction
            << "Unknown LDUGAMGInterfaceField type "
            << coupleType << nl
            << "Valid LDUGAMGInterfaceField types are :"
            << lduInterfaceConstructorTablePtr_->sortedToc()
            << exit(FatalError);
    }

    return autoPtr<LDUGAMGInterfaceField>(cstrIter()(GAMGCp, doTransform, rank));
}


// ************************************************************************* //
