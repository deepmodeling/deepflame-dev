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

#include "DIVGAMGInterface.H"
#include "DIVGAMGAgglomeration.H"
#include "lduMatrix.H"


// * * * * * * * * * * * * * * * * Selectors * * * * * * * * * * * * * * * * //

Foam::autoPtr<Foam::DIVGAMGInterface> Foam::DIVGAMGInterface::New
(
    const label index,
    const lduInterfacePtrsList& coarseInterfaces,
    const lduInterface& fineInterface,
    const labelField& localRestrictAddressing,
    const labelField& neighbourRestrictAddressing,
    const label fineLevelIndex,
    const label coarseComm
)
{
    const word coupleType(fineInterface.type());

    lduInterfaceConstructorTable::iterator cstrIter =
        lduInterfaceConstructorTablePtr_->find(coupleType);

    if (cstrIter == lduInterfaceConstructorTablePtr_->end())
    {
        FatalErrorInFunction
            << "Unknown DIVGAMGInterface type " << coupleType << ".\n"
            << "Valid DIVGAMGInterface types are :"
            << lduInterfaceConstructorTablePtr_->sortedToc()
            << exit(FatalError);
    }

    return autoPtr<DIVGAMGInterface>
    (
        cstrIter()
        (
            index,
            coarseInterfaces,
            fineInterface,
            localRestrictAddressing,
            neighbourRestrictAddressing,
            fineLevelIndex,
            coarseComm
        )
    );
}


Foam::autoPtr<Foam::DIVGAMGInterface> Foam::DIVGAMGInterface::New
(
    const word& coupleType,
    const label index,
    const lduInterfacePtrsList& coarseInterfaces,
    Istream& is
)
{
    IstreamConstructorTable::iterator cstrIter =
        IstreamConstructorTablePtr_->find(coupleType);

    if (cstrIter == IstreamConstructorTablePtr_->end())
    {
        FatalErrorInFunction
            << "Unknown DIVGAMGInterface type " << coupleType << ".\n"
            << "Valid DIVGAMGInterface types are :"
            << IstreamConstructorTablePtr_->sortedToc()
            << exit(FatalError);
    }

    return autoPtr<DIVGAMGInterface>(cstrIter()(index, coarseInterfaces, is));
}


// ************************************************************************* //
