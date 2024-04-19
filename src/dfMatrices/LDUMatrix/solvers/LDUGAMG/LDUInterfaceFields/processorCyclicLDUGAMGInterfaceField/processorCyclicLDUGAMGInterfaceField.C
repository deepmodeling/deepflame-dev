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

#include "processorCyclicLDUGAMGInterfaceField.H"
#include "addToRunTimeSelectionTable.H"
#include "lduMatrix.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(processorCyclicLDUGAMGInterfaceField, 0);
    addToRunTimeSelectionTable
    (
        LDUGAMGInterfaceField,
        processorCyclicLDUGAMGInterfaceField,
        lduInterface
    );
    addToRunTimeSelectionTable
    (
        LDUGAMGInterfaceField,
        processorCyclicLDUGAMGInterfaceField,
        lduInterfaceField
    );
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::processorCyclicLDUGAMGInterfaceField::processorCyclicLDUGAMGInterfaceField
(
    const LDUGAMGInterface& GAMGCp,
    const lduInterfaceField& fineInterface
)
:
    processorLDUGAMGInterfaceField(GAMGCp, fineInterface)
{}


Foam::processorCyclicLDUGAMGInterfaceField::processorCyclicLDUGAMGInterfaceField
(
    const LDUGAMGInterface& GAMGCp,
    const bool doTransform,
    const int rank
)
:
    processorLDUGAMGInterfaceField(GAMGCp, doTransform, rank)
{}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::processorCyclicLDUGAMGInterfaceField::~processorCyclicLDUGAMGInterfaceField()
{}


// ************************************************************************* //
