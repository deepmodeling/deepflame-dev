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

#include "cyclicdfGAMGInterfaceField.H"
#include "addToRunTimeSelectionTable.H"
#include "lduMatrix.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(cyclicdfGAMGInterfaceField, 0);
    addToRunTimeSelectionTable
    (
        dfGAMGInterfaceField,
        cyclicdfGAMGInterfaceField,
        lduInterface
    );
    addToRunTimeSelectionTable
    (
        dfGAMGInterfaceField,
        cyclicdfGAMGInterfaceField,
        lduInterfaceField
    );

    // Add under name cyclicSlip
    addNamedToRunTimeSelectionTable
    (
        dfGAMGInterfaceField,
        cyclicdfGAMGInterfaceField,
        lduInterface,
        cyclicSlip
    );
    addNamedToRunTimeSelectionTable
    (
        dfGAMGInterfaceField,
        cyclicdfGAMGInterfaceField,
        lduInterfaceField,
        cyclicSlip
    );
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::cyclicdfGAMGInterfaceField::cyclicdfGAMGInterfaceField
(
    const dfGAMGInterface& GAMGCp,
    const lduInterfaceField& fineInterface
)
:
    dfGAMGInterfaceField(GAMGCp, fineInterface),
    cyclicInterface_(refCast<const cyclicdfGAMGInterface>(GAMGCp)),
    doTransform_(false),
    rank_(0)
{
    const cyclicLduInterfaceField& p =
        refCast<const cyclicLduInterfaceField>(fineInterface);

    doTransform_ = p.doTransform();
    rank_ = p.rank();
}


Foam::cyclicdfGAMGInterfaceField::cyclicdfGAMGInterfaceField
(
    const dfGAMGInterface& GAMGCp,
    const bool doTransform,
    const int rank
)
:
    dfGAMGInterfaceField(GAMGCp, doTransform, rank),
    cyclicInterface_(refCast<const cyclicdfGAMGInterface>(GAMGCp)),
    doTransform_(doTransform),
    rank_(rank)
{}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::cyclicdfGAMGInterfaceField::~cyclicdfGAMGInterfaceField()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::cyclicdfGAMGInterfaceField::updateInterfaceMatrix
(
    scalarField& result,
    const scalarField& psiInternal,
    const scalarField& coeffs,
    const direction cmpt,
    const Pstream::commsTypes
) const
{
    // Get neighbouring field
    scalarField pnf
    (
        cyclicInterface_.neighbPatch().interfaceInternalField(psiInternal)
    );

    transformCoupleField(pnf, cmpt);

    const labelUList& faceCells = cyclicInterface_.faceCells();

    forAll(faceCells, elemI)
    {
        result[faceCells[elemI]] -= coeffs[elemI]*pnf[elemI];
    }
}


// ************************************************************************* //
