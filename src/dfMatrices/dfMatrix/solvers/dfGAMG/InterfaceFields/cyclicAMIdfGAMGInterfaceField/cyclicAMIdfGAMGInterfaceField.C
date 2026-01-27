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

#include "cyclicAMIdfGAMGInterfaceField.H"
#include "addToRunTimeSelectionTable.H"
#include "lduMatrix.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(cyclicAMIdfGAMGInterfaceField, 0);
    addToRunTimeSelectionTable
    (
        dfGAMGInterfaceField,
        cyclicAMIdfGAMGInterfaceField,
        lduInterface
    );
    addToRunTimeSelectionTable
    (
        dfGAMGInterfaceField,
        cyclicAMIdfGAMGInterfaceField,
        lduInterfaceField
    );
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::cyclicAMIdfGAMGInterfaceField::cyclicAMIdfGAMGInterfaceField
(
    const dfGAMGInterface& dfGAMGCp,
    const lduInterfaceField& fineInterface
)
:
    dfGAMGInterfaceField(dfGAMGCp, fineInterface),
    cyclicAMIInterface_(refCast<const cyclicAMIdfGAMGInterface>(dfGAMGCp)),
    doTransform_(false),
    rank_(0)
{
    const cyclicAMILduInterfaceField& p =
        refCast<const cyclicAMILduInterfaceField>(fineInterface);

    doTransform_ = p.doTransform();
    rank_ = p.rank();
}


Foam::cyclicAMIdfGAMGInterfaceField::cyclicAMIdfGAMGInterfaceField
(
    const dfGAMGInterface& dfGAMGCp,
    const bool doTransform,
    const int rank
)
:
    dfGAMGInterfaceField(dfGAMGCp, doTransform, rank),
    cyclicAMIInterface_(refCast<const cyclicAMIdfGAMGInterface>(dfGAMGCp)),
    doTransform_(doTransform),
    rank_(rank)
{}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::cyclicAMIdfGAMGInterfaceField::~cyclicAMIdfGAMGInterfaceField()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::cyclicAMIdfGAMGInterfaceField::updateInterfaceMatrix
(
    scalarField& result,
    const scalarField& psiInternal,
    const scalarField& coeffs,
    const direction cmpt,
    const Pstream::commsTypes
) const
{
    const cyclicAMIdfGAMGInterface& thisInterface = cyclicAMIInterface_;
    const cyclicAMIdfGAMGInterface& neiInterface = thisInterface.neighbPatch();

    // Get neighbouring field
    scalarField pnf(neiInterface.interfaceInternalField(psiInternal));

    // Transform according to the transformation tensors
    transformCoupleField(pnf, cmpt);

    // Transform and interpolate
    scalarField pf(size(), Zero);
    if (thisInterface.owner())
    {
        forAll(thisInterface.AMIs(), i)
        {
            const scalar r =
                pow
                (
                    inv(thisInterface.AMITransforms()[i]).R()(cmpt, cmpt),
                    rank()
                );

            pf += thisInterface.AMIs()[i].interpolateToSource(r*pnf);
        }
    }
    else
    {
        forAll(neiInterface.AMIs(), i)
        {
            const scalar r =
                pow
                (
                    neiInterface.AMITransforms()[i].R()(cmpt, cmpt),
                    rank()
                );

            pf += neiInterface.AMIs()[i].interpolateToTarget(r*pnf);
        }
    }

    // Multiply the field by coefficients and add into the result
    const labelUList& faceCells = cyclicAMIInterface_.faceCells();
    forAll(faceCells, elemI)
    {
        result[faceCells[elemI]] -= coeffs[elemI]*pf[elemI];
    }
}


// ************************************************************************* //
