/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2015-2019 OpenCFD Ltd.
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

#include "multiBandAbsorptionEmission.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    namespace radiation
    {
        defineTypeNameAndDebug(multiBandAbsorptionEmission, 0);

        addToRunTimeSelectionTable
        (
            absorptionEmissionModel,
            multiBandAbsorptionEmission,
            dictionary
        );
    }
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::radiation::multiBandAbsorptionEmission::multiBandAbsorptionEmission
(
    const dictionary& dict,
    const fvMesh& mesh
)
:
    absorptionEmissionModel(dict, mesh),
    coeffsDict_(dict.subDict(typeName + "Coeffs")),
    absCoeffs_(maxBands_),
    emiCoeffs_(maxBands_),
    nBands_(0)
{
    coeffsDict_.readEntry("absorptivity", absCoeffs_);
    coeffsDict_.readEntry("emissivity", emiCoeffs_);
    nBands_ = absCoeffs_.size();
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::radiation::multiBandAbsorptionEmission::
~multiBandAbsorptionEmission()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::tmp<Foam::volScalarField>
Foam::radiation::multiBandAbsorptionEmission::aCont
(
    const label bandI
) const
{
    return volScalarField::New
    (
        "a",
        // IOobject::NO_REGISTER,
        mesh(),
        dimensionedScalar("a", dimless/dimLength, absCoeffs_[bandI])
    );
}


Foam::tmp<Foam::volScalarField>
Foam::radiation::multiBandAbsorptionEmission::eCont
(
    const label bandI
) const
{
    return volScalarField::New
    (
        "e",
        // IOobject::NO_REGISTER,
        mesh(),
        dimensionedScalar("e", dimless/dimLength, emiCoeffs_[bandI])
    );
}


Foam::tmp<Foam::volScalarField>
Foam::radiation::multiBandAbsorptionEmission::ECont
(
    const label bandI
) const
{
    return volScalarField::New
    (
        "E",
        // IOobject::NO_REGISTER,
        mesh(),
        dimensionedScalar(dimMass/dimLength/pow3(dimTime), Zero)
    );
}


// ************************************************************************* //
