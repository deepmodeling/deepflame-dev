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

#include "multiBandZoneAbsorptionEmission.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    namespace radiation
    {
        defineTypeNameAndDebug(multiBandZoneAbsorptionEmission, 0);

        addToRunTimeSelectionTable
        (
            absorptionEmissionModel,
            multiBandZoneAbsorptionEmission,
            dictionary
        );
    }
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::radiation::multiBandZoneAbsorptionEmission::
multiBandZoneAbsorptionEmission
(
    const dictionary& dict,
    const fvMesh& mesh
)
:
    absorptionEmissionModel(dict, mesh),
    coeffsDict_(dict.subDict(typeName + "Coeffs")),
    absCoeffs_(maxBands_),
    emiCoeffs_(maxBands_),
    nBands_(0),
    zoneAbsorptivity_(),
    zoneEmissivity_(),
    zoneIds_()
{
    coeffsDict_.readEntry("absorptivity", absCoeffs_);
    coeffsDict_.readEntry("emissivity", emiCoeffs_);
    nBands_ = absCoeffs_.size();

    const dictionary& zoneDict = coeffsDict_.subDict("zones");

    zoneDict.readEntry("absorptivity", zoneAbsorptivity_);
    zoneDict.readEntry("emissivity", zoneEmissivity_);

    zoneIds_.resize(zoneAbsorptivity_.size(), -1);

    label numZones = 0;
    forAllConstIters(zoneAbsorptivity_, iter)
    {
        label zoneID = mesh.cellZones().findZoneID(iter.key());
        if (zoneID == -1)
        {
            FatalErrorInFunction
                << "Cannot find cellZone " << iter.key() << endl
                << "Valid cellZones are " << mesh.cellZones().names()
                << exit(FatalError);
        }
        zoneIds_[numZones] = zoneID;
        ++numZones;
    }
    // zoneIds_.resize(numZones);
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::radiation::multiBandZoneAbsorptionEmission::
~multiBandZoneAbsorptionEmission()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::tmp<Foam::volScalarField>
Foam::radiation::multiBandZoneAbsorptionEmission::aCont
(
    const label bandI
) const
{
    auto ta = volScalarField::New
    (
        "a",
        // IOobject::NO_REGISTER,
        mesh(),
        dimensionedScalar("a", dimless/dimLength, absCoeffs_[bandI])
    );
    scalarField& a = ta.ref().primitiveFieldRef();

    for (const label zonei : zoneIds_)
    {
        const cellZone& zn = mesh().cellZones()[zonei];
        const auto iter = zoneAbsorptivity_.cfind(zn.name());

        if (iter.good())  // Check is redundant (cannot fail)
        {
            const scalarList& absorb = iter.val();

            UIndirectList<scalar>(a, zn) = absorb[bandI];
        }
    }

    return ta;
}


Foam::tmp<Foam::volScalarField>
Foam::radiation::multiBandZoneAbsorptionEmission::eCont
(
    const label bandI
) const
{
    auto te = volScalarField::New
    (
        "e",
        // IOobject::NO_REGISTER,
        mesh(),
        dimensionedScalar("e", dimless/dimLength, emiCoeffs_[bandI])
    );
    scalarField& e = te.ref().primitiveFieldRef();

    for (const label zonei : zoneIds_)
    {
        const cellZone& zn = mesh().cellZones()[zonei];
        const auto iter = zoneEmissivity_.cfind(zn.name());

        if (iter.good())  // Check is redundant (cannot fail)
        {
            const scalarList& emit = iter.val();

            UIndirectList<scalar>(e, zn) = emit[bandI];
        }
    }

    return te;
}


Foam::tmp<Foam::volScalarField>
Foam::radiation::multiBandZoneAbsorptionEmission::ECont
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
