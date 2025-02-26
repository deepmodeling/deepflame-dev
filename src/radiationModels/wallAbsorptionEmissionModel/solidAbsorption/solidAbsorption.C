/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2015-2018 OpenCFD Ltd.
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

#include "solidAbsorption.H"
#include "addToRunTimeSelectionTable.H"
#include "mappedPatchBase.H"
#include "radiationModel.H"
#include "absorptionEmissionModel.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    namespace radiation
    {
        defineTypeNameAndDebug(solidAbsorption, 0);

        addToRunTimeSelectionTable
        (
            wallAbsorptionEmissionModel,
            solidAbsorption,
            dictionary
        );
    }
}


// * * * * * * * * * * * * * * * * Private members * * * * * * * * * * * * * //

const Foam::fvMesh& Foam::radiation::solidAbsorption::nbrRegion() const
{
    const mappedPatchBase& mpp = refCast<const mappedPatchBase>(pp_);
    return (refCast<const fvMesh>(mpp.sampleMesh()));
}


Foam::label Foam::radiation::solidAbsorption::nbrPatchIndex() const
{
    const mappedPatchBase& mpp = refCast<const mappedPatchBase>(pp_);
    return (mpp.samplePolyPatch().index());
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::radiation::solidAbsorption::solidAbsorption
(
    const dictionary& dict,
    const polyPatch& pp
)
:
    wallAbsorptionEmissionModel(dict, pp)
{
    if (!isA<mappedPatchBase>(pp))
    {
        FatalErrorInFunction
            << "\n    patch type '" << pp.type()
            << "' not type '" << mappedPatchBase::typeName << "'"
            << "\n    for patch " << pp.name()
            << abort(FatalIOError);
    }
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::radiation::solidAbsorption::~solidAbsorption()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::tmp<Foam::scalarField> Foam::radiation::solidAbsorption::a
(
    const label bandI,
    const vectorField* incomingDirection,
    const scalarField* T
) const
{
    // Since we're inside initEvaluate/evaluate there might be processor
    // comms underway. Change the tag we use.
    const int oldTag = UPstream::incrMsgType();

    const fvMesh& nbrMesh = nbrRegion();

    const radiation::radiationModel& radiation =
        nbrMesh.lookupObject<radiation::radiationModel>
        (
            "radiationProperties"
        );

    scalarField absorptivity
    (
        radiation.absorptionEmission().a
        (
            bandI
        )().boundaryField()
        [
            nbrPatchIndex()
        ]
    );

    const mappedPatchBase& mpp = refCast<const mappedPatchBase>(pp_);

    mpp.distribute(absorptivity);

    UPstream::msgType(oldTag);  // Restore tag

    return tmp<scalarField>::New(std::move(absorptivity));
}


Foam::scalar Foam::radiation::solidAbsorption::a
(
    const label faceI,
    const label bandI,
    const vector dir,
    const scalar T
) const
{
    return a(bandI, nullptr, nullptr)()[faceI];
}

Foam::tmp<Foam::scalarField> Foam::radiation::solidAbsorption::e
(
    const label bandI,
    const vectorField* incomingDirection,
    const scalarField* T
) const
{

    // Since we're inside initEvaluate/evaluate there might be processor
    // comms underway. Change the tag we use.
    const int oldTag = UPstream::incrMsgType();

    const fvMesh& nbrMesh = nbrRegion();

    const radiation::radiationModel& radiation =
        nbrMesh.lookupObject<radiation::radiationModel>
        (
            "radiationProperties"
        );

    scalarField emissivity
    (
        radiation.absorptionEmission().e
        (
            bandI
        )().boundaryField()
        [
            nbrPatchIndex()
        ]
    );

    const mappedPatchBase& mpp = refCast<const mappedPatchBase>(pp_);

    mpp.distribute(emissivity);

    UPstream::msgType(oldTag);  // Restore tag

    return tmp<scalarField>::New(std::move(emissivity));
}


Foam::scalar Foam::radiation::solidAbsorption::e
(
    const label faceI,
    const label bandI,
    const vector dir,
    const scalar T
) const
{
    return e(bandI, nullptr, nullptr)()[faceI];
}


Foam::label Foam::radiation::solidAbsorption::nBands() const
{
     const fvMesh& nbrMesh = nbrRegion();

     const radiation::radiationModel& radiation =
        nbrMesh.lookupObject<radiation::radiationModel>
        (
            "radiationProperties"
        );

    return (radiation.absorptionEmission().nBands());
}
// ************************************************************************* //
