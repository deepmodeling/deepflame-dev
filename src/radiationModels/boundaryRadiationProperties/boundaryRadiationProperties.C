/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2015-2022 OpenCFD Ltd.
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

#include "boundaryRadiationProperties.H"
#include "radiationModel.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    namespace radiation
    {
        defineTypeNameAndDebug(boundaryRadiationProperties, 0);
    }
}


// * * * * * * * * * * * * * * * * Constructors * * * * * * * * * * * * * * //

Foam::radiation::boundaryRadiationProperties::boundaryRadiationProperties
(
    const fvMesh& mesh
)
:
    MeshObject_type(mesh),

    radBoundaryPropertiesPtrList_(mesh.boundary().size()),
    radZonePropertiesPtrList_(mesh.faceZones().size())
{
    IOobject boundaryIO
    (
        boundaryRadiationProperties::typeName,
        mesh.time().constant(),
        mesh,
        IOobject::MUST_READ,
        IOobject::NO_WRITE
        // IOobject::NO_REGISTER
    );

    if (boundaryIO.typeHeaderOk<IOdictionary>(true))
    {
        const radiationModel& radiation =
            mesh.lookupObject<radiationModel>
            (
                "radiationProperties"
            );

        // Model number of bands
        label nBands = radiation.nBands();

        // Load in dictionary
        const IOdictionary radiationDict(boundaryIO);


        wordHashSet matchedEntries;

        // Match patches
        {
            const auto& pbm = mesh.boundaryMesh();
            for (const auto& pp : pbm)
            {
                const label patchi = pp.index();

                // Allow wildcard matches
                const auto* eptr =
                    radiationDict.findEntry(pp.name(), keyType::REGEX);

                if (eptr && eptr->isDict())
                {
                    radBoundaryPropertiesPtrList_.set
                    (
                        patchi,
                        boundaryRadiationPropertiesPatch::New(eptr->dict(), pp)
                    );

                    matchedEntries.insert(pp.name());

                    if
                    (
                        nBands
                     != radBoundaryPropertiesPtrList_[patchi].nBands()
                    )
                    {
                        FatalErrorInFunction
                            << "Radiation bands : " <<  nBands << nl
                            << "Bands on patch : " << patchi << " is "
                            << radBoundaryPropertiesPtrList_[patchi].nBands()
                            << abort(FatalError);
                    }
                }
            }
        }


        // Match faceZones if any dictionary entries have not been used for
        // patch matching.
        //
        // Note: radiation properties are hardcoded to take patch reference.
        //       Supply patch0 for now.
        {
            const auto& dummyRef = mesh.boundaryMesh()[0];

            const auto& fzs = mesh.faceZones();

            for (const auto& fz : fzs)
            {
                const label zonei = fz.index();

                if (!matchedEntries.contains(fz.name()))
                {
                    // Note: avoid wildcard matches. Assume user explicitly
                    // provided information for faceZones.

                    const auto* eptr = radiationDict.findEntry
                    (
                        fz.name(),
                        keyType::LITERAL
                    );

                    if (eptr && eptr->isDict())
                    {
                        radZonePropertiesPtrList_.set
                        (
                            zonei,
                            boundaryRadiationPropertiesPatch::New
                            (
                                eptr->dict(),
                                dummyRef
                            )
                        );

                        matchedEntries.insert(fz.name());

                        if
                        (
                            nBands
                         != radZonePropertiesPtrList_[zonei].nBands()
                        )
                        {
                            FatalErrorInFunction
                                << "Radiation bands : " <<  nBands << nl
                                << "Bands on zone : " << zonei << " is "
                                <<  radBoundaryPropertiesPtrList_
                                    [
                                        zonei
                                    ].nBands()
                                << abort(FatalError);
                        }
                    }
                }
            }
        }
    }
}


// * * * * * * * * * * * * * * * Member Functions * * * * * * * * * * * * *  //

Foam::tmp<Foam::scalarField>
Foam::radiation::boundaryRadiationProperties::emissivity
(
    const label patchi,
    const label bandi,
    const vectorField* incomingDirection,
    const scalarField* T
) const
{
    if (radBoundaryPropertiesPtrList_.set(patchi))
    {
        return radBoundaryPropertiesPtrList_[patchi].e
        (
            bandi,
            incomingDirection,
            T
        );
    }

    FatalErrorInFunction
        << "Patch : " << mesh().boundaryMesh()[patchi].name()
        << " is not found in the boundaryRadiationProperties. "
        << "Please add it"
        << exit(FatalError);

    return nullptr;
}


Foam::scalar Foam::radiation::boundaryRadiationProperties::faceEmissivity
(
    const label patchi,
    const label facei,
    const label bandi,
    vector incomingDirection,
    scalar T
) const
{
    if (radBoundaryPropertiesPtrList_.set(patchi))
    {
        return radBoundaryPropertiesPtrList_[patchi].e
        (
            facei,
            bandi,
            incomingDirection,
            T
        );
    }

    FatalErrorInFunction
        << "Patch : " << mesh().boundaryMesh()[patchi].name()
        << " is not found in the boundaryRadiationProperties. "
        << "Please add it"
        << exit(FatalError);

    return Zero;
}


Foam::tmp<Foam::scalarField>
Foam::radiation::boundaryRadiationProperties::absorptivity
(
    const label patchi,
    const label bandi,
    const vectorField* incomingDirection,
    const scalarField* T
) const
{
    if (radBoundaryPropertiesPtrList_.set(patchi))
    {
        return radBoundaryPropertiesPtrList_[patchi].a
        (
            bandi,
            incomingDirection,
            T
        );
    }

    FatalErrorInFunction
        << "Patch : " << mesh().boundaryMesh()[patchi].name()
        << " is not found in the boundaryRadiationProperties. "
        << "Please add it"
        << exit(FatalError);

    return nullptr;
}


Foam::scalar Foam::radiation::boundaryRadiationProperties::faceAbsorptivity
(
    const label patchi,
    const label facei,
    const label bandi,
    vector incomingDirection,
    scalar T
) const
{
    if (radBoundaryPropertiesPtrList_.set(patchi))
    {
        return radBoundaryPropertiesPtrList_[patchi].a
        (
            facei,
            bandi,
            incomingDirection,
            T
        );
    }

    FatalErrorInFunction
        << "Patch : " << mesh().boundaryMesh()[patchi].name()
        << " is not found in the boundaryRadiationProperties. "
        << "Please add it"
        << exit(FatalError);

    return Zero;
}


Foam::tmp<Foam::scalarField>
Foam::radiation::boundaryRadiationProperties::transmissivity
(
    const label patchi,
    const label bandi,
    const vectorField* incomingDirection,
    const scalarField* T
) const
{
    if (radBoundaryPropertiesPtrList_.set(patchi))
    {
        return radBoundaryPropertiesPtrList_[patchi].t
        (
            bandi,
            incomingDirection,
            T
        );
    }

    FatalErrorInFunction
        << "Patch : " << mesh().boundaryMesh()[patchi].name()
        << " is not found in the boundaryRadiationProperties. "
        << "Please add it"
        << exit(FatalError);

    return nullptr;
}


Foam::scalar Foam::radiation::boundaryRadiationProperties::faceTransmissivity
(
    const label patchi,
    const label facei,
    const label bandi,
    vector incomingDirection,
    scalar T
) const
{
    if (radBoundaryPropertiesPtrList_.set(patchi))
    {
        return radBoundaryPropertiesPtrList_[patchi].t
        (
            facei,
            bandi,
            incomingDirection,
            T
        );
    }

    FatalErrorInFunction
        << "Patch : " << mesh().boundaryMesh()[patchi].name()
        << " is not found in the boundaryRadiationProperties. "
        << "Please add it"
        << exit(FatalError);

    return Zero;
}


Foam::tmp<Foam::scalarField>
Foam::radiation::boundaryRadiationProperties::zoneTransmissivity
(
    const label zonei,
    const labelUList& faceIDs,
    const label bandi,
    vector incomingDirection,
    scalar T
) const
{
    if (radZonePropertiesPtrList_.set(zonei))
    {
        auto tfld = tmp<scalarField>::New(faceIDs.size());
        auto& fld = tfld.ref();
        forAll(fld, i)
        {
            fld[i] = radZonePropertiesPtrList_[zonei].t
            (
                faceIDs[i],
                bandi,
                incomingDirection,
                T
            );
        }
        return tfld;
    }

    FatalErrorInFunction
        << "Zone : " << mesh().faceZones()[zonei].name()
        << " is not found in the boundaryRadiationProperties. "
        << "Please add it"
        << exit(FatalError);

    return nullptr;
}


Foam::tmp<Foam::scalarField>
Foam::radiation::boundaryRadiationProperties::diffReflectivity
(
    const label patchi,
    const label bandi,
    const vectorField* incomingDirection,
    const scalarField* T
) const
{
    if (radBoundaryPropertiesPtrList_.set(patchi))
    {
        return radBoundaryPropertiesPtrList_[patchi].rDiff
        (
            bandi,
            incomingDirection,
            T
        );
    }

    FatalErrorInFunction
        << "Patch : " << mesh().boundaryMesh()[patchi].name()
        << " is not found in the boundaryRadiationProperties. "
        << "Please add it"
        << exit(FatalError);

    return nullptr;
}


Foam::scalar Foam::radiation::boundaryRadiationProperties::faceDiffReflectivity
(
    const label patchi,
    const label facei,
    const label bandi,
    vector incomingDirection,
    scalar T
) const
{
    if (radBoundaryPropertiesPtrList_.set(patchi))
    {
        return radBoundaryPropertiesPtrList_[patchi].rDiff
        (
            facei,
            bandi,
            incomingDirection,
            T
        );
    }

    FatalErrorInFunction
        << "Patch : " << mesh().boundaryMesh()[patchi].name()
        << " is not found in the boundaryRadiationProperties. "
        << "Please add it"
        << exit(FatalError);

    return Zero;
}


Foam::tmp<Foam::scalarField>
Foam::radiation::boundaryRadiationProperties::specReflectivity
(
    const label patchi,
    const label bandi,
    const vectorField* incomingDirection,
    const scalarField* T
) const
{
    if (radBoundaryPropertiesPtrList_.set(patchi))
    {
        return radBoundaryPropertiesPtrList_[patchi].rSpec
        (
            bandi,
            incomingDirection,
            T
        );
    }

    FatalErrorInFunction
        << "Patch : " << mesh().boundaryMesh()[patchi].name()
        << " is not found in the boundaryRadiationProperties. "
        << "Please add it"
        << exit(FatalError);

    return nullptr;
}


Foam::scalar Foam::radiation::boundaryRadiationProperties::faceSpecReflectivity
(
    const label patchi,
    const label facei,
    const label bandi,
    vector incomingDirection,
    scalar T
) const
{
    if (radBoundaryPropertiesPtrList_.set(patchi))
    {
        return radBoundaryPropertiesPtrList_[patchi].rSpec
        (
            facei,
            bandi,
            incomingDirection,
            T
        );
    }

    FatalErrorInFunction
        << "Patch : " << mesh().boundaryMesh()[patchi].name()
        << " is not found in the boundaryRadiationProperties. "
        << "Please add it"
        << exit(FatalError);

    return Zero;
}


// ************************************************************************* //
