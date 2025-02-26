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

Class
    Foam::radiation::boundaryRadiationProperties

Description
    Boundary radiation properties holder

SourceFiles
    boundaryRadiationProperties.C

\*---------------------------------------------------------------------------*/

#ifndef Foam_boundaryRadiationProperties_H
#define Foam_boundaryRadiationProperties_H

#include "MeshObject.H"
#include "boundaryRadiationPropertiesPatch.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Forward Declarations
class fvMesh;

namespace radiation
{

/*---------------------------------------------------------------------------*\
                 Class boundaryRadiationProperties Declaration
\*---------------------------------------------------------------------------*/

class boundaryRadiationProperties
:
    public MeshObject
    <
        fvMesh,
        GeometricMeshObject,
        boundaryRadiationProperties
    >
{
    // Private Typedefs

        typedef MeshObject
        <
            fvMesh,
            GeometricMeshObject,
            boundaryRadiationProperties
        > MeshObject_type;


    // Private Data

        //- Per patch the boundaryRadiationProperties
        PtrList<boundaryRadiationPropertiesPatch>
            radBoundaryPropertiesPtrList_;

        //- Per (face)zone the boundaryRadiationProperties
        PtrList<boundaryRadiationPropertiesPatch>
            radZonePropertiesPtrList_;


public:

    // Declare name of the class and its debug switch
    TypeName("boundaryRadiationProperties");


    // Constructors

        //- Construct given fvMesh
        explicit boundaryRadiationProperties(const fvMesh& mesh);


    // Member Functions

        //- Return identifiers of face zones activated for boundary radiation
        const labelList faceZoneIds() const
        {
            DynamicList<label> ncZones;

            forAll(radZonePropertiesPtrList_, i)
            {
                if (radZonePropertiesPtrList_.set(i))
                {
                    ncZones.append(i);
                }
            }

            return ncZones;
        }

        //- Access boundary emissivity on patch
        tmp<scalarField> emissivity
        (
            const label patchI,
            const label bandI = 0,
            const vectorField* incomingDirection = nullptr,
            const scalarField* T = nullptr
        ) const;

        //- Access boundary emissivity on face
        scalar faceEmissivity
        (
            const label patchI,
            const label faceI,
            const label bandI = 0,
            vector incomingDirection = Zero,
            scalar T = 0
        ) const;

        //- Access boundary absorptivity on patch
        tmp<scalarField> absorptivity
        (
            const label patchI,
            const label bandI = 0,
            const vectorField* incomingDirection = nullptr,
            const scalarField* T = nullptr
        ) const;

        //- Access boundary absorptivity on face
        scalar faceAbsorptivity
        (
            const label patchI,
            const label faceI,
            const label bandI = 0,
            vector incomingDirection = Zero,
            scalar T = 0
        ) const;

        //- Access boundary transmissivity on patch
        tmp<scalarField> transmissivity
        (
            const label patchI,
            const label bandI = 0,
            const vectorField* incomingDirection = nullptr,
            const scalarField* T = nullptr
        ) const;

        //- Access boundary transmissivity on face
        scalar faceTransmissivity
        (
            const label patchI,
            const label faceI,
            const label bandI = 0,
            vector incomingDirection = Zero,
            scalar T = 0
        ) const;

        // Specialisations for faceZones

            //- Access transmissivity on set of (internal) faces. Zone name only
            //  used to lookup the properties in boundaryRadiationProperties
            tmp<scalarField> zoneTransmissivity
            (
                const label zoneI,
                const labelUList& faceIDs,  // internal faces
                const label bandI = 0,
                vector incomingDirection = Zero,
                scalar T = 0
            ) const;


        //- Access boundary diffuse reflectivity on patch
        tmp<scalarField> diffReflectivity
        (
            const label patchI,
            const label bandI = 0,
            const vectorField* incomingDirection = nullptr,
            const scalarField* T = nullptr
        ) const;

        //- Access boundary diffuse reflectivity on face
        scalar faceDiffReflectivity
        (
            const label patchI,
            const label faceI,
            const label bandI = 0,
            vector incomingDirection = Zero,
            scalar T = 0
        ) const;

        //- Access boundary specular reflectivity on patch
        tmp<scalarField> specReflectivity
        (
            const label patchI,
            const label bandI = 0,
            const vectorField* incomingDirection = nullptr,
            const scalarField* T = nullptr
        ) const;

        //- Access boundary specular reflectivity on face
        scalar faceSpecReflectivity
        (
            const label patchI,
            const label faceI,
            const label bandI = 0,
            vector incomingDirection = Zero,
            scalar T = 0
        ) const;


    //- Destructor
    ~boundaryRadiationProperties() = default;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace radiation
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //


#endif

// ************************************************************************* //
