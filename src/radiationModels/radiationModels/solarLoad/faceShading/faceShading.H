/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2015 OpenFOAM Foundation
    Copyright (C) 2017-2022 OpenCFD Ltd.
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
    Foam::faceShading

Description
    Helper class to calculate visible faces for global, sun-like illumination.

    faceShading uses the transmissivity value in the boundaryRadiationProperties
    in order to evaluate which faces are "hit" by the "direction" vector.
    NOTE: Only transmissivity values of zero are considered for opaque walls.

SourceFiles
    faceShading.C

\*---------------------------------------------------------------------------*/

#ifndef faceShading_H
#define faceShading_H

#include "boundaryRadiationProperties.H"
#include "DynamicField.H"
#include "triSurface.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Forward Declarations
class fvMesh;
class polyMesh;

/*---------------------------------------------------------------------------*\
                           Class faceShading Declaration
\*---------------------------------------------------------------------------*/

class faceShading
{
    // Private Data

        //- Reference to mesh
        const fvMesh& mesh_;

        //- Patches to check for visibility
        const labelList patchIDs_;

        //- FaceZones to check for visibility
        const labelList zoneIDs_;

        //- Direction
        vector direction_;

        //- Faces directly hit by vector direction
        labelList rayStartFaces_;


    // Private Member Functions

        //- Calculate ray start faces
        void calculate();

        //- Construct a triSurface from selected faces on patches or faceZones.
        //  Resulting surface has regioning with mesh patches first, faceZones
        //  second.
        triSurface triangulate
        (
            const labelUList& faceIDs,
            const PackedBoolList& flipMap
        ) const;

        //- Find opaque faces by looking at the radiation properties.
        PackedBoolList selectOpaqueFaces
        (
            const radiation::boundaryRadiationProperties& boundaryRadiation,
            const labelUList& patchIDs,
            const labelUList& zoneIDs
        ) const;

        //- Pick up candidate faces that might be blocking visibility for
        //  other faces:
        //  - if transmissivity is 0
        //  - (optional) if pointing wrong way
        void selectFaces
        (
            const bool useNormal,
            const PackedBoolList& isCandidateFace,
            const labelUList& patchIDs,
            const labelUList& zoneIDs,

            labelList& faceIDs,
            PackedBoolList& flipMap
        ) const;

        //- Write rays
        void writeRays
        (
            const fileName& fName,
            const DynamicField<point>& endCf,
            const pointField& myFc
        );

        //- No copy construct
        faceShading(const faceShading&) = delete;

        //- No copy assignment
        void operator=(const faceShading&) = delete;


public:

    // Declare name of the class and its debug switch
    ClassName("faceShading");


    // Constructors

        //- Helper: return all uncoupled patches
        static labelList nonCoupledPatches(const polyMesh& mesh);

        //- Construct from mesh and vector to 'sun'. All uncoupled patches
        //  are checked for visibility. faceZones are ignored.
        faceShading
        (
            const fvMesh& mesh,
            const vector& dir
        );

        //- Construct from mesh and vector to 'sun' and selected patches
        //  and faceZones.
        faceShading
        (
            const fvMesh& mesh,
            const labelList& patchIDs,
            const labelList& zoneIDs,
            const vector& dir
        );


    //- Destructor
    ~faceShading() = default;


    // Member Functions

        // Access

            //- Const access to direction
            const vector direction() const
            {
                return direction_;
            }

            //- Non-const access to direction
            vector& direction()
            {
                return direction_;
            }

            //- Const access to rayStartFaces
            const labelList& rayStartFaces() const
            {
                return rayStartFaces_;
            }


        //- Recalculate rayStartFaces using direction vector
        void correct();
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
