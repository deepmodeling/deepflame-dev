/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2018 OpenFOAM Foundation
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
    Foam::faceReflecting

Description

    Calculates the reflecting faces from specular surfaces. It only
    takes into account the first reflection coming from a surface hit
    by the primary Sun hit faces.

SourceFiles
    faceReflecting.C

\*---------------------------------------------------------------------------*/

#ifndef faceReflecting_H
#define faceReflecting_H

#include "fvMesh.H"
#include "wallPolyPatch.H"
#include "triSurfaceTools.H"
#include "vectorList.H"
#include "distributedTriSurfaceMesh.H"
#include "faceShading.H"
#include "solarCalculator.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                        Class faceReflecting Declaration
\*---------------------------------------------------------------------------*/

class faceReflecting
{
    // Private data

        //- Reference to mesh
        const fvMesh& mesh_;

         //- Number of solid angles in theta
        label nTheta_;

        //- Number of solid angles in phi
        label nPhi_ ;

        //- Total number of rays (1 per direction)
        label nRay_;

        //- Discretised angles for reflected rays
        vectorList refDiscAngles_;

        //- Spectral distribution for the integrated solar heat flux
        const scalarList& spectralDistribution_;

        //- Net reflected radiative heat flux [W/m2]
        PtrList<volScalarField> qreflective_;

        //- Primary hits faces
        const faceShading& directHitFaces_;

        //- Distributed surface for ray tracing
        autoPtr<distributedTriSurfaceMesh> surfacesMesh_;

        //- Faces from which rays are shot
        autoPtr<labelList> shootFacesIds_;

        //- Face centres from which rays are shot
        autoPtr<pointField> Cfs_;

        //- Face normal from which rays are shot
        autoPtr<vectorField> Nfs_;

        //- Solar calculator
        const solarCalculator& solarCalc_;

        //- Shooting rays pacthes
        labelHashSet includePatches_;

        //- Map from triSurface Index to global
        labelList mapTriToGlobal_;


    // Private members

        //- Calculate ray start faces
        void calculate();

        //- Initialise model
        void initialise(const dictionary& dict);


        //- No copy construct
        faceReflecting(const faceReflecting&) = delete;

        //- No copy assignment
        void operator=(const faceReflecting&) = delete;


public:

    // Declare name of the class and its debug switch
    ClassName("faceReflecting");


    // Constructors

        //- Construct from components
        faceReflecting
        (
            const fvMesh& mesh,
            const faceShading& directHiyFaces,
            const solarCalculator& ,
            const scalarList& spectralDistribution,
            const dictionary& dict
        );



    //- Destructor
    ~faceReflecting() = default;


    // Member Functions

        // Const access to qreflective
        const volScalarField& qreflective(const label bandI) const
        {
            return qreflective_[bandI];
        }


        //- Correct reflected flux
        void correct();
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
