/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2018 OpenCFD Ltd.
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
    Foam::radiation::transparent

Description
    Radiation boundary model for transparent walls. It requires to
    specify a wallAbsorptionEmissionModel in boundaryRadiationProperties
    file.

Usage
    \verbatim

        type       transparent;
        wallAbsorptionEmissionModel
        {
            type            multiBandAbsorption;
            absorptivity    (0.3 0.7);
            emissivity      (0.3 0.7);
        };

    \endverbatim

\*---------------------------------------------------------------------------*/

#ifndef Foam_radiation_transparent_H
#define Foam_radiation_transparent_H

#include "boundaryRadiationPropertiesPatch.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace radiation
{

/*---------------------------------------------------------------------------*\
                        Class transparent Declaration
\*---------------------------------------------------------------------------*/

class transparent
:
    public boundaryRadiationPropertiesPatch
{

protected:

    // Protected data

        //- Reference to the polyPatch
        const polyPatch& pp_;


public:

    //- Runtime type information
    TypeName("transparent");

    // Constructors

        //- Construct from components
        transparent(const dictionary& dict, const polyPatch& pp);


    //- Destructor
    virtual ~transparent() = default;


    // Member Functions

        //- Return emissivity
        virtual tmp<scalarField> e
        (
            const label bandI,
            const vectorField* incomingDirection,
            const scalarField* T
        ) const;

        //- Return emissivity on face
        virtual scalar e
        (
            const label faceI,
            const label bandI,
            const vector& dir,
            const scalar T
        ) const;

        //- Return absorptivity on patch
        virtual tmp<scalarField> a
        (
            const label bandI,
            const vectorField* incomingDirection,
            const scalarField* T
        ) const;

        //- Return absorptivity on face
        virtual scalar a
        (
            const label faceI,
            const label bandI,
            const vector& dir,
            const scalar T
        ) const;

        //- Return transmissivity on patch
        virtual tmp<scalarField> t
        (
            const label bandI,
            const vectorField* incomingDirection,
            const scalarField* T
        ) const;

        //- Return transmissivity on face
        virtual scalar t
        (
            const label faceI,
            const label bandI,
            const vector& dir,
            const scalar T
        ) const;

        //- Return specular reflectivity on patch
        virtual tmp<scalarField> rSpec
        (
            const label bandI,
            const vectorField* incomingDirection,
            const scalarField* T
        ) const;

        //- Return specular reflectivity on face
        virtual scalar rSpec
        (
            const label faceI,
            const label bandI,
            const vector& dir,
            const scalar T
        ) const;

        //- Return diffusive reflectivity on patch
        virtual tmp<scalarField> rDiff
        (
            const label bandI,
            const vectorField* incomingDirection,
            const scalarField* T
        ) const;

        //- Return diffusive reflectivity on face
        virtual scalar rDiff
        (
            const label faceI,
            const label bandI,
            const vector& dir,
            const scalar T
        ) const;

        //- Is Grey
        virtual bool isGrey() const;

        //- Number of bands
        virtual label nBands() const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace radiation
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
