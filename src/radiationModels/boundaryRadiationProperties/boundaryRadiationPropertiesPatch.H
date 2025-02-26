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
    Foam::boundaryRadiationPropertiesPatch

Description

SourceFiles
    boundaryRadiationPropertiesPatch.C

\*---------------------------------------------------------------------------*/

#ifndef Foam_radiation_boundaryRadiationPropertiesPatch_H
#define Foam_radiation_boundaryRadiationPropertiesPatch_H

#include "scalarField.H"
//#include "Enum.H"
#include "fvPatch.H"
#include "calculatedFvPatchFields.H"
#include "runTimeSelectionTables.H"
#include "wallTransmissivityModel.H"
#include "wallAbsorptionEmissionModel.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace radiation
{
/*---------------------------------------------------------------------------*\
              Class boundaryRadiationPropertiesPatch Declaration
\*---------------------------------------------------------------------------*/

class boundaryRadiationPropertiesPatch
{
    // Private Data

        //- Dictionary
        const dictionary dict_;

        //- reference to fvPatch
        const polyPatch& patch_;


    // Private Member Functions

        //- Return nbr patch index
        label nbrPatchIndex() const;

        //- Return nbr mesh
        const fvMesh& nbrRegion() const;


protected:

    // Protected Member Functions

        //- Absorption/emission model
        autoPtr<wallAbsorptionEmissionModel> absorptionEmission_;

        //- Transmissivity model
        autoPtr<wallTransmissivityModel> transmissivity_;


public:

    //- Runtime type information
    TypeName("boundaryRadiationPropertiesPatch");

      // Declare runtime constructor selection table

        declareRunTimeSelectionTable
        (
            autoPtr,
            boundaryRadiationPropertiesPatch,
            dictionary,
            (
                const dictionary& dict,
                const polyPatch& pp
            ),
            (dict, pp)
        );


    // Constructors

        //- Construct from patch, internal field and dictionary
        boundaryRadiationPropertiesPatch
        (
            const dictionary&,
            const polyPatch&
        );


        //- Selector
        static autoPtr<boundaryRadiationPropertiesPatch> New
        (
            const dictionary& dict,
            const polyPatch& pp
        );

        //- Destructor
        virtual ~boundaryRadiationPropertiesPatch() = default;


    // Member Functions

        //- Return absorptionEmissionModel
        const wallAbsorptionEmissionModel& absorptionEmission() const;

        //- Return transmissivityModel
        const wallTransmissivityModel& transmissiveModel() const;

        //- Return emissivity on patch
        virtual tmp<scalarField> e
        (
            const label bandI = 0,
            const vectorField* incomingDirection = nullptr,
            const scalarField* T = nullptr
        ) const = 0;

        //- Return emissivity on face
        virtual scalar e
        (
            const label faceI,
            const label bandI = 0,
            const vector& dir = Zero,
            const scalar T = 0
        ) const = 0;

        //- Return absorptivity on patch
        virtual tmp<scalarField> a
        (
            const label bandI = 0,
            const vectorField* incomingDirection = nullptr,
            const scalarField* T = nullptr
        ) const = 0;

        //- Return absorptivity on face
        virtual scalar a
        (
            const label faceI,
            const label bandI = 0,
            const vector& dir = Zero,
            const scalar T = 0
        ) const = 0;

        //- Return transmissivity on patch
        virtual tmp<scalarField> t
        (
            const label bandI = 0,
            const vectorField* incomingDirection = nullptr,
            const scalarField* T = nullptr
        ) const = 0;

        //- Return transmissivity on face
        virtual scalar t
        (
            const label faceI,
            const label bandI = 0,
            const vector& dir = Zero,
            const scalar T = 0
        ) const = 0;

        //- Return specular reflectivity on patch
        virtual tmp<scalarField> rSpec
        (
            const label bandI = 0,
            const vectorField* incomingDirection = nullptr,
            const scalarField* T = nullptr
        ) const = 0;

        //- Return specular reflectivity on face
        virtual scalar rSpec
        (
            const label faceI,
            const label bandI = 0,
            const vector& dir = Zero,
            const scalar T = 0
        ) const = 0;

        //- Return diffusive reflectivity on patch
        virtual tmp<scalarField> rDiff
        (
            const label bandI = 0,
            const vectorField* incomingDirection = nullptr,
            const scalarField* T = nullptr
        ) const = 0;

        //- Return diffusive reflectivity on face
        virtual scalar rDiff
        (
            const label faceI,
            const label bandI = 0,
            const vector& dir = Zero,
            const scalar T = 0
        ) const = 0;

        //- Is Grey
        virtual bool isGrey() const = 0;

        //- Number of bands
        virtual label nBands() const = 0;

        //- Write
        virtual void write(Ostream&) const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam
} // End namespace radiation

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
