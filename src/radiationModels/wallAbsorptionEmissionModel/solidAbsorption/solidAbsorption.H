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

Class
    Foam::radiation::solidAbsorption

Description
    Radiation absorptivity-emissivity model to be used on walls on
    inter-region patches when the solid opaque radiation model is used
    in the solid and the wall emissivity and absorptivity are taken from
    the solid radiation properties

Usage
    Example usage
    \verbatim
        wallAbsorptionEmissionModel
        {
            type            solidAbsorption;
        };
    \endverbatim


SourceFiles
    solidAbsorption.C

\*---------------------------------------------------------------------------*/

#ifndef Foam_radiation_solidAbsorption_H
#define Foam_radiation_solidAbsorption_H

#include "wallAbsorptionEmissionModel.H"
#include "fvMesh.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace radiation
{

/*---------------------------------------------------------------------------*\
                        Class solidAbsorption Declaration
\*---------------------------------------------------------------------------*/

class solidAbsorption
:
    public wallAbsorptionEmissionModel
{

    // Private members

        //- Nbr region
        const fvMesh& nbrRegion() const;

        //- Nbr index patch
        label nbrPatchIndex() const;

public:

    //- Runtime type information
    TypeName("solidAbsorption");


    // Constructors

        //- Construct from components
        solidAbsorption(const dictionary& dict, const polyPatch& pp);


    //- Destructor
    virtual ~solidAbsorption();


    // Member Functions

        //- absorptivity coefficient
        tmp<scalarField> a
        (
            const label bandI = 0,
            const vectorField* incomingDirection = nullptr,
            const scalarField* T = nullptr
        ) const;

        //- Return absorptivity on face
        scalar a
        (
            const label faceI,
            const label bandI = 0,
            const vector dir = Zero,
            const scalar T = 0
        ) const;

        //- Return emission coefficient
        tmp<scalarField> e
        (
            const label bandI = 0,
            const vectorField* incomingDirection = nullptr,
            const scalarField* T = nullptr
        ) const;

        //- Return emission coefficient
        scalar e
        (
            const label faceI,
            const label bandI = 0,
            const vector dir = Zero,
            const scalar T = 0
        ) const;


        //- Is Grey
        inline bool isGrey() const
        {
            return true;
        }

        //- Number of bands
        inline label nBands() const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace radiation
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
