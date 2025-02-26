/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2018, 2024 OpenCFD Ltd.
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
    Foam::radiation::constantAbsorption

Group
    grpRadiationTransmissivitySubModels

Description
    Constant radiation transmissivity coefficient

SourceFiles
    constantAbsorption.C

\*---------------------------------------------------------------------------*/

#ifndef Foam_radiation_constantAbsorption_H
#define Foam_radiation_constantAbsorption_H

#include "wallAbsorptionEmissionModel.H"
#include "Function1.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace radiation
{

/*---------------------------------------------------------------------------*\
                   Class constantAbsorption Declaration
\*---------------------------------------------------------------------------*/

class constantAbsorption
:
    public wallAbsorptionEmissionModel
{

    // Private data

        //- Coefficients dictionary
        dictionary coeffsDict_;

        //- Absorptivity coefficient
        autoPtr<Function1<scalar>> a_;

        //- Emissivity coefficient
        autoPtr<Function1<scalar>> e_;


public:

    //- Runtime type information
    TypeName("constantAbsorption");


    // Constructors

        //- Construct from components
        constantAbsorption(const dictionary& dict, const polyPatch& pp);


    //- Destructor
    virtual ~constantAbsorption() = default;


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
        inline label nBands() const
        {
            return 1;
        }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace radiation
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
