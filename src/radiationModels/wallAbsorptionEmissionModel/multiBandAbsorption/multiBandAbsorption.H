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
    Foam::radiation::multiBandAbsorption

Group
    wallAbsorptionEmissionModel

Description
    multiBandAbsorption radiation transmissivity for solids.

Usage
    \verbatim

        wallAbsorptionEmissionModel
        {
            type            multiBandAbsorption;
            absorptivity    (0.3 0.7);
            emissivity      (0.3 0.7);
        };

    \endverbatim

SourceFiles
    multiBandAbsorption.C

\*---------------------------------------------------------------------------*/

#ifndef Foam_radiation_multiBandAbsorption_H
#define Foam_radiation_multiBandAbsorption_H

#include "wallAbsorptionEmissionModel.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace radiation
{

/*---------------------------------------------------------------------------*\
                  Class multiBandAbsorption Declaration
\*---------------------------------------------------------------------------*/

class multiBandAbsorption
:
    public wallAbsorptionEmissionModel
{
public:

    // Public data

        //- Maximum number of bands
        static const label maxBands_ = 5;


private:

    // Private data

        //- Absorption model dictionary
        dictionary coeffsDict_;

        //-  Absorption coefficients
        scalarList aCoeffs_;

        //-  Emission coefficients
        scalarList eCoeffs_;

        //- Bands
        label nBands_;


public:

    //- Runtime type information
    TypeName("multiBandAbsorption");


    // Constructors

        //- Construct from components
        multiBandAbsorption
        (
            const dictionary& dict,
            const polyPatch& pp
        );


    //- Destructor
    virtual ~multiBandAbsorption();


    // Member Functions

        // Access

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
                return false;
            }

            //- Number of bands
            inline label nBands() const
            {
                return nBands_;
            }

};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace radiation
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
