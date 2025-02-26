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
    Foam::radiation::multiBandTransmissivity

Group
    grpwallTransmissivityModel

Description
    multiBandTransmissivity radiation transmissivity for solids.

Usage
    \verbatim

        wallTransmissivityModel
        {
            type            multiBandTransmissivity;
            transmissivity  (0.3 0.7);
        };

    \endverbatim

SourceFiles
    multiBandTransmissivity.C

\*---------------------------------------------------------------------------*/

#ifndef Foam_radiation_multiBandTransmissivity_H
#define Foam_radiation_multiBandTransmissivity_H

#include "wallTransmissivityModel.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace radiation
{

/*---------------------------------------------------------------------------*\
                  Class multiBandTransmissivity Declaration
\*---------------------------------------------------------------------------*/

class multiBandTransmissivity
:
    public wallTransmissivityModel
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
        scalarList tauCoeffs_;

        //- Bands
        label nBands_;


public:

    //- Runtime type information
    TypeName("multiBandTransmissivity");


    // Constructors

        //- Construct from components
        multiBandTransmissivity
        (
            const dictionary& dict,
            const polyPatch& p
        );


    //- Destructor
    virtual ~multiBandTransmissivity() = default;


    // Member Functions

        // Access

            //- Return transmissivity
            tmp<scalarField> t
            (
                const label bandI = 0,
                const vectorField* incomingDirection = nullptr,
                const scalarField* T = nullptr
            ) const;

            //- Return transmissivity on face
            scalar t
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
