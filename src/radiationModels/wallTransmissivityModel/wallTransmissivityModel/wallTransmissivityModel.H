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
    Foam::radiation::wallTransmissivityModel

Description
    Base class for wall transmissivity models

\*---------------------------------------------------------------------------*/

#ifndef Foam_radiation_wallTransmissivityModel_H
#define Foam_radiation_wallTransmissivityModel_H

#include "dictionary.H"
#include "autoPtr.H"
#include "runTimeSelectionTables.H"
#include "polyPatch.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace radiation
{

/*---------------------------------------------------------------------------*\
                   Class wallTransmissivityModel Declaration
\*---------------------------------------------------------------------------*/

class wallTransmissivityModel
{

protected:

    // Protected data

        //- Reference to the polyPatch
        const polyPatch& pp_;


public:

    //- Runtime type information
    TypeName("wallTransmissivityModel");

    // Declare runtime constructor selection table

        declareRunTimeSelectionTable
        (
            autoPtr,
            wallTransmissivityModel,
            dictionary,
            (
                const dictionary& dict,
                const polyPatch& pp
            ),
            (dict, pp)
        );


    // Constructors

        //- Construct from components
        wallTransmissivityModel(const dictionary& dict, const polyPatch& pp);


    // Selector

        static autoPtr<wallTransmissivityModel> New
        (
            const dictionary& dict,
            const polyPatch& pp
        );


    //- Destructor
    virtual ~wallTransmissivityModel() = default;


    // Member Functions

        //- Return transmissivity
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
            const vector dir = Zero,
            const scalar T = 0
        ) const = 0;

        //- Is Grey
        virtual bool isGrey() const = 0;

        //- Number of bands
        virtual label nBands() const = 0;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace radiation
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
