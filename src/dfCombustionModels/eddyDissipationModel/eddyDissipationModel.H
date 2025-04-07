/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2015 OpenFOAM Foundation
     \\/     M anipulation  |
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
    Foam::combustionModels::eddyDissipationModel

Group
    grpCombustionModels

Description
    Simple infinitely fast chemistry combustion model based on the principle
    mixed is burnt. Additional parameter C is used to distribute the heat
    release rate.in time

SourceFiles
    eddyDissipationModel.C

\*---------------------------------------------------------------------------*/

#ifndef eddyDissipationModel_H
#define eddyDissipationModel_H

#include "singleStepCombustion.H"

#include "LESModel.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace combustionModels
{

/*---------------------------------------------------------------------------*\
                  Class eddyDissipationModel Declaration
\*---------------------------------------------------------------------------*/

template<class ReactionThermo>
class eddyDissipationModel
:
    public singleStepCombustion<ReactionThermo>
{
    // Private data

        //- Model constant
        scalar C_;

        scalar Cd_;

        scalar Cstiff_;

        //- Product volume - used by alphat BC

        volScalarField PV_;

    // Private Member Functions

        //- Return the reciprocal of the turbulent mixing time scale
        tmp<volScalarField> rtTurb() const;

        //- Return the reciprocal of the diffusion time scale
        tmp<volScalarField> rtDiff() const;

        //- Calculate the product volume
        void calcPV();

        //- No copy construct
        eddyDissipationModel(const eddyDissipationModel&) = delete;

        //- No copy assignment
        void operator=(const eddyDissipationModel&) = delete;


public:

    //- Runtime type information
    TypeName("eddyDissipationModel");


    // Constructors

        //- Construct from components
        eddyDissipationModel
        (
            const word& modelType,
            ReactionThermo& thermo,
            const compressibleTurbulenceModel& turb,
            const word& combustionProperties
        );


    //- Destructor
    virtual ~eddyDissipationModel();


    // Member Functions

        //- Correct combustion rate
        virtual void correct();

        //- Update properties
        virtual bool read();

        label getParameter() const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace combustionModels
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
    #include "eddyDissipationModel.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
