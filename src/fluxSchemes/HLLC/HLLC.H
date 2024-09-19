/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2019 Synthetik Applied Technologies
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is derivative work of OpenFOAM.

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
    Foam::fluxSchemes::HLLC

Description

    References:
    \verbatim
        Toro, E.F., Spruce, M., Speares, W.  (1993).
        Restoration of the contact surface in the HLL-Riemann solver.
        Shock Waves, 4, 25-34.
    \endverbatim

SourceFiles
    HLLC.C

Authors
    Original code:
    Jie Sun
    Adapted by:
    Minqi Lin 2024

\*---------------------------------------------------------------------------*/

#ifndef HLLC_H
#define HLLC_H

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#include "fluxScheme.H"

namespace Foam
{
namespace fluxSchemes
{

/*---------------------------------------------------------------------------*\
                           Class HLLC Declaration
\*---------------------------------------------------------------------------*/

class HLLC
:
    public fluxScheme
{

    // Saved variables

        // tmp<surfaceScalarField> SOwn_;
        // tmp<surfaceScalarField> SNei_;
        // tmp<surfaceScalarField> SStar_;
        // tmp<surfaceScalarField> pStarOwn_;
        // tmp<surfaceScalarField> pStarNei_;
        // tmp<surfaceScalarField> UvOwn_;
        // tmp<surfaceScalarField> UvNei_;

    // Private functions

        //- Calcualte fluxes
        virtual void calculateFluxes
        (
            const scalar& rhoOwn, const scalar& rhoNei,
            const scalarList& rhoYiOwn,
            const scalarList& rhoYiNei,
            const vector& UOwn, const vector& UNei,
            const scalar& eOwn, const scalar& eNei,
            const scalar& pOwn, const scalar& pNei,
            const scalar& cOwn, const scalar& cNei,
            const vector& Sf,
            scalar& phi,
            scalar& rhoPhi,
            scalarList& rhoPhiYi,
            vector& rhoUPhi,
            scalar& rhoEPhi,
            const label facei, const label patchi = -1
        );


public:

    //- Runtime type information
    TypeName("HLLC");

    // Constructor
    HLLC(const fvMesh& mesh);


    //- Destructor
    virtual ~HLLC();


    // Member Functions

        //- Clear savedFields
        virtual void clear();

        //- Allocate saved fields
        virtual void createSavedFields();
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace fluxSchemes
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //