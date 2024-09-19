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
    Foam::fluxScheme

Description
    Base class for flux schemes to interpolate fields and loop over faces
    and boundaries

SourceFiles
    fluxScheme.C
    newFluxScheme.C

Authors
    Original code:
    Jie Sun
    Adapted by:
    Minqi Lin 2024

\*---------------------------------------------------------------------------*/

#ifndef fluxScheme_H
#define fluxScheme_H

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#include "volFields.H"
#include "surfaceFields.H"
#include "regIOobject.H"
#include "dictionary.H"
#include "runTimeSelectionTables.H"
#include "fvc.H"

namespace Foam
{

/*---------------------------------------------------------------------------*\
                           Class fluxScheme Declaration
\*---------------------------------------------------------------------------*/

class fluxScheme
:
    public regIOobject
{
protected:

    //- Const reference to mesh
    const fvMesh& mesh_;

    //- Owner and neighbour surface fields
    tmp<surfaceScalarField> own_;
    tmp<surfaceScalarField> nei_;

    //- Saved interpolated rho fields
    tmp<surfaceScalarField> rhoOwn_;
    tmp<surfaceScalarField> rhoNei_;

    // Protected Functions

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
        ) = 0;

        //- Update fields before calculating fluxes
        virtual void preUpdate(const volScalarField& p)
        {}

        //- Correct fluxes
        virtual void postUpdate()
        {}

        //- Return velocity w.r.t. face
        scalar meshPhi(const label facei, const label patchi) const
        {
            if (!mesh_.moving())
            {
                return 0.0;
            }
            if (patchi != -1)
            {
                return mesh_.phi().boundaryField()[patchi][facei];
            }
            else
            {
                return mesh_.phi()[facei];
            }
        }

        inline word scheme(const word& name) const
        {
            word schemeName = "reconstruct(" + name + ")";
            if
            (
                !mesh_.schemesDict().subDict("interpolationSchemes").found(schemeName)
            )
            {
                WarningInFunction
                    << "Riemann schemes are used, but no limiter is " << nl
                    << "specified for " << name << "." << nl
                    << "This may result in unstable solutions." << endl;
            }
            return schemeName;
        }


public:

    //- Runtime type information
    TypeName("fluxScheme");

    // Declare runtime construction

        declareRunTimeSelectionTable
        (
            autoPtr,
            fluxScheme,
            dictionary,
            (const fvMesh& mesh),
            (mesh)
        );

    // Constructor
    fluxScheme(const fvMesh& mesh);


    //- Destructor
    virtual ~fluxScheme();


    // Selectors

        static autoPtr<fluxScheme> New(const fvMesh& e);


    // Member Functions

        //- Clear savedFields
        virtual void clear();

        //- Allocate saved fields
        virtual void createSavedFields();

        //- Update
        void update
        (
            const volScalarField& rho,
            const PtrList<volScalarField>& rhoYi,
            const scalar& nspecies,
            const volVectorField& U,
            const volScalarField& e,
            const volScalarField& p,
            const volScalarField& c,
            surfaceScalarField& phi,
            surfaceScalarField& rhoPhi,
            PtrList<surfaceScalarField>& rhoPhiYi,
            surfaceVectorField& rhoUPhi,
            surfaceScalarField& rhoEPhi
        );

        //- Return interpolated U field
        tmp<surfaceVectorField> Uf() const;

        // - Dummy write for regIOobject
        bool writeData(Ostream& os) const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //