/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2019 OpenFOAM Foundation
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
    Foam::LESModels::Sigma

Description
    Sigma is an eddy-viscosity SGS model, not only adapting to wall conditions
    but also to laminar flows where nut should also vanish.

    Reference:
    \verbatim
        F. Nicoud, H.B. Toda, O. Cabrit, S. Bose and J. Lee (2011).
	Using singular values to build a subgrid-scale model
	for large eddy simulations
    	Physics of Fluids, 23 (8), p. 085106.
    \endverbatim

    The default model coefficients are
    \verbatim
        SigmaCoeffs
        {
            filter   simple;
            Csg      1.5;
        }
    \endverbatim    

SourceFiles
    Sigma.C

\*---------------------------------------------------------------------------*/

#ifndef Sigma_H
#define Sigma_H
#include "SVD.H"

#include "LESeddyViscosity.H"
#include "simpleFilter.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace LESModels
{

/*---------------------------------------------------------------------------*\
                       Class Sigma Declaration
\*---------------------------------------------------------------------------*/

template<class BasicTurbulenceModel>
class Sigma
:
    public LESeddyViscosity<BasicTurbulenceModel>
{
    // Private Member Functions

        // Disallow default bitwise copy construct and assignment
        Sigma(const Sigma&);
        void operator=(const Sigma&);


protected:

    // Protected data

       dimensionedScalar Csg_;
       volScalarField k_;

       simpleFilter simpleFilter_;
       autoPtr<LESfilter> filterPtr_;
       LESfilter& filter_;

    // Protected Member Functions

        virtual void correctNut();


public:

    typedef typename BasicTurbulenceModel::alphaField alphaField;
    typedef typename BasicTurbulenceModel::rhoField rhoField;
    typedef typename BasicTurbulenceModel::transportModel transportModel;

    //- Runtime type information
    TypeName("Sigma");


    // Constructors

        //- Construct from components
        Sigma
        (
            const alphaField& alpha,
            const rhoField& rho,
            const volVectorField& U,
            const surfaceScalarField& alphaRhoPhi,
            const surfaceScalarField& phi,
            const transportModel& transport,
            const word& propertiesName = turbulenceModel::propertiesName,
            const word& type = typeName
        );


    //- Destructor
    virtual ~Sigma()
    {}


    // Member Functions

        //- Read model coefficients if they have changed
        virtual bool read();

        //- Return SGS kinetic energy
        virtual tmp<volScalarField> k() const
        {
            return k_;
        }

        //- Return the effective diffusivity for k
        tmp<volScalarField> DkEff() const
        {
            return tmp<volScalarField>
            (
                new volScalarField("DkEff", this->nut_ + this->nu())
            );
        }

        //- Correct Eddy-Viscosity and related properties
        virtual void correct();
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace LESModels
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
    #include "Sigma.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
