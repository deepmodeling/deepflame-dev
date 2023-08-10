/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2020 OpenCFD Ltd.
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
    Foam::LiquidEvapFuchsKnudsen

Group
    grpLagrangianIntermediatePhaseChangeSubModels

Description
    Liquid evaporation/condensation model for solution of liquid and solid.

    This model takes into account the Fuchs-Knudsen number correction, the
    modified Raoult's law is used to obtain the concenration of the evapora
    ble component on the surface and the activity coefficient is used.
    The correction Kelvin effect is used.

    Reference:
    \verbatim
        Xiaole Chen, Yu Feng, Wenqi Zhon, Clement Kleinstreuer.
        Numerical investigation of the interaction, transport and deposition
        of multicomponent droplets in a a simple mouth-throat model.
        Journal of Aerosol Science, 105(2017), 108-127.
        DOI:10.1016/j.jaerosci.2016.12.001
    \endverbatim


\*---------------------------------------------------------------------------*/

#ifndef LiquidEvapFuchsKnudsen_H
#define LiquidEvapFuchsKnudsen_H

#include "PhaseChangeModel.H"
#include "liquidMixtureProperties.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
/*---------------------------------------------------------------------------*\
                     Class LiquidEvapFuchsKnudsen Declaration
\*---------------------------------------------------------------------------*/

template<class CloudType>
class LiquidEvapFuchsKnudsen
:
    public PhaseChangeModel<CloudType>
{
public:

    // Public Enumerations

        //- Type of activity coefficient models
        enum activityCoeffMethodType
        {
            pUNIFAC,
            pHoff
        };


protected:

    // Protected Data

        //- Method used
        activityCoeffMethodType method_;

        //- Mean gas free path
        scalar gamma_;

        //- The mass thermal accomodation
        scalar alpha_;

        //- Global liquid properties data
        const liquidMixtureProperties& liquids_;

        //- List of active liquid names i.e (liquidName solidName)
        List<word> solution_;

        //- Mapping between liquid and carrier species
        label liqToCarrierMap_;

        //- Mapping between local and global liquid species
        label liqToLiqMap_;

        //- Mapping between local and global solid species
        label solToSolMap_;


    // Protected Member Functions

        //- Sherwood number as a function of Reynolds and Schmidt numbers
        scalar Sh(const scalar Re, const scalar Sc) const;

        //- Calculate the carrier phase component volume fractions at celli
        tmp<scalarField> calcXc(const label celli) const;

        //- Calculate volumetric fractions of components in the solution
        void calcXcSolution
        (
            const scalar massliq,
            const scalar masssol,
            scalar& Xliq,
            scalar& Xsol
        ) const;

        //- Return activity coefficient
        scalar activityCoeff(const scalar Xliq, const scalar Ysol) const;


public:

    //- Runtime type information
    TypeName("liquidEvapFuchsKnudsen");


    // Constructors

        //- Construct from dictionary
        LiquidEvapFuchsKnudsen(const dictionary& dict, CloudType& cloud);

        //- Construct copy
        LiquidEvapFuchsKnudsen(const LiquidEvapFuchsKnudsen<CloudType>& pcm);

        //- Construct and return a clone
        virtual autoPtr<PhaseChangeModel<CloudType>> clone() const
        {
            return autoPtr<PhaseChangeModel<CloudType>>
            (
                new LiquidEvapFuchsKnudsen<CloudType>(*this)
            );
        }


    //- Destructor
    virtual ~LiquidEvapFuchsKnudsen() = default;


    // Member Functions

        //- Update model
        virtual void calculate
        (
            const scalar dt,
            const label celli,
            const scalar Re,
            const scalar Pr,
            const scalar d,
            const scalar nu,
            const scalar rho,
            const scalar T,
            const scalar Ts,
            const scalar pc,
            const scalar Tc,
            const scalarField& X,
            const scalarField& Xsol,
            const scalarField& liqMass,
            scalarField& dMassPC
        ) const;

        //- Return the enthalpy per unit mass
        virtual scalar dh
        (
            const label idc,
            const label idl,
            const scalar p,
            const scalar T
        ) const;

        //- Return vapourisation temperature
        virtual scalar Tvap(const scalarField& X) const;

        //- Return maximum/limiting temperature
        virtual scalar TMax(const scalar p, const scalarField& X) const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
    #include "LiquidEvapFuchsKnudsen.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
