/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
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

\*---------------------------------------------------------------------------*/

#include "FGMLiquidEvaporationBoil.H"
// #include "specie.H"
#include "mathematicalConstants.H"


using namespace Foam::constant::mathematical;

// // * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

// template<class CloudType>
// Foam::tmp<Foam::scalarField> Foam::LiquidEvaporationBoil<CloudType>::calcXc
// (
//     const label celli
// ) const
// {
//     scalarField Xc(this->owner().thermo().carrier().Y().size());

//     forAll(Xc, i)
//     {
//         Xc[i] =
//             this->owner().thermo().carrier().Y()[i][celli]
//            /this->owner().thermo().carrier().Wi(i);
//     }

//     return Xc/sum(Xc);
// }


// template<class CloudType>
// Foam::scalar Foam::LiquidEvaporationBoil<CloudType>::Sh
// (
//     const scalar Re,
//     const scalar Sc
// ) const
// {
//     return 2.0 + 0.6*Foam::sqrt(Re)*cbrt(Sc);
// }


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class CloudType>
Foam::FGMLiquidEvaporationBoil<CloudType>::FGMLiquidEvaporationBoil
(
    const dictionary& dict,
    CloudType& owner
)
:
    LiquidEvaporationBoil<CloudType>(dict, owner),
    W_(this->owner().mesh().objectRegistry::lookupObject<const volScalarField>("Wt"))
{

}


template<class CloudType>
Foam::FGMLiquidEvaporationBoil<CloudType>::FGMLiquidEvaporationBoil
(
    const FGMLiquidEvaporationBoil<CloudType>& pcm
)
:
    LiquidEvaporationBoil<CloudType>(pcm),

    W_(this->owner().mesh().objectRegistry::lookupObject<const volScalarField>("Wt"))
{

}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

template<class CloudType>
Foam::FGMLiquidEvaporationBoil<CloudType>::~FGMLiquidEvaporationBoil()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class CloudType>
void Foam::FGMLiquidEvaporationBoil<CloudType>::calculate
(
    const scalar dt,
    const label celli,
    const scalar Re,
    const scalar Pr,
    const scalar d,
    const scalar nu,
    const scalar T,
    const scalar Ts,
    const scalar pc,
    const scalar Tc,
    const scalarField& X,
    scalarField& dMassPC
) const
{
    // immediately evaporate mass that has reached critical condition
    if ((this->liquids_.Tc(X) - T) < small)
    {
        if (debug)
        {
            WarningInFunction
                << "Parcel reached critical conditions: "
                << "evaporating all available mass" << endl;
        }

        forAll(this->activeLiquids_, i)
        {
            const label lid = this->liqToLiqMap_[i];
            dMassPC[lid] = great;
        }

        return;
    }

    // droplet surface pressure assumed to surface vapour pressure
    scalar ps = this->liquids_.pv(pc, Ts, X);

    // vapour density at droplet surface [kg/m^3]
    const scalar RR = 1000.0*constant::physicoChemical::R.value(); // J/(kmol·k)
    // vapour density at droplet surface [kg/m3]
    scalar rhos = ps*this->liquids_.W(X)/(RR*Ts);

    // // construct carrier phase species volume fractions for cell, celli
    // const scalarField XcMix(calcXc(celli));

    // // carrier thermo properties
    // scalar Hsc = 0.0;
    // scalar Hc = 0.0;
    // scalar Cpc = 0.0;
    // scalar kappac = 0.0;
    // this->owner().thermo().carrier().calcH(Tc, pc);
    // forAll(this->owner().thermo().carrier().Y(), i)
    // {
    //     const scalar Yc = this->owner().thermo().carrier().Y()[i][celli];
    //     Hc += Yc*this->owner().thermo().carrier().Ha(i, pc, Tc);
    // }

    // this->owner().thermo().carrier().calcH(Ts, ps);
    // this->owner().thermo().carrier().calcCp(Ts, ps);
    // this->owner().thermo().carrier().calcMu(Ts, ps);
    // forAll(this->owner().thermo().carrier().Y(), i)
    // {
    //     const scalar Yc = this->owner().thermo().carrier().Y()[i][celli];
    //     Hsc += Yc*this->owner().thermo().carrier().Ha(i, ps, Ts);
    //     Cpc += Yc*this->owner().thermo().carrier().Cp(i, ps, Ts);
    //     kappac += Yc*this->owner().thermo().carrier().kappa(i, ps, Ts);
    // }

    // calculate mass transfer of each specie in liquid
    forAll(this->activeLiquids_, i)
    {
        const label gid = this->liqToCarrierMap_[i];
        const label lid = this->liqToLiqMap_[i];

        // boiling temperature at cell pressure for liquid species lid [K]
        const scalar TBoil = this->liquids_.properties()[lid].pvInvert(pc);
        // limit droplet temperature to boiling/critical temperature
        const scalar Td = min(T, 0.999*TBoil);
        // saturation pressure for liquid species lid [Pa]
        const scalar pSat = this->liquids_.properties()[lid].pv(pc, Td);
        // surface molar fraction - Raoult's Law
        const scalar Xs = X[lid]*pSat/pc;


        // if (Xc*pc > pSat)
        // {
        //     // saturated vapour - no phase change
        // }
        // else
        {
            // vapour diffusivity [m2/s]
            const scalar Dab = this->liquids_.properties()[lid].D(ps, Ts);

            // Schmidt number
            const scalar Sc = nu/(Dab + rootVSmall);

            // Sherwood number
            const scalar Sh = this->Sh(Re, Sc);

            // mixture molar weight
            scalar W_gas = W_[celli];//FGMmodel_.getValue(celli, "W");

            // carrier phase concentration-fuel
            const scalar Yc = this->owner().thermo().carrier().Y()[gid][celli];

            // fuel molar weight [kg/kmol]
            const scalar W_fuel = this->liquids_.properties()[lid].W();
            const scalar Ys = Xs*W_fuel/W_gas/(1. + Xs*W_fuel/W_gas - Xs); // ref Hu.2017, Spalding.1953
            const scalar Bm = (Ys - Yc)/max(small, 1. - Ys);

            if (Bm > 0)
            {
                // mass transfer [kg]
                dMassPC[lid] += pi*d*Sh*Dab*rhos*log(1.0 + Bm)*dt;
            }
        }
    }
}




// ************************************************************************* //
