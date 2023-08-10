/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2020-2021 OpenCFD Ltd.
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

#include "LiquidEvapFuchsKnudsen.H"


// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

template<class CloudType>
void Foam::LiquidEvapFuchsKnudsen<CloudType>::calcXcSolution
(
    const scalar massliq,
    const scalar masssol,
    scalar& Xliq,
    scalar& Xsol
) const
{
    const scalar Yliq = massliq/(massliq + masssol);
    const scalar Ysol = 1 - Yliq;
    Xliq = Yliq/liquids_.properties()[liqToLiqMap_].W();
    Xsol = Ysol/this->owner().thermo().solids().properties()[solToSolMap_].W();
    Xliq /= (Xliq + Xsol);
    Xsol = 1 - Xliq;
}


template<class CloudType>
Foam::tmp<Foam::scalarField> Foam::LiquidEvapFuchsKnudsen<CloudType>::calcXc
(
    const label celli
) const
{
    scalarField Xc(this->owner().thermo().carrier().Y().size());

    forAll(Xc, i)
    {
        Xc[i] =
            this->owner().thermo().carrier().Y()[i][celli]
           /this->owner().thermo().carrier().W(i);
    }

    return Xc/sum(Xc);
}


template<class CloudType>
Foam::scalar Foam::LiquidEvapFuchsKnudsen<CloudType>::Sh
(
    const scalar Re,
    const scalar Sc
) const
{
    return cbrt(1 + Sc*Re)*max(1, pow(Re, 0.077));
}


template<class CloudType>
Foam::scalar Foam::LiquidEvapFuchsKnudsen<CloudType>::activityCoeff
(
    const scalar Xliq,
    const scalar Xsol
) const
{
    switch (method_)
    {
        case pUNIFAC:
        {
            FatalErrorInFunction
                << "Activity coefficient UNIFAC is not implemented " << nl
                << abort(FatalError);
            break;
        }
        case pHoff:
        {
            const scalar ic = this->coeffDict().getScalar("ic");
            return inv((1 + ic*Xsol/(Xliq + ROOTVSMALL)));
            break;
        }
    }
    return -1;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class CloudType>
Foam::LiquidEvapFuchsKnudsen<CloudType>::LiquidEvapFuchsKnudsen
(
    const dictionary& dict,
    CloudType& owner
)
:
    PhaseChangeModel<CloudType>(dict, owner, typeName),
    method_(pHoff),
    gamma_(this->coeffDict().getScalar("gamma")),
    alpha_(this->coeffDict().getScalar("alpham")),
    liquids_(owner.thermo().liquids()),
    solution_(this->coeffDict().lookup("solution")),
    liqToCarrierMap_(-1),
    liqToLiqMap_(-1),
    solToSolMap_(-1)
{
    if (solution_.size() > 2)
    {
        FatalErrorInFunction
            << "Solution is not well defined. It should be (liquid solid)"
            << nl <<  exit(FatalError);
    }
    else
    {
        Info<< "Participating liquid-solid species:" << endl;

        Info<< "    " << solution_[0] << endl;
        liqToCarrierMap_ =
            owner.composition().carrierId(solution_[0]);

        // Determine mapping between model active liquids and global liquids
        const label idLiquid = owner.composition().idLiquid();
        liqToLiqMap_ =
            owner.composition().localId(idLiquid, solution_[0]);

        // Mapping for the solid
        const label idSolid = owner.composition().idSolid();

        solToSolMap_ =
            owner.composition().localId(idSolid, solution_[1]);

        const word activityCoefficienType
        (
            this->coeffDict().getWord("activityCoefficient")
        );

        if (activityCoefficienType == "Hoff")
        {
            method_ = pHoff;
        }
        else if (activityCoefficienType == "UNIFAC")
        {
            method_ = pUNIFAC;
        }
        else
        {
            FatalErrorInFunction
                << "activityCoefficient must be either 'Hoff' or 'UNIFAC'"
                << nl << exit(FatalError);
        }
    }
}


template<class CloudType>
Foam::LiquidEvapFuchsKnudsen<CloudType>::LiquidEvapFuchsKnudsen
(
    const LiquidEvapFuchsKnudsen<CloudType>& pcm
)
:
    PhaseChangeModel<CloudType>(pcm),
    method_(pcm.method_),
    gamma_(pcm.gamma_),
    alpha_(pcm.alpha_),
    liquids_(pcm.owner().thermo().liquids()),
    solution_(pcm.solution_),
    liqToCarrierMap_(pcm.liqToCarrierMap_),
    liqToLiqMap_(pcm.liqToLiqMap_),
    solToSolMap_(pcm.solToSolMap_)
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class CloudType>
void Foam::LiquidEvapFuchsKnudsen<CloudType>::calculate
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
    const scalarField& solMass,
    const scalarField& liqMass,
    scalarField& dMassPC
) const
{
    const scalar rhog = this->owner().thermo().thermo().rho()()[celli];

    const label gid = liqToCarrierMap_;
    const label lid = liqToLiqMap_;
    const label sid = solToSolMap_;

    const scalar W = liquids_.properties()[lid].W();

    const scalar YeInf = this->owner().thermo().carrier().Y()[gid][celli];

    const scalar sigma = liquids_.properties()[lid].sigma(pc, Ts);

    // Kelvin effect
    const scalar RR = 1000.0*constant::physicoChemical::R.value(); // J/(kmol·k)
    const scalar Ke = exp(4*sigma*W/(RR*rho*d*T));

    // vapour diffusivity [m2/s]
    const scalar Dab = liquids_.properties()[lid].D(pc, Ts);

    // saturation pressure for species i [pa]
    const scalar pSat = liquids_.properties()[lid].pv(pc, T);

    scalar Xliq(0), Xsol(0);
    calcXcSolution(liqMass[lid], solMass[sid], Xliq, Xsol);

    // Activity Coefficient (gammaE*Xe)
    const scalar gamma = activityCoeff(Xliq, Xsol);

    // water concentration at surface
    const scalar Rliq = RR/W;
    const scalar YeSurf = max(gamma*Ke*pSat/(Rliq*T*rhog), 0);

    const scalar Kn = 2*gamma_/d;
    const scalar Cm =
        (1+Kn)/(1+ (4/(3*alpha_) + 0.377)*Kn + sqr(Kn)*4/(3*alpha_));

    // Schmidt number
    const scalar Sc = nu/(Dab + ROOTVSMALL);

    // Sherwood number
    const scalar Sherwood = Sh(Re, Sc);

    // mass flux density [kg/m2/s]
    const scalar Ni = (rhog*Sherwood*Dab*Cm/d)*log((1 - YeInf)/(1 - YeSurf));

    // mass transfer [kg]
    const scalar As = Foam::constant::mathematical::pi*d*d;
    dMassPC[lid] += Ni*As*dt;
}


template<class CloudType>
Foam::scalar Foam::LiquidEvapFuchsKnudsen<CloudType>::dh
(
    const label idc,
    const label idl,
    const scalar p,
    const scalar T
) const
{
    scalar dh = 0;

    typedef PhaseChangeModel<CloudType> parent;
    switch (parent::enthalpyTransfer_)
    {
        case (parent::etLatentHeat):
        {
            dh = liquids_.properties()[idl].hl(p, T);
            break;
        }
        case (parent::etEnthalpyDifference):
        {
            scalar hc = this->owner().composition().carrier().Ha(idc, p, T);
            scalar hp = liquids_.properties()[idl].h(p, T);

            dh = hc - hp;
            break;
        }
        default:
        {
            FatalErrorInFunction
                << "Unknown enthalpyTransfer type" << abort(FatalError);
        }
    }

    return dh;
}


template<class CloudType>
Foam::scalar Foam::LiquidEvapFuchsKnudsen<CloudType>::Tvap
(
    const scalarField& X
) const
{
    return Zero;
}


template<class CloudType>
Foam::scalar Foam::LiquidEvapFuchsKnudsen<CloudType>::TMax
(
    const scalar p,
    const scalarField& X
) const
{
    // If liquid is present calculates pvInter
    if (sum(X) > SMALL)
    {
        return liquids_.pvInvert(p, X);
    }
    else
    {
        return GREAT;
    }
}


// ************************************************************************* //
