/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2018-2020 OpenCFD Ltd.
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

#include "ReactingHeterogeneousParcel.H"
#include "mathematicalConstants.H"

using namespace Foam::constant::mathematical;


// * * * * * * * * * * * *  Private Member Functions * * * * * * * * * * * * //

template<class ParcelType>
template<class TrackCloudType>
Foam::scalar Foam::ReactingHeterogeneousParcel<ParcelType>::CpEff
(
    TrackCloudType& cloud,
    trackingData& td,
    const scalar p,
    const scalar T,
    const label idS
) const
{
    return cloud.composition().Cp(idS, this->Y_, p, T);
}


template<class ParcelType>
template<class TrackCloudType>
Foam::scalar Foam::ReactingHeterogeneousParcel<ParcelType>::HsEff
(
    TrackCloudType& cloud,
    trackingData& td,
    const scalar p,
    const scalar T,
    const label idS
) const
{
    return cloud.composition().Hs(idS, this->Y_, p, T);
}


template<class ParcelType>
template<class TrackCloudType>
Foam::scalar Foam::ReactingHeterogeneousParcel<ParcelType>::LEff
(
    TrackCloudType& cloud,
    trackingData& td,
    const scalar p,
    const scalar T,
    const label idS
) const
{
    return cloud.composition().L(idS, this->Y_, p, T);
}


// * * * * * * * * * * *  Protected Member Functions * * * * * * * * * * * * //


template<class ParcelType>
template<class TrackCloudType>
Foam::scalar Foam::ReactingHeterogeneousParcel<ParcelType>::updatedDeltaVolume
(
    TrackCloudType& cloud,
    const scalarField& dMass,
    const scalar p,
    const scalar T
)
{
    const auto& composition = cloud.composition();

    scalarField dVolSolid(dMass.size(), Zero);
    forAll(dVolSolid, i)
    {
        dVolSolid[i] =
            dMass[i]/composition.solids().properties()[i].rho();
    }

    return sum(dVolSolid);
}


template<class ParcelType>
template<class TrackCloudType>
void Foam::ReactingHeterogeneousParcel<ParcelType>::calc
(
    TrackCloudType& cloud,
    trackingData& td,
    const scalar dt
)
{
    typedef typename TrackCloudType::reactingCloudType reactingCloudType;

    const CompositionModel<reactingCloudType>& composition =
        cloud.composition();

    // Define local properties at beginning of timestep
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    const scalar np0 = this->nParticle_;
    const scalar d0 = this->d_;
    const vector& U0 = this->U_;
    const scalar T0 = this->T_;
    const scalar mass0 = this->mass();

    const scalar pc = td.pc();

    const label idS = composition.idSolid();

    // Calc surface values
    scalar Ts, rhos, mus, Prs, kappas;
    this->calcSurfaceValues(cloud, td, T0, Ts, rhos, mus, Prs, kappas);
    scalar Res = this->Re(rhos, U0, td.Uc(), d0, mus);

    // Sources
    //~~~~~~~~

    // Explicit momentum source for particle
    vector Su = Zero;

    // Linearised momentum source coefficient
    scalar Spu = 0.0;

    // Momentum transfer from the particle to the carrier phase
    vector dUTrans = Zero;

    // Explicit enthalpy source for particle
    scalar Sh = 0.0;

    // Linearised enthalpy source coefficient
    scalar Sph = 0.0;

    // Sensible enthalpy transfer from the particle to the carrier phase
    scalar dhsTrans = 0.0;

    // Molar flux of species emitted from the particle (kmol/m^2/s)
    scalar Ne = 0.0;

    // Surface concentrations of emitted species
    scalarField Cs(composition.carrier().species().size(), 0.0);

    // Sum Ni*Cpi*Wi of emission species
    scalar NCpW = 0.0;


    // Heterogeneous reactions
    // ~~~~~~~~~~~~~~~~~

    // Change in carrier phase composition due to surface reactions
    scalarField dMassSRSolid(this->Y_.size(), 0.0);
    scalarField dMassSRCarrier(composition.carrier().species().size(), 0.0);

    // Calc mass and enthalpy transfer due to reactions
    calcHeterogeneousReactions
    (
        cloud,
        td,
        dt,
        Res,
        mus/rhos,
        d0,
        T0,
        mass0,
        canCombust_,
        Ne,
        NCpW,
        this->Y_,
        F_,
        dMassSRSolid,
        dMassSRCarrier,
        Sh,
        dhsTrans
    );

    // 2. Update the parcel properties due to change in mass
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    scalarField dMassSolid(dMassSRSolid);

    scalar mass1 = mass0 - sum(dMassSolid);

    // Remove the particle when mass falls below minimum threshold
    if (np0*mass1 < cloud.constProps().minParcelMass())
    {
        td.keepParticle = false;

        return;
    }

    // Only solid is used to update mass fractions
    (void)this->updateMassFraction(mass0, dMassSolid, this->Y_);


    if
    (
        cloud.constProps().volUpdateType()
     == constantProperties::volumeUpdateType::mUndefined
    )
    {
        if (cloud.constProps().constantVolume())
        {
            this->rho_ = mass1/this->volume();
        }
        else
        {
            this->d_ = cbrt(mass1/this->rho_*6/pi);
        }
    }
    else
    {
        switch (cloud.constProps().volUpdateType())
        {
            case constantProperties::volumeUpdateType::mConstRho :
            {
                this->d_ = cbrt(mass1/this->rho_*6/pi);
                break;
            }
            case constantProperties::volumeUpdateType::mConstVol :
            {
                this->rho_ = mass1/this->volume();
                break;
            }
            case constantProperties::volumeUpdateType::mUpdateRhoAndVol :
            {
                scalar deltaVol =
                    updatedDeltaVolume
                    (
                        cloud,
                        dMassSolid,
                        pc,
                        T0
                    );

                this->rho_ = mass1/(this->volume() + deltaVol);
                this->d_ = cbrt(mass1/this->rho_*6.0/pi);
                break;
            }
        }
    }
    // Correct surface values due to emitted species
    this->correctSurfaceValues(cloud, td, Ts, Cs, rhos, mus, Prs, kappas);
    Res = this->Re(rhos, U0, td.Uc(), this->d_, mus);


    // 3. Compute heat- and momentum transfers
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // Heat transfer
    // ~~~~~~~~~~~~~

    // Calculate new particle temperature
    this->T_ =
        this->calcHeatTransfer
        (
            cloud,
            td,
            dt,
            Res,
            Prs,
            kappas,
            NCpW,
            Sh,
            dhsTrans,
            Sph
        );

    //DebugVar(np0);

    this->Cp_ = CpEff(cloud, td, pc, this->T_, idS);

    // Motion
    // ~~~~~~

    // Calculate new particle velocity
    this->U_ =
        this->calcVelocity(cloud, td, dt, Res, mus, mass1, Su, dUTrans, Spu);


    // 4. Accumulate carrier phase source terms
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if (cloud.solution().coupled())
    {
        // No mapping between solid components and carrier phase
        /*
        forAll(this->Y_, i)
        {
            scalar dm = np0*dMassSolid[i];
            label gid = composition.localToCarrierId(SLD, i);
            scalar hs = composition.carrier().Hs(gid, pc, T0);
            cloud.rhoTrans(gid)[this->cell()] += dm;
            cloud.UTrans()[this->cell()] += dm*U0;
            cloud.hsTrans()[this->cell()] += dm*hs;
        }
        */

        forAll(dMassSRCarrier, i)
        {
            scalar dm = np0*dMassSRCarrier[i];
            scalar hs = composition.carrier().Hs(i, pc, T0);
            cloud.rhoTrans(i)[this->cell()] += dm;
            cloud.UTrans()[this->cell()] += dm*U0;
            cloud.hsTrans()[this->cell()] += dm*hs;
        }

        // Update momentum transfer
        cloud.UTrans()[this->cell()] += np0*dUTrans;
        cloud.UCoeff()[this->cell()] += np0*Spu;

        // Update sensible enthalpy transfer
        cloud.hsTrans()[this->cell()] += np0*dhsTrans;
        cloud.hsCoeff()[this->cell()] += np0*Sph;

        // Update radiation fields
        if (cloud.radiation())
        {
            const scalar ap = this->areaP();
            const scalar T4 = pow4(T0);
            cloud.radAreaP()[this->cell()] += dt*np0*ap;
            cloud.radT4()[this->cell()] += dt*np0*T4;
            cloud.radAreaPT4()[this->cell()] += dt*np0*ap*T4;
        }
    }
}


// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

template<class ParcelType>
template<class TrackCloudType>
void Foam::ReactingHeterogeneousParcel<ParcelType>::calcHeterogeneousReactions
(
    TrackCloudType& cloud,
    trackingData& td,
    const scalar dt,
    const scalar Re,
    const scalar nu,
    const scalar d,
    const scalar T,
    const scalar mass,
    const label canCombust,
    const scalar Ne,
    scalar& NCpW,
    const scalarField& YSolid,
    scalarField& F,
    scalarField& dMassSRSolid,
    scalarField& dMassSRCarrier,
    scalar& Sh,
    scalar& dhsTrans
) const
{
    // Check that model is active
    if (!cloud.heterogeneousReaction().active())
    {
        return;
    }

    // Initialise demand-driven constants
    (void)cloud.constProps().hRetentionCoeff();
    (void)cloud.constProps().TMax();

    // Check that model is active
    if (canCombust != 1)
    {
        return;
    }

    // Update reactions
    const scalar hReaction = cloud.heterogeneousReaction().calculate
    (
        dt,
        Re,
        nu,
        this->cell(),
        d,
        T,
        td.Tc(),
        td.pc(),
        td.rhoc(),
        mass,
        YSolid,
        F,
        Ne,
        NCpW,
        dMassSRSolid,
        dMassSRCarrier
    );

    cloud.heterogeneousReaction().addToSurfaceReactionMass
    (
        this->nParticle_*sum(dMassSRSolid)
    );

    const scalar xsi = min(T/cloud.constProps().TMax(), 1.0);
    const scalar coeff =
        (1.0 - xsi*xsi)*cloud.constProps().hRetentionCoeff();

    Sh += coeff*hReaction/dt;

    dhsTrans += (1.0 - coeff)*hReaction;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class ParcelType>
Foam::ReactingHeterogeneousParcel<ParcelType>::ReactingHeterogeneousParcel
(
    const ReactingHeterogeneousParcel<ParcelType>& p
)
:
    ParcelType(p),
    F_(p.F_),
    canCombust_(p.canCombust_)
{}


template<class ParcelType>
Foam::ReactingHeterogeneousParcel<ParcelType>::ReactingHeterogeneousParcel
(
    const ReactingHeterogeneousParcel<ParcelType>& p,
    const polyMesh& mesh
)
:
    ParcelType(p, mesh),
    F_(p.F_),
    canCombust_(p.canCombust_)
{}


// * * * * * * * * * * * * * * IOStream operators  * * * * * * * * * * * * * //

#include "ReactingHeterogeneousParcelIO.C"

// ************************************************************************* //
