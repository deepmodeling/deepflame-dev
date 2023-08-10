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

#include "ReactingHeterogeneousCloud.H"
#include "HeterogeneousReactingModel.H"

// * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * * //

template<class CloudType>
void Foam::ReactingHeterogeneousCloud<CloudType>::setModels()
{
    heterogeneousReactionModel_.reset
    (
        HeterogeneousReactingModel<ReactingHeterogeneousCloud<CloudType>>::New
        (
            this->subModelProperties(),
            *this
        ).ptr()
    );
}


template<class CloudType>
void Foam::ReactingHeterogeneousCloud<CloudType>::cloudReset
(
    ReactingHeterogeneousCloud<CloudType>& c
)
{
    CloudType::cloudReset(c);
    heterogeneousReactionModel_.reset(c.heterogeneousReactionModel_.ptr());
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class CloudType>
Foam::ReactingHeterogeneousCloud<CloudType>::ReactingHeterogeneousCloud
(
    const word& cloudName,
    const volScalarField& rho,
    const volVectorField& U,
    const dimensionedVector& g,
    const SLGThermo& thermo,
    bool readFields
)
:
    CloudType(cloudName, rho, U, g, thermo, false),
    reactingHeterogeneousCloud(),
    cloudCopyPtr_(nullptr),
    heterogeneousReactionModel_(nullptr)
{
    if (this->solution().active())
    {
        setModels();

        if (readFields)
        {
            parcelType::readFields(*this, this->composition());
            this->deleteLostParticles();
        }
    }
}


template<class CloudType>
Foam::ReactingHeterogeneousCloud<CloudType>::ReactingHeterogeneousCloud
(
    ReactingHeterogeneousCloud<CloudType>& c,
    const word& name
)
:
    CloudType(c, name),
    reactingHeterogeneousCloud(),
    cloudCopyPtr_(nullptr),
    heterogeneousReactionModel_(c.heterogeneousReactionModel_->clone())
{}


template<class CloudType>
Foam::ReactingHeterogeneousCloud<CloudType>::ReactingHeterogeneousCloud
(
    const fvMesh& mesh,
    const word& name,
    const ReactingHeterogeneousCloud<CloudType>& c
)
:
    CloudType(mesh, name, c),
    reactingHeterogeneousCloud(),
    cloudCopyPtr_(nullptr),
    heterogeneousReactionModel_(c.heterogeneousReactionModel_->clone())
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class CloudType>
void Foam::ReactingHeterogeneousCloud<CloudType>::setParcelThermoProperties
(
    parcelType& parcel,
    const scalar lagrangianDt
)
{
    CloudType::setParcelThermoProperties(parcel, lagrangianDt);
    label idS = this->composition().idSolid();

    // Set initial particle composition. Overwrite thermoProperties from
    // ReactingCloud
    parcel.Y() = this->composition().Y0(idS);

    // Set initial progress to 0
    parcel.F().setSize(heterogeneousReactionModel_->nF(), 0.0);

    // Set the parcel to combust
    parcel.canCombust() = 1;

    // If rho0 was given in constProp use it. If not use the composition
    // to set tho
    if (this->constProps_.rho0() == -1)
    {
        const label idGas = this->composition().idGas();
        const label idLiquid = this->composition().idLiquid();
        const label idSolid = this->composition().idSolid();

        const scalarField& Ygas = this->composition().Y0(idGas);
        const scalarField& Yliq = this->composition().Y0(idLiquid);
        const scalarField& Ysol = this->composition().Y0(idSolid);

        const scalar p0 =
            this->composition().thermo().thermo().p()[parcel.cell()];
        const scalar T0 = this->constProps_.T0();

        parcel.rho() = this->composition().rho(Ygas, Yliq, Ysol, T0, p0);
    }
}


template<class CloudType>
void Foam::ReactingHeterogeneousCloud<CloudType>::checkParcelProperties
(
    parcelType& parcel,
    const scalar lagrangianDt,
    const bool fullyDescribed
)
{
    CloudType::checkParcelProperties(parcel, lagrangianDt, false);

    const label solId = this->composition().idSolid();
    const label liqId = this->composition().idLiquid();
    const label gasId = this->composition().idGas();

    // Check YMixture is pure solid
    if
    (
        this->composition().YMixture0()[solId] != 1.0
     || this->composition().YMixture0()[liqId] != 0.0
     || this->composition().YMixture0()[gasId] != 0.0
    )
    {
        FatalErrorInFunction
            << "The supplied composition must be : " << nl
            << "    YGasTot0        0 : " << nl
            << "    YLiquidTot0     0 : " << nl
            << "    YSolidTot0      1 : " << nl
            << "This Cloud only works with pure solid particles."
            << abort(FatalError);
    }
    if (this->composition().liquids().size() > 0)
    {
        FatalErrorInFunction
            << "The supplied composition has a liquid phase. " << nl
            << "This Cloud only works with pure solid particles."
            << abort(FatalError);
    }
}


template<class CloudType>
void Foam::ReactingHeterogeneousCloud<CloudType>::storeState()
{
    cloudCopyPtr_.reset
    (
        static_cast<ReactingHeterogeneousCloud<CloudType>*>
        (
            clone(this->name() + "Copy").ptr()
        )
    );
}


template<class CloudType>
void Foam::ReactingHeterogeneousCloud<CloudType>::restoreState()
{
    cloudReset(cloudCopyPtr_());
    cloudCopyPtr_.clear();
}


template<class CloudType>
void Foam::ReactingHeterogeneousCloud<CloudType>::evolve()
{
    if (this->solution().canEvolve())
    {
        typename parcelType::trackingData td(*this);

        this->solve(*this, td);
    }
}


template<class CloudType>
void Foam::ReactingHeterogeneousCloud<CloudType>::autoMap
(
    const mapPolyMesh& mapper
)
{
    Cloud<parcelType>::autoMap(mapper);

    this->updateMesh();
}


template<class CloudType>
void Foam::ReactingHeterogeneousCloud<CloudType>::info()
{
    CloudType::info();

    heterogeneousReactionModel_->info();
}


template<class CloudType>
void Foam::ReactingHeterogeneousCloud<CloudType>::writeFields() const
{
    CloudType::writeFields();
}


template<class CloudType>
void Foam::ReactingHeterogeneousCloud<CloudType>::
readObjects(const objectRegistry& obr)
{
    CloudType::particleType::readObjects(*this, this->composition(), obr);
}


template<class CloudType>
void Foam::ReactingHeterogeneousCloud<CloudType>::
writeObjects(objectRegistry& obr) const
{
    CloudType::particleType::writeObjects(*this, this->composition(), obr);
}


// ************************************************************************* //
