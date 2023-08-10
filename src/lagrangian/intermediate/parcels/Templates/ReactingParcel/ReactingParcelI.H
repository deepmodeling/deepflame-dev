/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
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

\*---------------------------------------------------------------------------*/

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class ParcelType>
inline
Foam::ReactingParcel<ParcelType>::constantProperties::constantProperties()
:
    ParcelType::constantProperties(),
    pMin_(this->dict_, 0.0),
    constantVolume_(this->dict_, false),
    volUpdateType_(this->dict_, mUndefined)
{}


template<class ParcelType>
inline Foam::ReactingParcel<ParcelType>::constantProperties::constantProperties
(
    const constantProperties& cp
)
:
    ParcelType::constantProperties(cp),
    pMin_(cp.pMin_),
    constantVolume_(cp.constantVolume_),
    volUpdateType_(cp.volUpdateType_)
{}


template<class ParcelType>
inline Foam::ReactingParcel<ParcelType>::constantProperties::constantProperties
(
    const dictionary& parentDict
)
:
    ParcelType::constantProperties(parentDict),
    pMin_(this->dict_, "pMin", 1000.0),
    constantVolume_(this->dict_, "constantVolume", false),
    volUpdateType_(this->dict_, "volumeUpdateMethod")
{
    word updateName;

    // If constantVolume found use it
    if (this->dict_.found("constantVolume"))
    {
        volUpdateType_.setValue(mUndefined);
    }
    else if (this->dict_.readIfPresent("volumeUpdateMethod", updateName))
    {
        // Manual version of Enum<volumeUpdateType>
        if (updateName == "constantRho")
        {
            volUpdateType_.setValue(mConstRho);
        }
        else if (updateName == "constantVolume")
        {
            volUpdateType_.setValue(mConstVol);
        }
        else if (updateName == "updateRhoAndVol")
        {
            volUpdateType_.setValue(mUpdateRhoAndVol);
        }
        else
        {
            // FatalIOErrorInLookup

            FatalIOErrorInFunction(this->dict_)
                << "Unknown volumeUpdateMethod type " << updateName
                << "\n\nValid volumeUpdateMethod types :\n"
                << "(constantRho constantVolume updateRhoAndVol)" << nl
                << exit(FatalIOError);
        }
    }
    else
    {
        constantVolume_.setValue(false);
    }
}


template<class ParcelType>
inline Foam::ReactingParcel<ParcelType>::ReactingParcel
(
    const polyMesh& mesh,
    const barycentric& coordinates,
    const label celli,
    const label tetFacei,
    const label tetPti
)
:
    ParcelType(mesh, coordinates, celli, tetFacei, tetPti),
    mass0_(0.0),
    Y_(0)
{}


template<class ParcelType>
inline Foam::ReactingParcel<ParcelType>::ReactingParcel
(
    const polyMesh& mesh,
    const vector& position,
    const label celli
)
:
    ParcelType(mesh, position, celli),
    mass0_(0.0),
    Y_(0)
{}


template<class ParcelType>
inline Foam::ReactingParcel<ParcelType>::ReactingParcel
(
    const polyMesh& mesh,
    const barycentric& coordinates,
    const label celli,
    const label tetFacei,
    const label tetPti,
    const label typeId,
    const scalar nParticle0,
    const scalar d0,
    const scalar dTarget0,
    const vector& U0,
    const vector& f0,
    const vector& angularMomentum0,
    const vector& torque0,
    const scalarField& Y0,
    const constantProperties& constProps
)
:
    ParcelType
    (
        mesh,
        coordinates,
        celli,
        tetFacei,
        tetPti,
        typeId,
        nParticle0,
        d0,
        dTarget0,
        U0,
        f0,
        angularMomentum0,
        torque0,
        constProps
    ),
    mass0_(0.0),
    Y_(Y0)
{
    // Set initial parcel mass
    mass0_ = this->mass();
}


// * * * * * * * * * constantProperties Member Functions * * * * * * * * * * //

template<class ParcelType>
inline Foam::scalar
Foam::ReactingParcel<ParcelType>::constantProperties::pMin() const
{
    return pMin_.value();
}


template<class ParcelType>
inline bool
Foam::ReactingParcel<ParcelType>::constantProperties::constantVolume() const
{
    return constantVolume_.value();
}


template<class ParcelType>
inline Foam::label
Foam::ReactingParcel<ParcelType>::constantProperties::volUpdateType() const
{
    return volUpdateType_.value();
}


// * * * * * * * * * * ThermoParcel Member Functions * * * * * * * * * * * * //

template<class ParcelType>
inline Foam::scalar Foam::ReactingParcel<ParcelType>::mass0() const
{
    return mass0_;
}


template<class ParcelType>
inline const Foam::scalarField& Foam::ReactingParcel<ParcelType>::Y() const
{
    return Y_;
}


template<class ParcelType>
inline const Foam::scalarField& Foam::ReactingParcel<ParcelType>::YGas() const
{
    return Y_;
}


template<class ParcelType>
inline const Foam::scalarField&
Foam::ReactingParcel<ParcelType>::YLiquid() const
{
    return Y_;
}


template<class ParcelType>
inline const Foam::scalarField&
Foam::ReactingParcel<ParcelType>::YSolid() const
{
    return Y_;
}


template<class ParcelType>
inline Foam::scalar& Foam::ReactingParcel<ParcelType>::mass0()
{
    return mass0_;
}


template<class ParcelType>
inline Foam::scalarField& Foam::ReactingParcel<ParcelType>::Y()
{
    return Y_;
}


// ************************************************************************* //
