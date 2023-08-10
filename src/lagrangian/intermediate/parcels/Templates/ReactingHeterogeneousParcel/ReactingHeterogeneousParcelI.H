/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2018-2019 OpenCFD Ltd.
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
inline Foam::ReactingHeterogeneousParcel<ParcelType>::constantProperties::
constantProperties()
:
    ParcelType::constantProperties(),
    hRetentionCoeff_(this->dict_, 0.0)
{}


template<class ParcelType>
inline Foam::ReactingHeterogeneousParcel<ParcelType>::constantProperties::
constantProperties
(
    const constantProperties& cp
)
:
    ParcelType::constantProperties(cp),
    hRetentionCoeff_(cp.hRetentionCoeff_)
{}


template<class ParcelType>
inline Foam::ReactingHeterogeneousParcel<ParcelType>::constantProperties::
constantProperties
(
    const dictionary& parentDict
)
:
    ParcelType::constantProperties(parentDict),
    hRetentionCoeff_(this->dict_, "hRetentionCoeff")
{}


template<class ParcelType>
inline Foam::ReactingHeterogeneousParcel<ParcelType>::ReactingHeterogeneousParcel
(
    const polyMesh& mesh,
    const barycentric& coordinates,
    const label celli,
    const label tetFacei,
    const label tetPti
)
:
    ParcelType(mesh, coordinates, celli, tetFacei, tetPti),
    F_(0),
    canCombust_(0)
{}


template<class ParcelType>
inline Foam::ReactingHeterogeneousParcel<ParcelType>::ReactingHeterogeneousParcel
(
    const polyMesh& mesh,
    const vector& position,
    const label celli
)
:
    ParcelType(mesh, position, celli),
    F_(0),
    canCombust_(0)
{}


template<class ParcelType>
inline Foam::ReactingHeterogeneousParcel<ParcelType>::ReactingHeterogeneousParcel
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
    const scalarField& Y,
    const scalarField& F,
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
        Y,
        constProps
    ),
    F_(F),
    canCombust_(0)
{}


// * * * * * * * * * constantProperties Member Functions * * * * * * * * * * //


template<class ParcelType>
inline Foam::scalar
Foam::ReactingHeterogeneousParcel<ParcelType>::constantProperties::
hRetentionCoeff() const
{
    scalar value = hRetentionCoeff_.value();

    if ((value < 0) || (value > 1))
    {
        FatalErrorInFunction
            << "hRetentionCoeff must be in the range 0 to 1" << nl
            << exit(FatalError) << endl;
    }

    return value;
}


// * * * * * * * * * * ThermoParcel Member Functions * * * * * * * * * * * * //

template<class ParcelType>
inline const Foam::scalarField& Foam::ReactingHeterogeneousParcel<ParcelType>::
F() const
{
    return F_;
}


template<class ParcelType>
inline Foam::scalarField& Foam::ReactingHeterogeneousParcel<ParcelType>::
F()
{
    return F_;
}


template<class ParcelType>
inline Foam::label
Foam::ReactingHeterogeneousParcel<ParcelType>::canCombust() const
{
    return canCombust_;
}


template<class ParcelType>
inline Foam::label& Foam::ReactingHeterogeneousParcel<ParcelType>::canCombust()
{
    return canCombust_;
}


// ************************************************************************* //
