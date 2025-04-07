/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2018 OpenFOAM Foundation
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

#include "dfSingleStepReactingMixture.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //


inline const Foam::dimensionedScalar
Foam::dfSingleStepReactingMixture::stoicRatio() const
{
    return stoicRatio_;
}



inline const Foam::dimensionedScalar
Foam::dfSingleStepReactingMixture::s() const
{
    return s_;
}



inline const Foam::dimensionedScalar
Foam::dfSingleStepReactingMixture::qFuel() const
{
    return qFuel_;
}



inline const Foam::List<Foam::scalar>&
Foam::dfSingleStepReactingMixture::specieStoichCoeffs() const
{
    return specieStoichCoeffs_;
}



inline Foam::tmp<Foam::volScalarField>
Foam::dfSingleStepReactingMixture::fres
(
    const label index
) const
{
    return fres_[index];
}



inline Foam::label
Foam::dfSingleStepReactingMixture::inertIndex() const
{
    return inertIndex_;
}



inline Foam::label
Foam::dfSingleStepReactingMixture::fuelIndex() const
{
    return fuelIndex_;
}



inline const Foam::List<int>&
Foam::dfSingleStepReactingMixture::specieProd() const
{
    return specieProd_;
}



inline const Foam::scalarList&
Foam::dfSingleStepReactingMixture::Yprod0() const
{
    return Yprod0_;
}

inline const Foam::List<int>&
Foam::dfSingleStepReactingMixture::getProducts() const
{
    return products_;
}

inline const Foam::List<int>&
Foam::dfSingleStepReactingMixture::getReactants() const
{
    return reactants_;
}

// ************************************************************************* //
