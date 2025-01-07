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

#include "infinitelyFastChemistry.H"

namespace Foam
{
namespace combustionModels
{

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class ReactionThermo>
infinitelyFastChemistry<ReactionThermo>::infinitelyFastChemistry
(
    const word& modelType,
    ReactionThermo& thermo,
    const compressibleTurbulenceModel& turb,
    const word& combustionProperties
)
:
    singleStepCombustion<ReactionThermo>
    (
        modelType,
        thermo,
        turb,
        combustionProperties
    ),
    C_(readScalar(this->coeffs().lookup("C")))
{}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

template<class ReactionThermo>
infinitelyFastChemistry<ReactionThermo>::~infinitelyFastChemistry()
{}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

template<class ReactionThermo>
void infinitelyFastChemistry<ReactionThermo>::correct()
{
    this->wFuel_ ==
        dimensionedScalar(dimMass/pow3(dimLength)/dimTime, 0);

    this->singleMixture_.fresCorrect();

    const label fuelI = this->singleMixture_.fuelIndex();

    const volScalarField& YFuel = this->singleMixture_.Y()[fuelI];

    const dimensionedScalar s = this->singleMixture_.s();

    if (this->singleMixture_.species().contains("O2"))
    {
        const volScalarField& YO2 = this->singleMixture_.Y("O2");

        this->wFuel_ ==
            this->rho()/(this->mesh().time().deltaT()*C_)
           *min(YFuel, YO2/s.value());
    }
}


template<class ReactionThermo>
bool infinitelyFastChemistry<ReactionThermo>::read()
{
    if (singleStepCombustion<ReactionThermo>::read())
    {
        this->coeffs().lookup("C") >> C_ ;
        return true;
    }
    else
    {
        return false;
    }
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace combustionModels
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
