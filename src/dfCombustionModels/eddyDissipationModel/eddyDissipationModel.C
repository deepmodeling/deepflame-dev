/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2015 OpenFOAM Foundation
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

#include "eddyDissipationModel.H"
#include "turbulenceModel.H"
#include "turbulentFluidThermoModel.H"
#include "volFields.H"

namespace Foam
{
namespace combustionModels
{

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class ReactionThermo>
eddyDissipationModel<ReactionThermo>::eddyDissipationModel
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
    C_(readScalar(this->coeffs().lookup("C_EDC"))),
    Cd_(readScalar(this->coeffs().lookup("C_Diff"))),
    Cstiff_(readScalar(this->coeffs().lookup("C_Stiff"))),
    PV_
    (
        IOobject
        (
            "PV",
            this->mesh().time().timeName(),
            this->mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh(),
        dimensionedScalar("zero",dimless,1.0)
    )
{}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

template<class ReactionThermo>
eddyDissipationModel<ReactionThermo>::~eddyDissipationModel()
{}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

template<class ReactionThermo>
Foam::tmp<Foam::volScalarField>
eddyDissipationModel<ReactionThermo>::rtTurb() const
{
    return C_*this->turbulence().epsilon()/
              max(this->turbulence().k(),
              dimensionedScalar("SMALL",dimVelocity*dimVelocity,SMALL));
}

template<class ReactionThermo>
Foam::tmp<Foam::volScalarField>
eddyDissipationModel<ReactionThermo>::rtDiff() const
{
    const CanteraMixture& mixture_ = dynamic_cast<const CanteraMixture&>(this->thermo());
    const volScalarField& YO2 = mixture_.Y("O2");
    const compressible::LESModel& lesModel =
        YO2.db().lookupObject<compressible::LESModel>
        (
         turbulenceModel::propertiesName
        );

    return Cd_*this->thermo().alpha()/this->rho()/sqr(lesModel.delta());
}

template<class ReactionThermo>
void eddyDissipationModel<ReactionThermo>::correct()
{
    //- Set the product volume field, needed by alphat BC
    calcPV();

    this->wFuel_ ==
        dimensionedScalar("zero", dimMass/pow3(dimLength)/dimTime, 0.0);

    // if (this->active())
    {
        CanteraMixture& mixture_ = dynamic_cast<CanteraMixture&>(this->thermo());
        this->singleMixture_.fresCorrect();

        const label fuelI = this->singleMixture_.fuelIndex();

        const volScalarField& YFuel = mixture_.Y(fuelI);

        const dimensionedScalar s = this->singleMixture_.s();

        if (mixture_.species().contains("O2"))
        {
            const volScalarField& YO2 = mixture_.Y("O2");

//            this->wFuel_ ==
//                this->rho()/(this->mesh().time().deltaT()*C_)
//               *min(YFuel, YO2/s.value());

/*
            this->wFuel_ ==
                  C_
                * this->rho()
                * this->turbulence().epsilon()
                / max(this->turbulence().k(),
                  dimensionedScalar("SMALL",dimVelocity*dimVelocity,SMALL))
                * min(YFuel, YO2/s.value());
*/

/*
            this->wFuel_ ==
                  this->rho()
                * min(YFuel, YO2/s.value())
                * max(rtTurb(),rtDiff());
*/

            volScalarField rt(max(rtTurb(),rtDiff()));

            // clipping of wFuel to prevent negative HRR
            // this->wFuel_ ==
            //     this->rho()
            //     * min(max(0*YFuel,YFuel), max(0*YO2,YO2)/s.value())
            //     / this->mesh_.time().deltaT() *
            //     min(1./ Cstiff_* (1 - exp(- Cstiff_*this->mesh_.time().deltaT() * rt)),1.0);

            this->wFuel_ ==
                  this->rho()
                * min(YFuel, YO2/s.value())
                / this->mesh_.time().deltaT() / Cstiff_
                * (1 - exp(- Cstiff_*this->mesh_.time().deltaT() * rt));
        }
    }
}


template<class ReactionThermo>
bool eddyDissipationModel<ReactionThermo>::read()
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

template<class ReactionThermo>
label eddyDissipationModel<ReactionThermo>::getParameter() const
{
    return 123;
}

template<class ReactionThermo>
void eddyDissipationModel<ReactionThermo>::calcPV()
{
    CanteraMixture& mixture_ = dynamic_cast<CanteraMixture&>(this->thermo());
    //- Get species mass fraction
    const label fuelI = this->singleMixture_.fuelIndex();
    const volScalarField& YFuel = mixture_.Y(fuelI);
    const volScalarField& YO2 = mixture_.Y("O2");
    const volScalarField& YCO2 = mixture_.Y("CO2");

    const dimensionedScalar s = this->singleMixture_.s();

    //- Get Mspecies/Mfuel from reaction equation
    scalar rCO2(this->singleMixture_.specieStoichCoeffs()
            [mixture_.species()["CO2"]]);
    scalar rH2O(this->singleMixture_.specieStoichCoeffs()
            [mixture_.species()["H2O"]]);

    PV_ = (YCO2*(1.0+rH2O/rCO2)+SMALL)/(YCO2*(1.0+rH2O/rCO2)+SMALL + min(YFuel,YO2/s.value())*(1.0+s.value()));

    return;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace combustionModels
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
