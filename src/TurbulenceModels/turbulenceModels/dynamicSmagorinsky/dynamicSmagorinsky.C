/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
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

#include "dynamicSmagorinsky.H"
#include "fvOptions.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace LESModels
{

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

template<class BasicTurbulenceModel>
void dynamicSmagorinsky<BasicTurbulenceModel>::correctNut
(
    const tmp<volTensorField>& gradU
)
{
    this->nut_ = max(Cs_*sqr(this->delta())*mag(dev(symm(gradU))),-1.0*this->nu());
    this->nut_.correctBoundaryConditions();
    fv::options::New(this->mesh_).correct(this->nut_);

    BasicTurbulenceModel::correctNut();
}


template<class BasicTurbulenceModel>
void dynamicSmagorinsky<BasicTurbulenceModel>::correctNut()
{
    correctNut(fvc::grad(this->U_));
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
dynamicSmagorinsky<BasicTurbulenceModel>::dynamicSmagorinsky
(
    const alphaField& alpha,
    const rhoField& rho,
    const volVectorField& U,
    const surfaceScalarField& alphaRhoPhi,
    const surfaceScalarField& phi,
    const transportModel& transport,
    const word& propertiesName,
    const word& type
)
:
    LESeddyViscosity<BasicTurbulenceModel>
    (
        type,
        alpha,
        rho,
        U,
        alphaRhoPhi,
        phi,
        transport,
        propertiesName
    ),
    Cs_
    (
        IOobject
        (
            IOobject::groupName("Cs", this->U_.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar ("Cs", dimless,SMALL)
    ),
    k_
    (
        IOobject
        (
            IOobject::groupName("k", this->U_.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        this->mesh_,
        dimensionedScalar ("k", dimensionSet(0,2,-2,0,0,0,0),SMALL)
    ),
    simpleFilter_(U.mesh()),
    filterPtr_(LESfilter::New(U.mesh(), this->coeffDict())),
    filter_(filterPtr_())
{
//    bound(k_, this->kMin_);  Not commented out by J.C. Massey

    if (type == typeName)
    {
        this->printCoeffs(type);
    }
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
bool dynamicSmagorinsky<BasicTurbulenceModel>::read()
{
    if (LESeddyViscosity<BasicTurbulenceModel>::read())
    {
        filter_.read(this->coeffDict());

        return true;
    }
    else
    {
        return false;
    }
}


template<class BasicTurbulenceModel>
void dynamicSmagorinsky<BasicTurbulenceModel>::correct()
{
    if (!this->turbulence_)
    {
        return;
    }

    // Local references
    const volVectorField& U = this->U_;

    LESeddyViscosity<BasicTurbulenceModel>::correct();

    tmp<volTensorField> tgradU(fvc::grad(U));
    const volTensorField& gradU = tgradU();

    volSymmTensorField S(dev(symm(gradU)));
    volScalarField magS(mag(S));

    volVectorField Uf(filter_(U));

    volSymmTensorField Sf(filter_(S));
//    volSymmTensorField Sf(dev(symm(fvc::grad(Uf))));  Not commented out by J.C. Massey

    volScalarField magSf(mag(Sf));


    volSymmTensorField LL = (dev(filter_(sqr(U)) - (sqr(filter_(U))))).ref();

    volSymmTensorField MM
    (
        sqr(this->delta())*(filter_(magS*S) - 4.0*magSf*Sf) // J.C. Massey - 4 comes from test filter size
    );

    volScalarField MMMM = fvc::average(magSqr(MM)).ref();
    MMMM.max(VSMALL);

//  J.C. Massey - computed Cs here is the Smagorinsky constant squared to allow for backscatter
    Cs_= 0.5* fvc::average(LL && MM)/MMMM;


//  J.C. Massey - Below is removed for calculation of k
//    volScalarField KK =
//    0.5*(filter_(magSqr(U)) - magSqr(filter_(U)));

//    volScalarField mm
//    (
//        sqr(this->delta())*(4.0*sqr(mag(Sf)) - filter_(sqr(magS)))
//
//    );

//    volScalarField mmmm = fvc::average(magSqr(mm));
//    mmmm.max(VSMALL);

//    k_ = fvc::average(KK*mm)/mmmm * sqr(this->delta())*magSqr(S);

    correctNut(gradU);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace LESModels
} // End namespace Foam

// ************************************************************************* //
