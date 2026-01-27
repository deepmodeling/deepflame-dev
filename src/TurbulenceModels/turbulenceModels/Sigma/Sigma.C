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

#include "Sigma.H"
#include "fvOptions.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace LESModels
{

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

template<class BasicTurbulenceModel>
void Sigma<BasicTurbulenceModel>::correctNut()
{
   tmp<volTensorField> tgradU(fvc::grad(this->U_));
   tensorField& gradU = tgradU.ref().primitiveFieldRef();
   volScalarField Dsg(mag(fvc::grad(this->U_)));
   scalarField& DsgCells = Dsg.primitiveFieldRef();
   const cellList& cells = this->mesh().cells();
   Switch zeroSigma = false;

   forAll(cells,celli)
   {
      RectangularMatrix<scalar> gradUMatrix(3,3);
      for (label ii = 0; ii<3; ii++)
      {
         for (label jj = 0; jj<3; jj++)
         {
            gradUMatrix[ii][jj] = gradU[celli].component(ii*3+jj);
         }
      }

      DiagonalMatrix<scalar> sigmas(SVD(gradUMatrix).S());
      scalar sigma_1(max(sigmas));
      scalar sigma_2(sigma_1);
      scalar sigma_3(min(sigmas));

      for (label ii = 0; ii<3; ii++)
      {
         if(sigmas[ii]>=sigma_3 && sigmas[ii]<=sigma_1) sigma_2 = sigmas[ii];

         if(sigmas[ii]<0)
         {
            WarningInFunction
            << "Negative sigular values!!! sigmas = \n"
            << sigmas <<endl;
         }
      }

      if(std::abs(sigma_1)<ROOTVSMALL && !zeroSigma) zeroSigma = true;

      DsgCells[celli] = sigma_3*(sigma_1-sigma_2)*(sigma_2-sigma_3)
                        /(sqr(sigma_1)+VSMALL);
   }

   if(zeroSigma) WarningInFunction << "Zero sigma_1 value!!!" << endl;

   this->nut_ = sqr(Csg_*this->delta())*Dsg;
   this->nut_.correctBoundaryConditions();
   fv::options::New(this->mesh_).correct(this->nut_);

   BasicTurbulenceModel::correctNut();
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
Sigma<BasicTurbulenceModel>::Sigma
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
    Csg_
    (
	dimensioned<scalar>::lookupOrAddToDict
        (
            "Csg",
            this->coeffDict_,
            1.5
        )
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
    if (type == typeName)
    {
        this->printCoeffs(type);
    }
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
bool Sigma<BasicTurbulenceModel>::read()
{
    if (LESeddyViscosity<BasicTurbulenceModel>::read())
    {
        Csg_.read(this->coeffDict());

        return true;
    }
    else
    {
        return false;
    }
}


template<class BasicTurbulenceModel>
void Sigma<BasicTurbulenceModel>::correct()
{
    if (!this->turbulence_)
    {
        return;
    }

    LESeddyViscosity<BasicTurbulenceModel>::correct();
    correctNut();
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace LESModels
} // End namespace Foam

// ************************************************************************* //
