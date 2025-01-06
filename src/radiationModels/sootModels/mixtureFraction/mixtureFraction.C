/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2013-2019 OpenFOAM Foundation
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

#include "mixtureFraction.H"
#include "dfSingleStepReactingMixture.H"


// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

const Foam::dfSingleStepReactingMixture&
Foam::radiationModels::sootModels::mixtureFraction::checkThermo
(
    const fluidThermo& thermo
)
{
    if (isA<dfSingleStepReactingMixture>(thermo))
    {
        return dynamic_cast<const dfSingleStepReactingMixture& >
        (
            thermo
        );
    }
    else
    {
        FatalErrorInFunction
            << "Inconsistent thermo package for " << thermo.type()
            << "Please select a thermo package based on "
            << "dfSingleStepReactingMixture" << exit(FatalError);

        return dynamic_cast<const dfSingleStepReactingMixture& >
        (
            thermo
        );
    }

}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::radiationModels::sootModels::mixtureFraction::mixtureFraction
(
    const dictionary& dict,
    const fvMesh& mesh,
    const word& modelType
)
:
    sootModel(dict, mesh, modelType),
    soot_
    (
        IOobject
        (
            "soot",
            mesh_.time().timeName(),
            mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh_
    ),
    coeffsDict_(dict.subOrEmptyDict(modelType + "Coeffs")),
    nuSoot_(readScalar(coeffsDict_.lookup("nuSoot"))),
    Wsoot_(readScalar(coeffsDict_.lookup("Wsoot"))),
    sootMax_(-1),
    mappingFieldName_
    (
        coeffsDict_.lookupOrDefault<word>("mappingField", "none")
    ),
    mapFieldMax_(1),
    thermo_(mesh.lookupObject<fluidThermo>(basicThermo::dictName)),
    mixture_(checkThermo(thermo_))
{


    // const Reaction& reaction = mixture_.operator[](0);

    const scalarList& specieStoichCoeffs(mixture_.specieStoichCoeffs());

    scalar totalMol = 0.0;
    // 遍历反应右边各组分
    // forAll(reaction.rhs(), i)
    // {
    //     label speciei = reaction.rhs()[i].index;
    //     totalMol += mag(specieStoichCoeffs[speciei]);
    // }
    forAll(mixture_.getProducts(), i)
    {
        label speciei = mixture_.getProducts()[i];
        totalMol += mag(specieStoichCoeffs[speciei]);
    }

    totalMol += nuSoot_;

    //scalarList Xi(reaction.rhs().size());
    scalarList Xi(mixture_.getProducts().size());

    scalar Wm = 0.0;
    // forAll(reaction.rhs(), i)
    // {
    //     const label speciei = reaction.rhs()[i].index;
    //     Xi[i] = mag(specieStoichCoeffs[speciei])/totalMol;
    //     Wm += Xi[i]*mixture_.Wi(speciei);
    // }
    forAll(mixture_.getProducts(), i)
    {
        const label speciei = mixture_.getProducts()[i];
        Xi[i] = mag(specieStoichCoeffs[speciei])/totalMol;
        Wm += Xi[i]*mixture_.Wi(speciei);
    }

    const scalar XSoot = nuSoot_/totalMol;
    Wm += XSoot*Wsoot_;

    sootMax_ = XSoot*Wsoot_/Wm;

    Info << "Maximum soot mass concentrations: " << sootMax_ << nl;

    if (mappingFieldName_ == "none")
    {
        //const label index = reaction.rhs()[0].index;
        const label index = mixture_.getProducts()[0];
        mappingFieldName_ = mixture_.Y(index).name();
    }

    const label mapFieldIndex = mixture_.species()[mappingFieldName_];

    mapFieldMax_ = mixture_.Yprod0()[mapFieldIndex];

}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::radiationModels::sootModels::mixtureFraction::
~mixtureFraction()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::radiationModels::sootModels::mixtureFraction::correct()
{
    const volScalarField& mapField =
        mesh_.lookupObject<volScalarField>(mappingFieldName_);

    soot_ = sootMax_*(mapField/mapFieldMax_);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
