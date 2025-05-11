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

#include "CanteraMixture.H"
#include "fvMesh.H"
#include "cantera/thermo/Species.h"

Foam::word Foam::CanteraMixture::energyName_ = "InvalidEnergyName";

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::CanteraMixture::CanteraMixture
(
    const dictionary& thermoDict,
    const fvMesh& mesh,
    const word& phaseName
)
:
    CanteraTorchProperties_
    (
        IOobject
        (
            "CanteraTorchProperties",
            mesh.time().constant(),
            mesh,
            IOobject::MUST_READ_IF_MODIFIED,
            IOobject::NO_WRITE
        )
    ),
    CanteraMechanismFile_(fileName(CanteraTorchProperties_.lookup("CanteraMechanismFile")).expand()),
    transportModelName_(CanteraTorchProperties_.lookup("transportModel")),
    Tref_(mesh.objectRegistry::lookupObject<volScalarField>("T")),
    pref_(mesh.objectRegistry::lookupObject<volScalarField>("p"))
{
    if(!isFile(CanteraMechanismFile_))
    {
        FatalErrorInFunction
            <<"Chemical mechanism "
            <<CanteraMechanismFile_
            <<" doesn't exist!\n"
            <<exit(FatalError);
    }

    CanteraSolution_ = Cantera::newSolution(CanteraMechanismFile_, "");
    CanteraGas_ = CanteraSolution_->thermo();
    CanteraKinetics_ = CanteraSolution_->kinetics();
    CanteraTransport_ = newTransportMgr(transportModelName_, CanteraGas_.get());

    species_lowCpCoeffs_.resize(nSpecies());
    species_highCpCoeffs_.resize(nSpecies());
    for(int i=0; i<nSpecies(); ++i)
    {
        Cantera::AnyMap map = CanteraGas_->species(i)->thermo->parameters();
        auto coeffs = map["data"].asVector<Cantera::vector_fp>();
        for(int j=0; j<7; ++j)
        {
            const scalar RR = constant::physicoChemical::R.value()*1e3;
            if(coeffs.size() == 1)
            {
                species_lowCpCoeffs_[i][j] = coeffs[0][j]*RR/Wi(i);
                species_highCpCoeffs_[i][j] = coeffs[0][j]*RR/Wi(i);
            }
            else
            if(coeffs.size() == 2)
            {
                species_lowCpCoeffs_[i][j] = coeffs[0][j]*RR/Wi(i);
                species_highCpCoeffs_[i][j] = coeffs[1][j]*RR/Wi(i);
            }
            else
            {
                FatalErrorInFunction
                    <<"Check your Chemical mechanism, species "
                    <<CanteraGas_->speciesName(i)
                    <<"!\n"
                    <<exit(FatalError);
            }
        }

        //size=3
        //Tranges[0] T_low
        //Tranges[1] T_common
        //Tranges[2] T_high
        Cantera::AnyMap map2 = CanteraGas_->species(i)->thermo->input();
        auto Tranges = map2["temperature-ranges"].asVector<double>();
        Tcommon_ = Tranges[1];
    }

    Y_.resize(nSpecies());
    yTemp_.resize(nSpecies());
    HaTemp_.resize(nSpecies());
    CpTemp_.resize(nSpecies());
    CvTemp_.resize(nSpecies());
    muTemp_.resize(nSpecies());
    forAll(Y_, i)
    {
        species_.append(CanteraGas_->speciesName(i));
    }

    tmp<volScalarField> tYdefault;

    forAll(Y_, i)
    {
        IOobject header
        (
            species_[i],
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ
        );

        // check if field exists and can be read
        if (header.typeHeaderOk<volScalarField>(true))
        {
            Y_.set
            (
                i,
                new volScalarField
                (
                    IOobject
                    (
                        species_[i],
                        mesh.time().timeName(),
                        mesh,
                        IOobject::MUST_READ,
                        IOobject::AUTO_WRITE
                    ),
                    mesh
                )
            );
        }
        else
        {
            // Read Ydefault if not already read
            if (!tYdefault.valid())
            {
                word YdefaultName("Ydefault");

                IOobject timeIO
                (
                    YdefaultName,
                    mesh.time().timeName(),
                    mesh,
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE
                );

                IOobject constantIO
                (
                    YdefaultName,
                    mesh.time().constant(),
                    mesh,
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE
                );

                IOobject time0IO
                (
                    YdefaultName,
                    Time::timeName(0),
                    mesh,
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE
                );

                if (timeIO.typeHeaderOk<volScalarField>(true))
                {
                    tYdefault = new volScalarField(timeIO, mesh);
                }
                else if (constantIO.typeHeaderOk<volScalarField>(true))
                {
                    tYdefault = new volScalarField(constantIO, mesh);
                }
                else
                {
                    tYdefault = new volScalarField(time0IO, mesh);
                }
            }

            Y_.set
            (
                i,
                new volScalarField
                (
                    IOobject
                    (
                        species_[i],
                        mesh.time().timeName(),
                        mesh,
                        IOobject::NO_READ,
                        IOobject::AUTO_WRITE
                    ),
                    tYdefault()
                )
            );
        }
    }


}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::CanteraMixture::read(const dictionary& thermoDict)
{
    //mixture_ = ThermoType(thermoDict.subDict("mixture"));
}


const Foam::CanteraMixture& Foam::CanteraMixture::cellMixture(const label celli) const
{
    scalarList y(yTemp_.size());
    forAll(Y_, i)
    {
        y[i] = Y_[i][celli];
    }
    setState_TPY(Tref_[celli], pref_[celli], y.begin());

    return *this;
}


const Foam::CanteraMixture& Foam::CanteraMixture::patchFaceMixture
(
    const label patchi,
    const label facei
) const
{
    scalarList y(yTemp_.size());
    forAll(Y_, i)
    {
        y[i] = Y_[i].boundaryField()[patchi][facei];
    }
    setState_TPY(Tref_.boundaryField()[patchi][facei], pref_.boundaryField()[patchi][facei], y.begin());

    return *this;
}


// ************************************************************************* //
