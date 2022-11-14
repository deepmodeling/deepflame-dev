/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
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

Class
    Foam::CanteraMixture

Description
    Foam::CanteraMixture

SourceFiles
    CanteraMixture.C

\*---------------------------------------------------------------------------*/

#ifndef CanteraMixture_H
#define CanteraMixture_H


#include "cantera/zerodim.h"
#include "cantera/transport.h"
#include "cantera/thermo/IdealGasPhase.h"
#include "IOdictionary.H"
#include "fvMesh.H"
#include "word.H"
#include "volMesh.H"
#include "GeometricField.H"
#include "volFieldsFwd.H"
#include "fvPatchField.H"
#include "hashedWordList.H"
#include "physicoChemicalConstants.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                         Class CanteraMixture Declaration
\*---------------------------------------------------------------------------*/

class CanteraMixture
{
    // Private Data

        IOdictionary CanteraTorchProperties_;
        const word CanteraMechanismFile_;
        std::shared_ptr<Cantera::Solution> CanteraSolution_;
        std::shared_ptr<Cantera::ThermoPhase> CanteraGas_;
        word transportModelName_;
        std::shared_ptr<Cantera::Transport> CanteraTransport_;
        hashedWordList species_;
        PtrList<volScalarField> Y_;
        const volScalarField& Tref_;
        const volScalarField& pref_;
        mutable scalarList yTemp_;


        //- Construct as copy (not implemented)
        CanteraMixture(const CanteraMixture&);


public:

    typedef CanteraMixture basicMixtureType;
    typedef CanteraMixture thermoType;
    static const bool incompressible = false;
    static const bool isochoric = false;

    // Constructors

        //- Construct from dictionary, mesh and phase name
        CanteraMixture(const dictionary&, const fvMesh&, const word&);


    // Member Functions

        //- Return the instantiated type name
        static word typeName()
        {
            return "CanteraMixture";
        }

        const CanteraMixture& cellMixture(const label celli) const;

        const CanteraMixture& patchFaceMixture(const label patchi, const label facei) const;

        scalar THE
        (
            const scalar& h,
            const scalar& p,
            const scalar& T
        ) const;

        scalar psi
        (
            const scalar& p,
            const scalar& T
        ) const
        {
            return CanteraGas_->meanMolecularWeight()/CanteraGas_->RT();
        }

        scalar mu
        (
            const scalar& p,
            const scalar& T
        ) const
        {
            return CanteraTransport_->viscosity();
        }

        scalar alphah
        (
            const scalar& p,
            const scalar& T
        ) const
        {
            return CanteraTransport_->thermalConductivity()/(CanteraGas_->cp_mass());
        }

        scalar HE
        (
            const scalar& p,
            const scalar& T
        ) const
        {
            return CanteraGas_->enthalpy_mass(); // J/kg
        }

        scalar Hc() const; // J/kg

        scalar Cp
        (
            const scalar& p,
            const scalar& T
        ) const
        {
            return CanteraGas_->cp_mass(); // J/kg/K
        }

        scalar Cv
        (
            const scalar& p,
            const scalar& T
        ) const
        {
            return CanteraGas_->cv_mass(); // J/kg/K
        }

        scalar gamma
        (
            const scalar& p,
            const scalar& T
        ) const
        {
            return Cp(p, T)/Cv(p, T);
        }

        scalar Cpv
        (
            const scalar& p,
            const scalar& T
        ) const
        {
            return Cp(p, T);
        }

        scalar CpByCpv
        (
            const scalar& p,
            const scalar& T
        ) const
        {
            return 1;
        }

        scalar W() const
        {
            return CanteraGas_->meanMolecularWeight();
        }

        static inline word heName() {return "ha";}

        PtrList<volScalarField>& Y() {return Y_;}
        const PtrList<volScalarField>& Y() const {return Y_;}

        volScalarField& Y(const label i) {return Y_[i];}
        const volScalarField& Y(const label i) const {return Y_[i];}

        volScalarField& Y(const word& specieName) {return Y_[species_[specieName]];}
        const volScalarField& Y(const word& specieName) const {return Y_[species_[specieName]];}

        const hashedWordList& species() const {return species_;}
        int nSpecies() {return int(CanteraGas_->nSpecies());}

        std::shared_ptr<Cantera::ThermoPhase> CanteraGas() {return CanteraGas_;}

        std::shared_ptr<Cantera::Solution> CanteraSolution() {return CanteraSolution_;}

        std::shared_ptr<Cantera::Transport> CanteraTransport() {return CanteraTransport_;}

        const word& transportModelName() {return transportModelName_;}


private:

    scalarList HaTemp_; // J/kmol
    scalarList CpTemp_; // J/(kmol·k)
    scalarList CvTemp_; // J/(kmol·k)
    scalarList muTemp_; // kg/(m·s)


public:

    void calcCp(const scalar T, const scalar p)
    {
        const scalar RR = constant::physicoChemical::R.value()*1e3; // J/(kmol·k)

        CanteraGas_->setState_TP(T, p);

        scalarList Cp_R(nSpecies());
        CanteraGas_->getCp_R(Cp_R.begin());
        CpTemp_ = Cp_R*RR;
        for(int i=0; i<(int)nSpecies(); ++i)
        {
            CvTemp_[i] = CpTemp_[i] - RR;
        }
    }

    void calcMu(const scalar T, const scalar p)
    {
        CanteraGas_->setState_TP(T, p);

        CanteraTransport_->getSpeciesViscosities(muTemp_.begin());
    }

    void calcH(const scalar T, const scalar p)
    {
        const scalar RT = constant::physicoChemical::R.value()*1e3*T; // J/kmol/K

        CanteraGas_->setState_TP(T, p);

        scalarList Ha_RT(nSpecies());
        CanteraGas_->getEnthalpy_RT(Ha_RT.begin());
        HaTemp_ = Ha_RT*RT;
    }

    // J/(kg·K)
    scalar Cp(label i, scalar p, scalar T) const
    {
        return CpTemp_[i]/CanteraGas_->molecularWeight(i);
    }

    // J/(kg·K)
    scalar Cv(label i, scalar p, scalar T) const
    {

        return CvTemp_[i]/CanteraGas_->molecularWeight(i);
    }

    // kg/(m·s)
    scalar mu(label i, scalar p, scalar T) const
    {
        return muTemp_[i];
    }

    // J/kg
    scalar Ha(label i, scalar p, scalar T) const
    {
        return HaTemp_[i]/CanteraGas_->molecularWeight(i);
    }

    scalar Hc(label i) const {return CanteraGas_->Hf298SS(i)/CanteraGas_->molecularWeight(i);} // J/kg

    scalar Hs(label i, scalar p, scalar T) const {return Ha(i, p, T) - Hc(i);} // J/kg

    // W/m/K
    scalar kappa(label i, scalar p, scalar T) const
    {
        // should be kappa of single species

        // but now lack of access function in Cantera
        //return CanteraTransport_->thermalConductivity();

        // use OpenFOAM method to calc kappa
        scalar Cv_ = Cv(i, p, T); // J/(kg·K)
        const scalar RR = constant::physicoChemical::R.value()*1e3; // J/(kmol·k)
        return mu(i, p, T)*Cv_*(1.32 + 1.77*RR/CanteraGas_->molecularWeight(i)/Cv_);
    } // W/m/K

    scalar Wi(label i) const {return CanteraGas_->molecularWeight(i);} // kg/kmol


    //- Read dictionary
    void read(const dictionary&);
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

// #ifdef NoRepository
//     #include "CanteraMixture.C"
// #endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
