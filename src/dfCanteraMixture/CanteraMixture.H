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
#include "cantera/kinetics.h"
#include "IOdictionary.H"
#include "fvMesh.H"
#include "word.H"
#include "volMesh.H"
#include "GeometricField.H"
#include "volFieldsFwd.H"
#include "fvPatchField.H"
#include "hashedWordList.H"
#include "physicoChemicalConstants.H"
#include "IOmanip.H"

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
    const string CanteraMechanismFile_;
    std::shared_ptr<Cantera::Solution> CanteraSolution_;
    std::shared_ptr<Cantera::ThermoPhase> CanteraGas_;
    std::shared_ptr<Cantera::Kinetics> CanteraKinetics_;
    word transportModelName_;
    Cantera::Transport* CanteraTransport_;
    hashedWordList species_;
    PtrList<volScalarField> Y_;
    const volScalarField& Tref_;
    const volScalarField& pref_;
    mutable scalarList yTemp_;
    static word energyName_;

    typedef FixedList<scalar, 7> coeffArray;
    scalar Tlow_, Thigh_, Tcommon_;
    mutable coeffArray lowCpCoeffs_;
    mutable coeffArray highCpCoeffs_;


    mutable List<coeffArray> species_lowCpCoeffs_;
    mutable List<coeffArray> species_highCpCoeffs_;


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

    //- Return the instantiated type name
    static word typeName()
    {
        return "CanteraMixture";
    }

    const CanteraMixture& cellMixture(const label celli) const;

    const CanteraMixture& patchFaceMixture(const label patchi, const label facei) const;


    void setY(const scalar* y) const
    {
        std::memcpy(yTemp_.begin(), y, sizeof(scalar)*nSpecies());

        for(int j=0;j<7;++j)
        {
            lowCpCoeffs_[j] = 0;
            highCpCoeffs_[j]  = 0;
        }

        for(int i=0;i<nSpecies();++i)
        {
            for(int j=0;j<7;++j)
            {
                lowCpCoeffs_[j] += y[i]*species_lowCpCoeffs_[i][j];
                highCpCoeffs_[j] += y[i]*species_highCpCoeffs_[i][j];
            }
        }
    }

    scalar psi
    (
        const scalar p,
        const scalar T
    ) const
    {
        const Cantera::AnyMap& map = CanteraGas_->input();
        if(map["thermo"].asString()=="Peng-Robinson" || map["thermo"].asString()=="Redlich-Kwong")
        {
            // for non-ideal EoS, psi = \partial rho/ \partial p at constant enthalpy
            // Currently, Peng-Robinson and Redlich-Kwong are available in Cantera
            // The partial derivative was approximated using backward difference with a step size dx
            double h0 = CanteraGas_->enthalpy_mass();
            double dx = 0.0001;
            double rho1 = CanteraGas_->density();
            setState_HP(h0, p*(1-dx));
            double rho2 = CanteraGas_->density();
            return (rho1-rho2)/(dx*p);
        }
        else
        {
            // for ideal gas, psi = W/RT
            return CanteraGas_->meanMolecularWeight()/CanteraGas_->RT();
        }
    }

    scalar rho
    (
        const scalar p,
        const scalar T
    ) const
    {
        return CanteraGas_->density();
    }

    scalar mu
    (
        const scalar p,
        const scalar T
    ) const
    {
        return CanteraTransport_->viscosity();
    }

    scalar alphah
    (
        const scalar p,
        const scalar T
    ) const
    {
        return CanteraTransport_->thermalConductivity()/(CanteraGas_->cp_mass());
    }


    // J/kg
    scalar ha
    (
        const scalar p,
        const scalar T
    ) const
    {
        const coeffArray& a = coeffs(T);
        return
        (
            ((((a[4]/5.0*T + a[3]/4.0)*T + a[2]/3.0)*T + a[1]/2.0)*T + a[0])*T
        + a[5]
        );
        // + EquationOfState::H(p, T);
    }

    // J/kg
    scalar hs
    (
        const scalar p,
        const scalar T
    ) const
    {
        return ha(p, T) - Hc();
    }

    // J/kg
    scalar Hc() const
    {
#ifdef CANTERA_THERMO
        scalar chemicalEnthalpy = 0;
        forAll(yTemp_, i)
        {
            chemicalEnthalpy += yTemp_[i]*CanteraGas_->Hf298SS(i)/CanteraGas_->molecularWeight(i);
        }
        return chemicalEnthalpy;
#else
        const coeffArray& a = lowCpCoeffs_;
        const scalar Tstd = 298.15;
        return
        (
            (
                (((a[4]/5.0*Tstd + a[3]/4.0)*Tstd + a[2]/3.0)*Tstd + a[1]/2.0)*Tstd
            + a[0]
            )*Tstd + a[5]
        );
#endif
    }

    scalar Hc
    (
        const scalar p,
        const scalar T
    ) const
    {
        return Hc();
    }


    scalar HE
    (
        const scalar p,
        const scalar T
    ) const
    {
#ifdef CANTERA_THERMO
        setState_TP(T, p);
        if(energyName_=="ha") return CanteraGas_->enthalpy_mass(); // J/kg
        else if(energyName_=="hs") return CanteraGas_->enthalpy_mass() - Hc(); // J/kg
        else if(energyName_=="ea") return CanteraGas_->intEnergy_mass(); // J/kg
        else
        {
            FatalErrorInFunction
                << "Wrong CanteraMixture enregy name: " << energyName_
                << abort(FatalError);
        }
#else
        if(energyName_=="ha") return ha(p, T);
        else if(energyName_=="hs") return hs(p, T);
        else if(energyName_=="ea") return ha(p, T) - constant::physicoChemical::R.value()*1e3/W()*T;
        else
        {
            FatalErrorInFunction
                << "Wrong CanteraMixture enregy name: " << energyName_
                << abort(FatalError);
        }
#endif
    }

    scalar Cp
    (
        const scalar p,
        const scalar T
    ) const
    {
#ifdef CANTERA_THERMO
        return CanteraGas_->cp_mass(); // J/kg/K
#else
        const coeffArray& a = coeffs(T);
        return
            ((((a[4]*T + a[3])*T + a[2])*T + a[1])*T + a[0]);
            //+ EquationOfState::Cp(p, T);
#endif
    }

    scalar Cv
    (
        const scalar p,
        const scalar T
    ) const
    {
        return CanteraGas_->cv_mass(); // J/kg/K
    }

    scalar gamma
    (
        const scalar p,
        const scalar T
    ) const
    {
        return Cp(p, T)/Cv(p, T);
    }

    scalar Cpv
    (
        const scalar p,
        const scalar T
    ) const
    {
        return Cp(p, T);
    }

    scalar CpByCpv
    (
        const scalar p,
        const scalar T
    ) const
    {
        return 1;
    }

    scalar W() const
    {
        return CanteraGas_->meanMolecularWeight();
    }

    inline word heName() {return energyName_;}

    PtrList<volScalarField>& Y() {return Y_;}
    const PtrList<volScalarField>& Y() const {return Y_;}

    volScalarField& Y(const label i) {return Y_[i];}
    const volScalarField& Y(const label i) const {return Y_[i];}

    volScalarField& Y(const word& specieName) {return Y_[species_[specieName]];}
    const volScalarField& Y(const word& specieName) const {return Y_[species_[specieName]];}

    const hashedWordList& species() const {return species_;}
    int nSpecies() const {return int(CanteraGas_->nSpecies());}
    int nReactions() const {return int(CanteraKinetics_->nReactions());}

    std::shared_ptr<Cantera::ThermoPhase> CanteraGas() {return CanteraGas_;}

    std::shared_ptr<Cantera::Solution> CanteraSolution() {return CanteraSolution_;}

    std::shared_ptr<Cantera::Kinetics> CanteraKinetics() {return CanteraKinetics_;}

    Cantera::Transport* CanteraTransport() {return CanteraTransport_;}

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

        setState_TP(T, p);

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
        setState_TP(T, p);

        CanteraTransport_->getSpeciesViscosities(muTemp_.begin());
    }

    void calcH(const scalar T, const scalar p)
    {
        const scalar RT = constant::physicoChemical::R.value()*1e3*T; // J/kmol/K

        setState_TP(T, p);

        scalarList Ha_RT(nSpecies());
        CanteraGas_->getEnthalpy_RT(Ha_RT.begin());
        HaTemp_ = Ha_RT*RT;
    }

    inline const coeffArray& coeffs(const scalar T) const
    {
        if (T < Tcommon_)
        {
            return lowCpCoeffs_;
        }
        else
        {
            return highCpCoeffs_;
        }
    }

    inline const coeffArray& coeffs(int i, const scalar T) const
    {
        if (T < Tcommon_)
        {
            return species_lowCpCoeffs_[i];
        }
        else
        {
            return species_highCpCoeffs_[i];
        }
    }

    // J/(kg·K)
    scalar Cp(label i, scalar p, scalar T) const
    {
#ifdef CANTERA_THERMO
        return CpTemp_[i]/CanteraGas_->molecularWeight(i);
#else
        const coeffArray& a = coeffs(i, T);
        return
            ((((a[4]*T + a[3])*T + a[2])*T + a[1])*T + a[0]);
        //+ EquationOfState::Cp(p, T);
#endif
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
#ifdef CANTERA_THERMO
        return HaTemp_[i]/CanteraGas_->molecularWeight(i);
#else
        const coeffArray& a = coeffs(i, T);
        return
        (
            ((((a[4]/5.0*T + a[3]/4.0)*T + a[2]/3.0)*T + a[1]/2.0)*T + a[0])*T
        + a[5]
        );
#endif
    }

    // J/kg
    scalar Hc(label i) const
    {
#ifdef CANTERA_THERMO
        return CanteraGas_->Hf298SS(i)/CanteraGas_->molecularWeight(i);
#else
        const coeffArray& a = species_lowCpCoeffs_[i];
        const scalar Tstd = 298.15;
        return
        (
            (
                (((a[4]/5.0*Tstd + a[3]/4.0)*Tstd + a[2]/3.0)*Tstd + a[1]/2.0)*Tstd
            + a[0]
            )*Tstd + a[5]
        );
#endif
    }

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

    static void setEnergyName(word name)
    {
        energyName_ = name;
    }
    static word getEnergyName()
    {
        return energyName_;
    }

    void setState_TP(scalar T, scalar p) const
    {
        CanteraGas_->setState_TP(T, p);
    }

    void setState_PY(scalar p, scalar* y) const
    {
        setY(y);
        CanteraGas_->setState_PY(p, y);
    }

    void setState_HP(scalar h, scalar p) const {
        CanteraGas_->setState_HP(h, p);
    }

    void setState_TPY(scalar T, scalar p, const scalar* y) const
    {
        setY(y);
        CanteraGas_->setState_TPY(T, p, y);
    }

    inline scalar limit
    (
        const scalar T
    ) const
    {
        // no limit
        return T;
    }


    inline scalar THE
    (
        const scalar H,
        const scalar p,
        const scalar T0
    ) const
    {
#ifdef CANTERA_THERMO
        return CanteraGas_->temperature();
#else
        return T(H, p, T0, &CanteraMixture::HE, &CanteraMixture::Cp, &CanteraMixture::limit);
#endif
    }

    inline scalar T
    (
        const scalar f,
        const scalar p,
        const scalar T0,
        scalar (CanteraMixture::*F)(const scalar, const scalar) const,
        scalar (CanteraMixture::*dFdT)(const scalar, const scalar)
            const,
        scalar (CanteraMixture::*limit)(const scalar) const,
        const bool diagnostics = false
    ) const
    {
        scalar tol_ = 1.0e-4;
        label maxIter_ = 100;

        if (T0 < 0)
        {
            FatalErrorInFunction
                << "Negative initial temperature T0: " << T0
                << abort(FatalError);
        }

        scalar Test = T0;
        scalar Tnew = T0;
        scalar Ttol = T0*tol_;
        int    iter = 0;

        if (diagnostics)
        {
            const unsigned int width = IOstream::defaultPrecision() + 8;

            InfoInFunction
                << "Energy -> temperature conversion failed to converge:" << endl;
            Pout<< setw(width) << "iter"
                << setw(width) << "Test"
                << setw(width) << "e/h"
                << setw(width) << "Cv/p"
                << setw(width) << "Tnew"
                << endl;
        }

        do
        {
            Test = Tnew;
            Tnew =
                (this->*limit)
                (Test - ((this->*F)(p, Test) - f)/(this->*dFdT)(p, Test));

            if (diagnostics)
            {
                const unsigned int width = IOstream::defaultPrecision() + 8;

                Pout<< setw(width) << iter
                    << setw(width) << Test
                    << setw(width) << ((this->*F)(p, Test))
                    << setw(width) << ((this->*dFdT)(p, Test))
                    << setw(width) << Tnew
                    << endl;
            }

            if (iter++ > maxIter_)
            {
                if (!diagnostics)
                {
                    T(f, p, T0, F, dFdT, limit, true);
                }

                FatalErrorInFunction
                    << "Maximum number of iterations exceeded: " << maxIter_
                    << abort(FatalError);
            }

        } while (mag(Tnew - Test) > Ttol);

        return Tnew;
    }
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
