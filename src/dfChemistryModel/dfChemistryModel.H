/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
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

Class
    Foam::dfChemistryModel

Description
    Extends base chemistry model by adding a thermo package, and ODE functions.
    Introduces chemistry equation system and evaluation of chemical source
    terms.

SourceFiles
    dfChemistryModelI.H
    dfChemistryModel.C

\*---------------------------------------------------------------------------*/

#ifndef dfChemistryModel_H
#define dfChemistryModel_H

#include <torch/script.h>
#include "CanteraMixture.H"
#include "IOdictionary.H"
#include "Switch.H"
#include "scalarField.H"
#include "volFields.H"
#include "hashedWordList.H"
#include "psiThermo.H"
#include "physicoChemicalConstants.H" // for R
#include "loadBalancing/ChemistryProblem.H"
#include "loadBalancing/ChemistrySolution.H"
#include "loadBalancing/RecvBuffer.H"
#include "loadBalancing/SendBuffer.H"
#include "loadBalancing/LoadBalancer.H"
#include "OFstream.H"
#include "IOmanip.H"
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{


/*---------------------------------------------------------------------------*\
                      Class dfChemistryModel Declaration
\*---------------------------------------------------------------------------*/

template<class ThermoType>
class dfChemistryModel
:
public IOdictionary
{

    // Private Data
    ThermoType& thermo_;
    CanteraMixture& mixture_;
    std::shared_ptr<Cantera::ThermoPhase> CanteraGas_;
    const fvMesh& mesh_;
    Switch chemistry_;

    //- Relative tolerance to control CVode
    scalar relTol_;
    //- Absolute tolerance to control CVode
    scalar absTol_;

    PtrList<volScalarField>& Y_;
    // species mass diffusion coefficients, [kg/m/s]
    PtrList<volScalarField> rhoD_;
    // species absolute enthalpy, [J/kg]
    PtrList<volScalarField> hai_;
    // temp mass fraction
    mutable scalarList yTemp_;
    // temp mass diffusion coefficients
    mutable scalarList dTemp_;
    // temp H/RT
    mutable scalarList hrtTemp_; // absolute_enthalpy/RT
    // temp molar concentration
    mutable scalarList cTemp_;
    // mass change rate, [kg/m^3/s]
    PtrList<volScalarField::Internal> RR_;
    hashedWordList species_;
    volScalarField& alpha_;
    volScalarField& T_;
    const volScalarField& p_;
    const volScalarField& rho_;
    volScalarField& mu_;
    volScalarField& psi_;
    // heat release rate, [J/m^3/s]
    volScalarField Qdot_;

    Switch torchSwitch_;
    word torchModelName_;
    scalarList Xmu_;
    scalarList Xstd_;
    scalarList Ymu_;
    scalarList Ystd_;
    scalar Tact_;
    scalar Qdotact_;

    // Load balancing object
    LoadBalancer balancer_;
    // Field containing chemistry CPU time information
    volScalarField cpuTimes_;
    // A file to output the balancing stats
    autoPtr<OFstream>        cpuSolveFile_;

    // Private Member Functions

        void setNumerics(Cantera::ReactorNet &r);

        //- Disallow copy constructor
        dfChemistryModel(const dfChemistryModel&);

        //- Disallow default bitwise assignment
        void operator=(const dfChemistryModel&);

        // override for Cantera
        template<class DeltaTType>
        scalar solve(const DeltaTType& deltaT);

        template<class DeltaTType>
        scalar canteraSolve(const DeltaTType& deltaT);

        template<class DeltaTType>
        scalar torchSolve(const DeltaTType& deltaT);

public:

    //- Runtime type information
    TypeName("dfChemistry");

    // Constructors

        //- Construct from thermo
        dfChemistryModel(ThermoType& thermo);


    //- Destructor
    ~dfChemistryModel();


    // Member Functions

        //- Solve the reaction system for the given time step
        //  and return the characteristic time
        scalar solve(const scalar deltaT); //outer API-1

        //- Solve the reaction system for the given time step
        //  and return the characteristic time
        scalar solve(const scalarField& deltaT); //outer API-2

        //- Return const access to chemical source terms [kg/m^3/s]
        const volScalarField::Internal& RR(const label i) const {return RR_[i];}

        //- Return access to chemical source terms [kg/m^3/s]
        volScalarField::Internal& RR(const label i) {return RR_[i];}

        //- Return the heat release rate [J/m/s^3]
        const volScalarField& Qdot() const
        {
            return Qdot_;
        }

        const hashedWordList& species() const {return species_;}

        PtrList<volScalarField>& Y() {return Y_;}

        const volScalarField& rhoD(const label i) const {return rhoD_[i];}

        volScalarField& hai(const label i) {return hai_[i];}

        // update T, psi, mu, alpha, rhoD, hai (if needed)
        void correctThermo();

        void solveSingle(ChemistryProblem& problem, ChemistrySolution& solution);

        template<class DeltaTType>
        DynamicList<ChemistryProblem> getProblems(const DeltaTType& deltaT);

        //- Solve a list of chemistry problems and return a list of solutions
        DynamicList<ChemistrySolution>
        solveList(UList<ChemistryProblem>& problems);

        //- Solve the problem buffer coming from the balancer
        RecvBuffer<ChemistrySolution>
        solveBuffer(RecvBuffer<ChemistryProblem>& problems);

        //- Update the reaction rates from a list of solutions
        scalar updateReactionRates(const RecvBuffer<ChemistrySolution>& solutions);

        //- Create a load balancer object
        LoadBalancer createBalancer();

        template<class DeltaTType>
        scalar solve_loadBalance(const DeltaTType& deltaT);

        Foam::autoPtr<Foam::OFstream> logFile(const word& name) const
        {
            mkDir(mesh_.time().path() / "loadBal" / this->group());
            return autoPtr<OFstream>(new OFstream(
                mesh_.time().path() / "loadBal" / this->group() / name));
        }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
    #include "dfChemistryModel.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
