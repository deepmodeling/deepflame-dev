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

#ifdef USE_LIBTORCH
#include <torch/script.h>
#include "DNNInferencer.H"
#include "GpuProblem.H"
#include "GpuSolution.H"
#include "DynamicBuffer.H"
#endif

#ifdef USE_PYTORCH
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include "GpuProblem.H"
#include "GpuSolution.H"
#include "DynamicBuffer.H"
#endif

#include "CanteraMixture.H"
#include "IOdictionary.H"
#include "Switch.H"
#include "scalarField.H"
#include "volFields.H"
#include "hashedWordList.H"
// #include "psiThermo.H"
#include "rhoThermo.H"
#include "physicoChemicalConstants.H" // for R
#include "ChemistryProblem.H"
#include "ChemistrySolution.H"
#include "RecvBuffer.H"
#include "SendBuffer.H"
#include "LoadBalancer.H"
#include "OFstream.H"
#include "IOmanip.H"
#include "PstreamGlobals.H"
// blasDNN
#ifdef USE_BLASDNN
#include "DNNInferencer_blas.H"
#include "DNNThermo_blas.H"

#include <hbwmalloc.h>
#include <memkind.h>
#include <assert.h>
#include <numa.h>
#include <numaif.h>
#include <sys/mman.h>
#include <stdio.h>
#include <stdlib.h>

#define HBM_ALIGNED_SIZE (1 << 21)

template<typename DataType>
int alloc_on_hbm_hugetlb_with_check(DataType** addr, size_t length_bytes) {
        if(length_bytes > 0) {
            length_bytes = (length_bytes + HBM_ALIGNED_SIZE - 1) / HBM_ALIGNED_SIZE * HBM_ALIGNED_SIZE;
            void* mapAddress = NULL;
            mapAddress = mmap(NULL, length_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
            if (mapAddress == NULL) {
                printf("mmap failed ,length = %d \n", length_bytes);
                exit(-1);
            };
            *addr = (DataType*)mapAddress;
            printf("using mmap allocating memory success, length = %d \n", length_bytes);                
        } else {
            printf("illegal length_bytes \n");    
            exit(-1);
        }
}

#endif
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
        std::shared_ptr<Cantera::Kinetics> CanteraKinetics_;
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
        // species chemistry enthalpy, [J/kg]
        scalarList hc_;         
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
        // net production rates, [kg/m^3/s]
        PtrList<volScalarField::Internal> wrate_;
        hashedWordList species_;
        volScalarField& alpha_;
        volScalarField& T_;
        const volScalarField& p_;
        volScalarField& rho_;
        volScalarField mixfrac_;
        volScalarField& mu_;
        volScalarField& psi_;
        // heat release rate, [J/m^3/s]
        volScalarField Qdot_;
        // DNN selection fields
        volScalarField selectDNN_;
        bool useThermoTranNN;
        bool isFirst;

#if defined(USE_BLASDNN)
        std::vector<double> Xmu_, Xstd_, Ymu_, Ystd_,
                            thermomu_, thermostd_;

#endif

#if defined USE_LIBTORCH || defined USE_PYTORCH || defined USE_BLASDNN
        bool useDNN;
        Switch torchSwitch_;
        Switch gpu_;
        Switch gpulog_;

        label cvodeComm;

        // profiling
        double time_allsolve_;
        double time_submaster_;
        double time_sendProblem_;
        double time_RecvProblem_;
        double time_sendRecvSolution_;
        double time_getDNNinputs_;
        double time_DNNinference_;
        double time_updateSolutionBuffer_;
        double time_getProblems_;
#endif

#ifdef USE_LIBTORCH
        DNNInferencer DNNInferencer_;
        int cores_; // The number of cores per GPU when use libtorch
        int GPUsPerNode_;

        word torchModelName1_;
        word torchModelName2_;
        word torchModelName3_;
#endif

#ifdef USE_BLASDNN

#define USE_BLASDNN_HALF
// #define USE_BLASDNN_FLOAT
// #define USE_BLASDNN_DOUBLE

#if defined(USE_BLASDNN_HALF)
        DNNInferencer_blas<__fp16> DNNInferencer_blas_;
        DNNThermo_blas<__fp16> DNNThermo_blas_;
#elif defined(USE_BLASDNN_FLOAT)
        DNNThermo_blas<float> DNNThermo_blas_;
        DNNInferencer_blas<float> DNNInferencer_blas_;
#elif defined(USE_BLASDNN_DOUBLE)
        DNNInferencer_blas<double> DNNInferencer_blas_;
        DNNThermo_blas<double> DNNThermo_blas_;
#else
#error "must define one of USE_BLASDNN_HALF, USE_BLASDNN_FLOAT or USE_BLASDNN_DOUBLE"
#endif

        string BLASDNNModelPath_;
        string thermoDNNModelPath_;
#endif

#ifdef USE_PYTORCH
        int cores_; // The number of cores per node when use pytorch

        // profiling
        double time_vec2ndarray_;
        double time_python_;
        pybind11::module_ call_ThermoTranNN; // to import inference file
#endif

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

        //- override for Cantera
        template<class DeltaTType>
        scalar solve(const DeltaTType& deltaT);

        template<class DeltaTType>
        scalar canteraSolve(const DeltaTType& deltaT);

        //- Solve a single ChemistryProblem and put the solution to ChemistrySolution
        void solveSingle(ChemistryProblem& problem, ChemistrySolution& solution);

        //- Get the list of problems to be solved
        template<class DeltaTType>
        DynamicList<ChemistryProblem> getProblems(const DeltaTType& deltaT);

        //- Solve a list of chemistry problems and return a list of solutions
        DynamicList<ChemistrySolution>
        solveList(UList<ChemistryProblem>& problems);

        //- Solve the problem buffer coming from the balancer
        RecvBuffer<ChemistrySolution>
        solveBuffer(RecvBuffer<ChemistryProblem>& problems);

        //- Update the reaction rates and heat release rates from a list of solutions
        scalar updateReactionRates(const RecvBuffer<ChemistrySolution>& solutions,
            DynamicList<ChemistrySolution>& submasterODESolutions);

        //- Create a load balancer object
        LoadBalancer createBalancer();

        //- Solve the reaction system with DLB algorithm
        template<class DeltaTType>
        scalar solve_CVODE(const DeltaTType& deltaT);

#ifdef USE_BLASDNN
        template<class DeltaTType>
        scalar solve_DNN_blas(const DeltaTType& deltaT);
        void thermoDNN_blas(volScalarField& he, const volScalarField& p, PtrList<volScalarField>& Y,
                volScalarField& rho, volScalarField& T, volScalarField& psi, volScalarField& mu,
                volScalarField& alpha, PtrList<volScalarField>& rhoD);
#endif

        //- Output logFiles
        Foam::autoPtr<Foam::OFstream> logFile(const word& name) const
        {
            mkDir(mesh_.time().path() / "loadBal" / this->group());
            return autoPtr<OFstream>(new OFstream(mesh_.time().path() / "loadBal" / this->group() / name));
        }

#if defined USE_LIBTORCH || defined USE_PYTORCH
        /*============= private methods in solve_DNN =============*/
        //- get a list of GPU problems to be solved
        template<class DeltaTType>
        void getGPUProblems(const DeltaTType& deltaT, Foam::DynamicList<GpuProblem>& GPUproblemList,
            Foam::DynamicList<ChemistryProblem>& CPUproblemList);

        //- get the input for DNN inference
        void getDNNinputs(const DynamicBuffer<GpuProblem>& problemBuffer, std::vector<label>& outputlength,
        std::vector<std::vector<double>>& DNNinputs, std::vector<DynamicBuffer<label>>& cellIDBuffer,
        std::vector<std::vector<label>>& problemCounter);

        //- construct the output
        void updateSolutionBuffer(DynamicBuffer<GpuSolution>& solutionBuffer, const std::vector<std::vector<double>>& results,
            const std::vector<DynamicBuffer<label>>& cellIDBuffer, std::vector<std::vector<label>>& problemCounter);
#endif

#ifdef USE_LIBTORCH
        //- DNN solver that support the use of GPU (multi-cards)
        template<class DeltaTType>
        scalar solve_DNN(const DeltaTType& deltaT);
#endif

#ifdef USE_PYTORCH
        template<class DeltaTType>
        scalar solve_DNN(const DeltaTType& deltaT);
#endif

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

        void calculateW();

        volScalarField::Internal& wrate(const label i) {return wrate_[i];}

        tmp<volScalarField::Internal> calculateRR
        (
            const label reactionI,
            const label speciei
        ) const;

        //- Return the heat release rate [J/m/s^3]
        const volScalarField& Qdot() const
        {
            return Qdot_;
        }

        const hashedWordList& species() const {return species_;}

        PtrList<volScalarField>& Y() {return Y_;}

        const volScalarField& rhoD(const label i) const {return rhoD_[i];}

        const volScalarField& hai(const label i) {return hai_[i];}
        
        const scalar & hci(const label i) {return hc_[i];}

        // update T, psi, mu, alpha, rhoD, hai (if needed)
        void correctThermo();

        ThermoType& thermo() {return thermo_;}

        const CanteraMixture& mixture() {return mixture_;}

        void setEnergyName(word name){
            mixture_.setEnergyName(name);
        }
        void updateEnergy()
        {
            scalarField& heCells = thermo_.he().primitiveFieldRef();
            const scalarField& pCells = p_;
            const scalarField& TCells = T_;

            forAll(heCells, celli)
            {
                heCells[celli] =
                    mixture_.cellMixture(celli).HE(pCells[celli], TCells[celli]);
            }

            volScalarField::Boundary& heBf = thermo_.he().boundaryFieldRef();

            forAll(heBf, patchi)
            {
                fvPatchScalarField& phe = heBf[patchi];
                forAll(phe, facei)
                {
                    phe[facei] = mixture_.patchFaceMixture(patchi,facei).HE
                    (
                        p_.boundaryField()[patchi][facei],
                        T_.boundaryField()[patchi][facei]
                    );
                }
            }
        }

        bool ifChemstry() const {return chemistry_;}

    // profiling
#if defined USE_LIBTORCH || defined USE_PYTORCH
    double time_allsolve() {return time_allsolve_;}
    double time_submaster() {return time_submaster_;}
    double time_sendProblem() {return time_sendProblem_;}
    double time_getProblem() {return time_getProblems_;}
    double time_RecvProblem() {return time_RecvProblem_;}
    double time_sendRecvSolution() {return time_sendRecvSolution_;}
    double time_getDNNinputs() {return time_getDNNinputs_;}
    double time_DNNinference() {return time_DNNinference_;}
    double time_updateSolutionBuffer() {return time_updateSolutionBuffer_;}
#endif
#ifdef USE_PYTORCH
    double time_vec2ndarray() {return time_vec2ndarray_;}
    double time_python() {return time_python_;}
#endif

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
