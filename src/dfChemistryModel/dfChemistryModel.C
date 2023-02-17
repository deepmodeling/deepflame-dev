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

\*---------------------------------------------------------------------------*/

#include "dfChemistryModel.H"
#include "UniformField.H"
#include "clockTime.H"
#include "runtime_assert.H"


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class ThermoType>
Foam::dfChemistryModel<ThermoType>::dfChemistryModel
(
    ThermoType& thermo
)
:
    IOdictionary
    (
        IOobject
        (
            "CanteraTorchProperties",
            thermo.db().time().constant(),
            thermo.db(),
            IOobject::MUST_READ_IF_MODIFIED,
            IOobject::NO_WRITE
        )
    ),
    thermo_(thermo),
    mixture_(dynamic_cast<CanteraMixture&>(thermo)),
    CanteraGas_(mixture_.CanteraGas()),
    mesh_(thermo.p().mesh()),
    chemistry_(lookup("chemistry")),
    relTol_(this->subDict("odeCoeffs").lookupOrDefault("relTol",1e-9)),
    absTol_(this->subDict("odeCoeffs").lookupOrDefault("absTol",1e-15)),
    Y_(mixture_.Y()),
    rhoD_(mixture_.nSpecies()),
    hai_(mixture_.nSpecies()),
    hc_(mixture_.nSpecies()),
    yTemp_(mixture_.nSpecies()),
    dTemp_(mixture_.nSpecies()),
    hrtTemp_(mixture_.nSpecies()),
    cTemp_(mixture_.nSpecies()),
    RR_(mixture_.nSpecies()),
    alpha_(const_cast<volScalarField&>(thermo.alpha())),
    T_(thermo.T()),
    p_(thermo.p()),
    rho_(mesh_.objectRegistry::lookupObject<volScalarField>("rho")),
    mu_(const_cast<volScalarField&>(dynamic_cast<psiThermo&>(thermo).mu()())),
    psi_(const_cast<volScalarField&>(dynamic_cast<psiThermo&>(thermo).psi())),
    Qdot_
    (
        IOobject
        (
            "Qdot",
            mesh_.time().timeName(),
            mesh_,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        mesh_,
        dimensionedScalar(dimEnergy/dimVolume/dimTime, 0)
    ),
    selectDNN_
    (
        IOobject
        (
            "selectDNN",
            mesh_.time().timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh_,
        dimensionedScalar(dimless, -1)
    ),
    balancer_(createBalancer()),
    cpuTimes_
    (
        IOobject
        (
            "cellCpuTimes",
            mesh_.time().timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh_,
        scalar(0.0)
    )
{

#if defined USE_LIBTORCH || defined USE_PYTORCH
    useDNN = true;
    if (!Qdot_.typeHeaderOk<volScalarField>())
    {
        useDNN = false;
    }

    torchSwitch_ = this->subDict("TorchSettings").lookupOrDefault("torch", false);
    gpu_ = this->subDict("TorchSettings").lookupOrDefault("GPU", false),
    gpulog_ = this->subDict("TorchSettings").lookupOrDefault("log", false),

    time_allsolve_ = 0;
    time_submaster_ = 0;
    time_sendProblem_ = 0;
    time_RecvProblem_ = 0;
    time_sendRecvSolution_ = 0;
    time_getDNNinputs_ = 0;
    time_DNNinference_ = 0;
    time_updateSolutionBuffer_ = 0;
    time_getProblems_ = 0;
#endif

#ifdef USE_LIBTORCH
    torchModelName_ = this->lookupOrDefault("torchModel", word(""));
    torchModelName1_ = this->subDict("TorchSettings").lookupOrDefault("torchModel1", word(""));
    torchModelName2_ = this->subDict("TorchSettings").lookupOrDefault("torchModel2", word(""));
    torchModelName3_ = this->subDict("TorchSettings").lookupOrDefault("torchModel3", word(""));

    // set the number of cores slaved by each GPU card
    cores_ = this->subDict("TorchSettings").lookupOrDefault("coresPerGPU", 8);
    GPUsPerNode_ = this->subDict("TorchSettings").lookupOrDefault("GPUsPerNode", 4);

    // initialization the Inferencer (if use multi GPU)
    if(torchSwitch_)
    {
        if(!(Pstream::myProcNo() % cores_)) // Now is a master
        {
            torch::jit::script::Module torchModel1_ = torch::jit::load(torchModelName1_);
            torch::jit::script::Module torchModel2_ = torch::jit::load(torchModelName2_);
            torch::jit::script::Module torchModel3_ = torch::jit::load(torchModelName3_);
            std::string device_;
            if (gpu_)
            {
                int CUDANo = (Pstream::myProcNo() / cores_) % GPUsPerNode_;
                device_ = "cuda:" + std::to_string(CUDANo);
            }
            else
            {
                device_ = "cpu";
            }
            DNNInferencer DNNInferencer(torchModel1_, torchModel2_, torchModel3_, device_);
            DNNInferencer_ = DNNInferencer;
        }
    }
#endif

#ifdef USE_PYTORCH
    cores_ = this->subDict("TorchSettings").lookupOrDefault("coresPerNode", 8);

    time_vec2ndarray_ = 0;
    time_python_ = 0;
#endif

#if defined USE_LIBTORCH || defined USE_PYTORCH
    // if use torch, create new communicator for solving cvode
    if (torchSwitch_)
    {
        labelList subRank;
        for (size_t rank = 0; rank < Pstream::nProcs(); rank ++)
        {
            if (rank % cores_)
            {
                subRank.append(rank);
            }
        }
        cvodeComm = UPstream::allocateCommunicator(UPstream::worldComm, subRank, true);
        if(Pstream::myProcNo() % cores_)
        {
            label sub_rank;
            MPI_Comm_rank(PstreamGlobals::MPICommunicators_[cvodeComm], &sub_rank);
            std::cout<<"my ProcessNo in worldComm = " << Pstream::myProcNo() << ' '
            << "my ProcessNo in cvodeComm = "<<Pstream::myProcNo(cvodeComm)<<std::endl;
        }
    }
#endif

    for(const auto& name : CanteraGas_->speciesNames())
    {
        species_.append(name);
    }
    forAll(RR_, fieldi)
    {
        RR_.set
        (
            fieldi,
            new volScalarField::Internal
            (
                IOobject
                (
                    "RR." + Y_[fieldi].name(),
                    mesh_.time().timeName(),
                    mesh_,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE
                ),
                mesh_,
                dimensionedScalar(dimMass/dimVolume/dimTime, 0)
            )
        );
    }

    forAll(rhoD_, i)
    {
        rhoD_.set
        (
            i,
            new volScalarField
            (
                IOobject
                (
                    "rhoD_" + Y_[i].name(),
                    mesh_.time().timeName(),
                    mesh_,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE
                ),
                mesh_,
                dimensionedScalar(dimDensity*dimViscosity, 0) // kg/m/s
            )
        );
    }

    forAll(hai_, i)
    {
        hai_.set
        (
            i,
            new volScalarField
            (
                IOobject
                (
                    "hai_" + Y_[i].name(),
                    mesh_.time().timeName(),
                    mesh_,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE
                ),
                mesh_,
                dimensionedScalar(dimEnergy/dimMass, 0)
            )
        );
    }
    if(balancer_.log())
    {
        cpuSolveFile_ = logFile("cpu_solve.out");
        cpuSolveFile_() << "                  time" << tab
                        << "           getProblems" << tab
                        << "           updateState" << tab
                        << "               balance" << tab
                        << "           solveBuffer" << tab
                        << "             unbalance" << tab
                        << "               rank ID" << endl;
    }

    Info<<"--- I am here in Cantera-construct ---"<<endl;
    Info<<"relTol_ === "<<relTol_<<endl;
    Info<<"absTol_ === "<<absTol_<<endl;

    forAll(hc_, i)
    {
        hc_[i] = CanteraGas_->Hf298SS(i)/CanteraGas_->molecularWeight(i);
    }
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

template<class ThermoType>
Foam::dfChemistryModel<ThermoType>::
~dfChemistryModel()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //


template<class ThermoType>
template<class DeltaTType>
Foam::scalar Foam::dfChemistryModel<ThermoType>::solve
(
    const DeltaTType& deltaT
)
{
    scalar result = 0;
#ifdef USE_LIBTORCH
    if(torchSwitch_)
    {
        if (useDNN)
        {
            result = solve_DNN(deltaT);
        }
        else
        {
            result = solve_CVODE(deltaT);
            useDNN = true;
        }
    }
    else
    {
        result = solve_CVODE(deltaT);
    }
#elif USE_PYTORCH
    if(torchSwitch_)
    {
        if (useDNN)
        {
            result = solve_DNN(deltaT);
        }
        else
        {
            result = solve_CVODE(deltaT);
            useDNN = true;
        }
    }
    else
    {
        result = solve_CVODE(deltaT);
    }
#else
    result = solve_CVODE(deltaT);
#endif

    return result;
}

template<class ThermoType>
Foam::scalar Foam::dfChemistryModel<ThermoType>::solve
(
    const scalar deltaT
)
{
    // Don't allow the time-step to change more than a factor of 2
    return min
    (
        this->solve<UniformField<scalar>>(UniformField<scalar>(deltaT)),
        2*deltaT
    );
}


template<class ThermoType>
Foam::scalar Foam::dfChemistryModel<ThermoType>::solve
(
    const scalarField& deltaT
)
{
    return this->solve<scalarField>(deltaT);
}


template<class ThermoType>
void Foam::dfChemistryModel<ThermoType>::setNumerics(Cantera::ReactorNet &sim)
{
    sim.setTolerances(relTol_,absTol_);
}

template<class ThermoType>
void Foam::dfChemistryModel<ThermoType>::correctThermo()
{
    psi_.oldTime();

    forAll(T_, celli)
    {
        forAll(Y_, i)
        {
            yTemp_[i] = Y_[i][celli];
        }
        CanteraGas_->setState_PY(p_[celli], yTemp_.begin());
        CanteraGas_->setState_HP(thermo_.he()[celli], p_[celli]); // setState_HP needs (J/kg)

        T_[celli] = CanteraGas_->temperature();

        psi_[celli] = CanteraGas_->meanMolecularWeight()/CanteraGas_->RT(); // meanMolecularWeight() kg/kmol    RT() Joules/kmol

        mu_[celli] = mixture_.CanteraTransport()->viscosity(); // Pa-s

        alpha_[celli] = mixture_.CanteraTransport()->thermalConductivity()/(CanteraGas_->cp_mass()); // kg/(m*s)
        // thermalConductivity() W/m/K
        // cp_mass()   J/kg/K

        if (mixture_.transportModelName() == "UnityLewis")
        {
            forAll(rhoD_, i)
            {
                rhoD_[i][celli] = alpha_[celli];
            }
        }
        else
        {
            mixture_.CanteraTransport()->getMixDiffCoeffsMass(dTemp_.begin()); // m2/s

            CanteraGas_->getEnthalpy_RT(hrtTemp_.begin()); //hrtTemp_=m_h0_RT non-dimension
            // constant::physicoChemical::R.value()   J/(mol·k)
            const scalar RT = constant::physicoChemical::R.value()*1e3*T_[celli]; // J/kmol/K
            forAll(rhoD_, i)
            {
                rhoD_[i][celli] = rho_[celli]*dTemp_[i];

                // CanteraGas_->molecularWeight(i)    kg/kmol
                hai_[i][celli] = hrtTemp_[i]*RT/CanteraGas_->molecularWeight(i);
            }
        }
    }


    const volScalarField::Boundary& pBf = p_.boundaryField();

    const volScalarField::Boundary& rhoBf = rho_.boundaryField();

    volScalarField::Boundary& TBf = T_.boundaryFieldRef();

    volScalarField::Boundary& psiBf = psi_.boundaryFieldRef();

    volScalarField::Boundary& hBf = thermo_.he().boundaryFieldRef();

    volScalarField::Boundary& muBf = mu_.boundaryFieldRef();

    volScalarField::Boundary& alphaBf = alpha_.boundaryFieldRef();

    forAll(T_.boundaryField(), patchi)
    {
        const fvPatchScalarField& pp = pBf[patchi];
        const fvPatchScalarField& prho = rhoBf[patchi];
        fvPatchScalarField& pT = TBf[patchi];
        fvPatchScalarField& ppsi = psiBf[patchi];
        fvPatchScalarField& ph = hBf[patchi];
        fvPatchScalarField& pmu = muBf[patchi];
        fvPatchScalarField& palpha = alphaBf[patchi];

        if (pT.fixesValue())
        {
            forAll(pT, facei)
            {
                forAll(Y_, i)
                {
                    yTemp_[i] = Y_[i].boundaryField()[patchi][facei];
                }
                CanteraGas_->setState_TPY(pT[facei], pp[facei], yTemp_.begin());

                ph[facei] = CanteraGas_->enthalpy_mass();

                ppsi[facei] = CanteraGas_->meanMolecularWeight()/CanteraGas_->RT();

                pmu[facei] = mixture_.CanteraTransport()->viscosity();

                palpha[facei] = mixture_.CanteraTransport()->thermalConductivity()/(CanteraGas_->cp_mass());

                if (mixture_.transportModelName() == "UnityLewis")
                {
                    forAll(rhoD_, i)
                    {
                        rhoD_[i].boundaryFieldRef()[patchi][facei] = palpha[facei];
                    }
                }
                else
                {
                    mixture_.CanteraTransport()->getMixDiffCoeffsMass(dTemp_.begin());

                    CanteraGas_->getEnthalpy_RT(hrtTemp_.begin());
                    const scalar RT = constant::physicoChemical::R.value()*1e3*pT[facei];
                    forAll(rhoD_, i)
                    {
                        rhoD_[i].boundaryFieldRef()[patchi][facei] = prho[facei]*dTemp_[i];

                        hai_[i].boundaryFieldRef()[patchi][facei] = hrtTemp_[i]*RT/CanteraGas_->molecularWeight(i);
                    }
                }
            }
        }
        else
        {
            forAll(pT, facei)
            {
                forAll(Y_, i)
                {
                    yTemp_[i] = Y_[i].boundaryField()[patchi][facei];
                }
                CanteraGas_->setState_PY(pp[facei], yTemp_.begin());
                CanteraGas_->setState_HP(ph[facei], pp[facei]);

                pT[facei] = CanteraGas_->temperature();

                ppsi[facei] = CanteraGas_->meanMolecularWeight()/CanteraGas_->RT();

                pmu[facei] = mixture_.CanteraTransport()->viscosity();

                palpha[facei] = mixture_.CanteraTransport()->thermalConductivity()/(CanteraGas_->cp_mass());

                if (mixture_.transportModelName() == "UnityLewis")
                {
                    forAll(rhoD_, i)
                    {
                        rhoD_[i].boundaryFieldRef()[patchi][facei] = palpha[facei];
                    }
                }
                else
                {
                    mixture_.CanteraTransport()->getMixDiffCoeffsMass(dTemp_.begin());

                    CanteraGas_->getEnthalpy_RT(hrtTemp_.begin());
                    const scalar RT = constant::physicoChemical::R.value()*1e3*pT[facei];
                    forAll(rhoD_, i)
                    {
                        rhoD_[i].boundaryFieldRef()[patchi][facei] = prho[facei]*dTemp_[i];

                        hai_[i].boundaryFieldRef()[patchi][facei] = hrtTemp_[i]*RT/CanteraGas_->molecularWeight(i);
                    }
                }
            }
        }
    }
}

template<class ThermoType>
void Foam::dfChemistryModel<ThermoType>::solveSingle
(
    ChemistryProblem& problem, ChemistrySolution& solution
)
{

    // Timer begins
    clockTime time;
    time.timeIncrement();

    Cantera::Reactor react;
    const scalar Ti = problem.Ti;
    const scalar pi = problem.pi;
    const scalar rhoi = problem.rhoi;
    const scalarList yPre_ = problem.Y;
    scalar Qdoti_ = 0;

    CanteraGas_->setState_TPY(Ti, pi, yPre_.begin());

    react.insert(mixture_.CanteraSolution());
    react.setEnergy(0); // keep T const before and after sim.advance. this will give you a little improvement
    Cantera::ReactorNet sim;
    sim.addReactor(react);
    setNumerics(sim);

    sim.advance(problem.deltaT);

    CanteraGas_->getMassFractions(yTemp_.begin());

    for (int i=0; i<mixture_.nSpecies(); i++)
    {
        solution.RRi[i] = (yTemp_[i] - yPre_[i]) / problem.deltaT * rhoi;
        Qdoti_ -= hc_[i]*solution.RRi[i];
    }

    // Timer ends
    solution.cpuTime = time.timeIncrement();
    solution.Qdoti = Qdoti_;

    solution.cellid = problem.cellid;
    solution.local = problem.local;
}



template <class ThermoType>
template<class DeltaTType>
Foam::DynamicList<Foam::ChemistryProblem>
Foam::dfChemistryModel<ThermoType>::getProblems
(
    const DeltaTType& deltaT
)
{
    const scalarField& T = T_;
    const scalarField& p = p_;

    DynamicList<ChemistryProblem> solved_problems(p.size(), ChemistryProblem(mixture_.nSpecies()));

    forAll(T, celli)
    {
        {
            for(label i = 0; i < mixture_.nSpecies(); i++)
            {
                yTemp_[i] = Y_[i][celli];
            }

            CanteraGas_->setState_TPY(T[celli], p[celli], yTemp_.begin());
            CanteraGas_->getConcentrations(cTemp_.begin());

            ChemistryProblem problem;
            problem.Y = yTemp_;
            problem.Ti = T[celli];
            problem.pi = p[celli];
            problem.rhoi = rho_[celli];
            problem.deltaT = deltaT[celli];
            problem.cpuTime = cpuTimes_[celli];
            problem.cellid = celli;

            solved_problems[celli] = problem;
        }

    }

    return solved_problems;
}


template <class ThermoType>
Foam::DynamicList<Foam::ChemistrySolution>
Foam::dfChemistryModel<ThermoType>::solveList
(
    UList<ChemistryProblem>& problems
)
{
    DynamicList<ChemistrySolution> solutions(
        problems.size(), ChemistrySolution(mixture_.nSpecies()));

    for(label i = 0; i < problems.size(); ++i)
    {
        solveSingle(problems[i], solutions[i]);
    }
    return solutions;
}


template <class ThermoType>
Foam::RecvBuffer<Foam::ChemistrySolution>
Foam::dfChemistryModel<ThermoType>::solveBuffer
(
    RecvBuffer<ChemistryProblem>& problems
)
{
    // allocate the solutions buffer
    RecvBuffer<ChemistrySolution> solutions;

    for(auto& p : problems)
    {
        solutions.append(solveList(p));
    }
    return solutions;
}



template <class ThermoType>
Foam::scalar
Foam::dfChemistryModel<ThermoType>::updateReactionRates
(
    const RecvBuffer<ChemistrySolution>& solutions,
    DynamicList<ChemistrySolution>& submasterODESolutions
)
{
    scalar deltaTMin = great;

    for(const auto& array : solutions)
    {
        for(const auto& solution : array)
        {
            if (solution.local)
            {
                for(label j = 0; j < mixture_.nSpecies(); j++)
                {
                    this->RR_[j][solution.cellid] = solution.RRi[j];
                }
                this->Qdot_[solution.cellid] = solution.Qdoti;

                cpuTimes_[solution.cellid] = solution.cpuTime;
            }
            else
            {
                submasterODESolutions.append(solution);
            }
        }
    }

    return deltaTMin;
}



template <class ThermoType>
Foam::LoadBalancer
Foam::dfChemistryModel<ThermoType>::createBalancer()
{
    const IOdictionary chemistryDict_tmp
        (
            IOobject
            (
                "CanteraTorchProperties",
                thermo_.db().time().constant(),
                thermo_.db(),
                IOobject::MUST_READ,
                IOobject::NO_WRITE,
                false
            )
        );

    return LoadBalancer(chemistryDict_tmp);
}


template <class ThermoType>
template <class DeltaTType>
Foam::scalar Foam::dfChemistryModel<ThermoType>::solve_CVODE
(
    const DeltaTType& deltaT
)
{
    Info<<"=== begin solve_CVODE === "<<endl;
    // CPU time analysis
    clockTime timer;
    scalar t_getProblems(0);
    scalar t_updateState(0);
    scalar t_balance(0);
    scalar t_solveBuffer(0);
    scalar t_unbalance(0);

    if(!this->chemistry_)
    {
        return great;
    }

    timer.timeIncrement();
    DynamicList<ChemistryProblem> allProblems = getProblems(deltaT);
    t_getProblems = timer.timeIncrement();

    RecvBuffer<ChemistrySolution> incomingSolutions;

    if(balancer_.active())
    {
        Info<<"Now DLB algorithm is used!!"<<endl;
        timer.timeIncrement();
        balancer_.updateState(allProblems);
        t_updateState = timer.timeIncrement();

        timer.timeIncrement();
        auto guestProblems = balancer_.balance(allProblems);
        auto ownProblems = balancer_.getRemaining(allProblems);
        t_balance = timer.timeIncrement();

        timer.timeIncrement();
        auto ownSolutions = solveList(ownProblems);
        auto guestSolutions = solveBuffer(guestProblems);
        t_solveBuffer = timer.timeIncrement();

        timer.timeIncrement();
        incomingSolutions = balancer_.unbalance(guestSolutions);
        incomingSolutions.append(ownSolutions);
        t_unbalance = timer.timeIncrement();
    }
    else
    {
        Info<<"Now DLB algorithm is not used!!"<<endl;
        timer.timeIncrement();
        incomingSolutions.append(solveList(allProblems));
        t_solveBuffer = timer.timeIncrement();
    }

    if(balancer_.log())
    {
        balancer_.printState();
        cpuSolveFile_() << setw(22)
                        << this->time().timeOutputValue()<<tab
                        << setw(22) << t_getProblems<<tab
                        << setw(22) << t_updateState<<tab
                        << setw(22) << t_balance<<tab
                        << setw(22) << t_solveBuffer<<tab
                        << setw(22) << t_unbalance<<tab
                        << setw(22) << Pstream::myProcNo()
                        << endl;
    }
    DynamicList<ChemistrySolution> List;
    Info<<"=== end solve_CVODE === "<<endl;
    return updateReactionRates(incomingSolutions, List);
}

#if defined USE_LIBTORCH || defined USE_PYTORCH
template <class ThermoType>
template<class DeltaTType>
void Foam::dfChemistryModel<ThermoType>::getGPUProblems
(
    const DeltaTType &deltaT,
    Foam::DynamicList<GpuProblem>& GPUproblemList,
    Foam::DynamicList<ChemistryProblem>& CPUproblemList
)
{
    DynamicList<GpuProblem> problemList; //single core TODO:rename it

    // get cuda problemList, for all cell
    // each get problem
    forAll(T_, cellI)
    {
        scalar Ti = T_[cellI];
        scalar pi = p_[cellI];
        scalar rhoi = rho_[cellI];

        // if T < 700, set RR=0
        if (T_[cellI] < 700)
        {
            Qdot_[cellI] = 0;
            for (int i = 0; i < mixture_.nSpecies(); i++)
            {
                RR_[i][cellI] = 0.0;
            }
            continue;
        }

        // set problems
        GpuProblem problem(mixture_.nSpecies());
        ChemistryProblem ode_problem(mixture_.nSpecies());
        problem.cellid = cellI;
        problem.Ti = Ti;
        problem.pi = pi/101325;
        for (int i = 0; i < mixture_.nSpecies(); i++)
        {
            problem.Y[i] = Y_[i][cellI];
        }
        problem.rhoi = rhoi;

        // choose DNN module
        if (((Qdot_[cellI] < 3e7) && (T_[cellI] < 2000) && ( T_[cellI] >= 700)) || (T_[cellI] < 700))//choose1
        {
            // if use CVODE
            // ode_problem.Y = problem.Y;
            // ode_problem.Ti = Ti;
            // ode_problem.pi = pi;
            // ode_problem.rhoi = rhoi;
            // ode_problem.deltaT = deltaT[cellI];
            // ode_problem.cpuTime = cpuTimes_[cellI];
            // ode_problem.cellid = cellI;
            // if (!(Pstream::myProcNo() % cores_)) // submaster
            // {
            //     ode_problem.local = false;
            // }
            // CPUproblemList.append(ode_problem);
            
            // selectDNN_[cellI]=0;
            // continue;

            // if use DNN
            problem.DNNid = 0;
            GPUproblemList.append(problem);
            continue;
        }
        if(((Qdot_[cellI] >= 3e7) && (T_[cellI] < 2000)&&(T_[cellI] >= 700))||((Qdot_[cellI] > 7e8) && T_[cellI] > 2000)) //choose2
        {
            problem.DNNid = 1;
            GPUproblemList.append(problem);
            
            selectDNN_[cellI]=1;
            continue;
        }
        if  ((Qdot_[cellI] < 7e8) && (T_[cellI] >= 2000) && (Qdot_[cellI]!=0)) //choose3
        {
            problem.DNNid = 2;
            GPUproblemList.append(problem);
            selectDNN_[cellI]=2;
            continue;
        }

    }

    return;
}

template <class ThermoType>
void Foam::dfChemistryModel<ThermoType>::getDNNinputs
(
    const Foam::DynamicBuffer<GpuProblem>& problemBuffer,
    std::vector<label>& outputLength,
    std::vector<std::vector<double>>& DNNinputs,
    std::vector<Foam::DynamicBuffer<label>>& cellIDBuffer,
    std::vector<std::vector<label>>& problemCounter
)
{
    std::vector<label> problemCounter0;     // evaluate the number of the problems of each subslave for DNN0
    std::vector<label> problemCounter1;     // evaluate the number of the problems of each subslave for DNN1
    std::vector<label> problemCounter2;     // evaluate the number of the problems of each subslave for DNN2
    std::vector<double> inputsDNN0;         // the vector constructed for inference via DNN0
    std::vector<double> inputsDNN1;         // the vector constructed for inference via DNN1
    std::vector<double> inputsDNN2;         // the vector constructed for inference via DNN2
    DynamicList<label> cellIDList0;         // store the cellID of each problem in each subslave for DNN0
    DynamicList<label> cellIDList1;         // store the cellID of each problem in each subslave for DNN1
    DynamicList<label> cellIDList2;         // store the cellID of each problem in each subslave for DNN2
    DynamicBuffer<label> cellIDList0Buffer; // store the cellIDList0 of each subslave
    DynamicBuffer<label> cellIDList1Buffer; // store the cellIDList1 of each subslave
    DynamicBuffer<label> cellIDList2Buffer; // store the cellIDList2 of each subslave

    for (label i = 0; i < cores_; i++) // for all local core TODO: i may cause misleading
    {
        label counter0 = 0;
        label counter1 = 0;
        label counter2 = 0;
        //TODO: parallel the loop
        for (label cellI = 0; cellI < problemBuffer[i].size(); cellI++) // loop cores*problemBuffer[i].size() times
        {
            switch (problemBuffer[i][cellI].DNNid) //divide by Dnn id
            {
            case 0:
                inputsDNN0.push_back(problemBuffer[i][cellI].Ti);
                inputsDNN0.push_back(problemBuffer[i][cellI].pi);
                for (int speciID = 0; speciID < mixture_.nSpecies(); speciID++)
                {
                    inputsDNN0.push_back(problemBuffer[i][cellI].Y[speciID]);
                }
                inputsDNN0.push_back(problemBuffer[i][cellI].rhoi);
                counter0++;
                cellIDList0.append(problemBuffer[i][cellI].cellid); // store cellid for further send back
                break;

            case 1:
                inputsDNN1.push_back(problemBuffer[i][cellI].Ti);
                inputsDNN1.push_back(problemBuffer[i][cellI].pi);
                for (int speciID = 0; speciID < mixture_.nSpecies(); speciID++)
                {
                    inputsDNN1.push_back(problemBuffer[i][cellI].Y[speciID]);
                }
                inputsDNN1.push_back(problemBuffer[i][cellI].rhoi);
                counter1++;
                cellIDList1.append(problemBuffer[i][cellI].cellid);
                break;

            case 2:
                inputsDNN2.push_back(problemBuffer[i][cellI].Ti);
                inputsDNN2.push_back(problemBuffer[i][cellI].pi);
                for (int speciID = 0; speciID < mixture_.nSpecies(); speciID++)
                {
                    inputsDNN2.push_back(problemBuffer[i][cellI].Y[speciID]);
                }
                inputsDNN2.push_back(problemBuffer[i][cellI].rhoi);
                counter2++;
                cellIDList2.append(problemBuffer[i][cellI].cellid);
                break;

            default:
                Info<<"invalid input"<<endl;
                break;
            }
        }
        problemCounter0.push_back(counter0); //count number of inputs mapped to each dnn
        problemCounter1.push_back(counter1);
        problemCounter2.push_back(counter2);
        cellIDList0Buffer.append(cellIDList0);
        cellIDList1Buffer.append(cellIDList1);
        cellIDList2Buffer.append(cellIDList2);
        cellIDList0.clear();
        cellIDList1.clear();
        cellIDList2.clear();
    }

    // get cellNumbers for each model
    label length0 = std::accumulate(problemCounter0.begin(), problemCounter0.end(), 0);
    label length1 = length0 + std::accumulate(problemCounter1.begin(), problemCounter1.end(), 0);
    label length2 = length1 + std::accumulate(problemCounter2.begin(), problemCounter2.end(), 0);

    // set output
    outputLength = {length0, length1, length2};
    DNNinputs = {inputsDNN0, inputsDNN1, inputsDNN2};
    cellIDBuffer = {cellIDList0Buffer, cellIDList1Buffer, cellIDList2Buffer};
    problemCounter = {problemCounter0, problemCounter1, problemCounter2};

    if (gpulog_)
    {
        std::cout<<"inputsDNN0 = "<<inputsDNN0.size()/10 << "\n";
        std::cout<<"inputsDNN1 = "<<inputsDNN1.size()/10 << "\n";
        std::cout<<"inputsDNN2 = "<<inputsDNN2.size()/10 << "\n";
    }

    return;
}

template <class ThermoType>
void Foam::dfChemistryModel<ThermoType>::updateSolutionBuffer
(
    Foam::DynamicBuffer<Foam::GpuSolution>& solutionBuffer,
    const std::vector<std::vector<double>>& results,
    const std::vector<Foam::DynamicBuffer<Foam::label>>& cellIDBuffer,
    std::vector<std::vector<Foam::label>>& problemCounter
)
{
    GpuSolution solution(mixture_.nSpecies());
    DynamicList<GpuSolution> solutionList; //TODO: rename

    label outputCounter0 = 0;
    label outputCounter1 = 0;
    label outputCounter2 = 0;

    for (label i = 0; i < cores_; i++) //TODO: i may cause misleading
    {
        for (int cellI = 0; cellI < problemCounter[0][i]; cellI++)
        {
            for (int speciID = 0; speciID < mixture_.nSpecies(); speciID++)
            {
                solution.RRi[speciID] = results[0][outputCounter0 * mixture_.nSpecies() + speciID];
            }
            solution.cellid = cellIDBuffer[0][i][cellI]; //cellid are sequential so that's fine
            solutionList.append(solution);
            outputCounter0++;
        }
        for (int cellI = 0; cellI < problemCounter[1][i]; cellI++)
        {
            for (int speciID = 0; speciID < mixture_.nSpecies(); speciID++)
            {
                solution.RRi[speciID] = results[1][outputCounter1 * mixture_.nSpecies() + speciID];
            }
            solution.cellid = cellIDBuffer[1][i][cellI];
            solutionList.append(solution);
            outputCounter1++;
        }
        for (int cellI = 0; cellI < problemCounter[2][i]; cellI++)
        {
            for (int speciID = 0; speciID < mixture_.nSpecies(); speciID++)
            {
                solution.RRi[speciID] = results[2][outputCounter2 * mixture_.nSpecies() + speciID];
            }
            solution.cellid = cellIDBuffer[2][i][cellI];
            solutionList.append(solution);
            outputCounter2++;
        }
    solutionBuffer.append(solutionList);
    solutionList.clear();
    }
    return;
}
#endif

#ifdef USE_LIBTORCH
template <class ThermoType>
template <class DeltaTType>
Foam::scalar Foam::dfChemistryModel<ThermoType>::solve_DNN(
    const DeltaTType &deltaT)
{
    scalar deltaTMin = great;
    // set the cores slaved by a DCU
    if (!this->chemistry_)
    {
        return deltaTMin;
    }

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    Info << "=== begin solve_DNN === " << endl;
    if (gpu_)
    {
        Info << "now DNN inference is conducted on GPU" << endl;
    }
    else
    {
        Info << "now DNN inference is conducted on CPU" << endl;
    }

    /*=============================gather problems=============================*/
    std::chrono::steady_clock::time_point start10 = std::chrono::steady_clock::now();
    DynamicList<GpuProblem> GPUproblemList; //single core TODO:rename it
    DynamicList<ChemistryProblem> CPUproblemList;
    getGPUProblems(deltaT, GPUproblemList, CPUproblemList);
    label flag_mpi_init;
    MPI_Initialized(&flag_mpi_init);
    if(flag_mpi_init) MPI_Barrier(PstreamGlobals::MPI_COMM_FOAM);
    std::chrono::steady_clock::time_point stop10 = std::chrono::steady_clock::now();
    std::chrono::duration<double> processingTime10 = std::chrono::duration_cast<std::chrono::duration<double>>(stop10 - start10);
    time_getProblems_ += processingTime10.count();

    /*==============================send problems==============================*/
    std::chrono::steady_clock::time_point start2 = std::chrono::steady_clock::now();

    PstreamBuffers pBufs(Pstream::commsTypes::nonBlocking);
    if (Pstream::myProcNo() % cores_) //for slave
    {
        UOPstream send((Pstream::myProcNo()/cores_)*cores_, pBufs);// sending problem to master
        send << GPUproblemList;
    }
    pBufs.finishedSends();

    /*==============================send CVODE problems from submaster to neighbour==============================*/
    PstreamBuffers pBufs1(Pstream::commsTypes::nonBlocking);
    if (!(Pstream::myProcNo() % cores_)) // submaster
    {
        UOPstream send((Pstream::myProcNo() + 1), pBufs1);// sending CPUproblems to neighbour
        send << CPUproblemList;
    }
    pBufs1.finishedSends();
    if ((Pstream::myProcNo() % cores_) == 1) // neighbour of submaster
    {
        DynamicList<ChemistryProblem> CPUproblemList_submaster;
        UIPstream recv((Pstream::myProcNo() - 1), pBufs1);
        recv >> CPUproblemList_submaster;
        CPUproblemList.append(CPUproblemList_submaster);
    }

    /*========================================================================================================*/

    DynamicBuffer<GpuSolution> solutionBuffer;

    std::chrono::steady_clock::time_point stop2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> processingTime2 = std::chrono::duration_cast<std::chrono::duration<double>>(stop2 - start2);
    // std::cout << "sendProblemTime = " << processingTime2.count() << std::endl;
    time_sendProblem_ += processingTime2.count();

    /*=============================submaster work start=============================*/
    if (!(Pstream::myProcNo() % cores_))
    {
        std::chrono::steady_clock::time_point start1 = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point start3 = std::chrono::steady_clock::now();

        label problemSize = 0; // problemSize is defined to debug
        DynamicBuffer<GpuProblem> problemBuffer(cores_);//each submaster init a local problemBuffer TODO:rename it

        /*==============================gather problems==============================*/
        problemBuffer[0] = GPUproblemList; //problemList of submaster get index 0
        problemSize += problemBuffer[0].size();

        for (label i = 1; i < cores_; i++)
        {
            UIPstream recv(i + Pstream::myProcNo(), pBufs);
            recv >> problemBuffer[i];  //recv previous send problem and append to problemList
            problemSize += problemBuffer[i].size();
        }
        if (gpulog_)
        {
            Info << "problemSize = " << problemSize << endl;
        }

        std::chrono::steady_clock::time_point stop3 = std::chrono::steady_clock::now();
        std::chrono::duration<double> processingTime3 = std::chrono::duration_cast<std::chrono::duration<double>>(stop3 - start3);
        // std::cout << "RecvProblemTime = " << processingTime3.count() << std::endl;
        time_RecvProblem_ += processingTime3.count();

        /*==============================construct DNN inputs==============================*/
        std::vector<label> outputLength;
        std::vector<std::vector<double>> DNNinputs;     // tensors for the inference of DNN
        std::vector<DynamicBuffer<label>> cellIDBuffer; // Buffer contains the cell numbers
        std::vector<std::vector<label>> problemCounter; // evaluate the number of the problems of each subslave

        std::chrono::steady_clock::time_point start5 = std::chrono::steady_clock::now();
        getDNNinputs(problemBuffer, outputLength, DNNinputs, cellIDBuffer, problemCounter);
        std::chrono::steady_clock::time_point stop5 = std::chrono::steady_clock::now();
        std::chrono::duration<double> processingTime5 = std::chrono::duration_cast<std::chrono::duration<double>>(stop5 - start5);
        std::cout << "getDNNinputsTime = " << processingTime5.count() << std::endl;
        time_getDNNinputs_ += processingTime5.count();

        /*=============================inference via DNNInferencer=============================*/
        std::chrono::steady_clock::time_point start7 = std::chrono::steady_clock::now();

        auto results = DNNInferencer_.Inference_multiDNNs(DNNinputs, mixture_.nSpecies() + 3);

        std::chrono::steady_clock::time_point stop7 = std::chrono::steady_clock::now();
        std::chrono::duration<double> processingTime7 = std::chrono::duration_cast<std::chrono::duration<double>>(stop7 - start7);
        std::cout << "DNNinferenceTime = " << processingTime7.count() << std::endl;
        time_DNNinference_ += processingTime7.count();

        /*=============================construct solutions=============================*/
        std::chrono::steady_clock::time_point start6 = std::chrono::steady_clock::now();

        updateSolutionBuffer(solutionBuffer, results, cellIDBuffer, problemCounter);

        std::chrono::steady_clock::time_point stop6 = std::chrono::steady_clock::now();
        std::chrono::duration<double> processingTime6 = std::chrono::duration_cast<std::chrono::duration<double>>(stop6 - start6);
        std::cout << "updateSolutionBufferTime = " << processingTime6.count() << std::endl;
        time_updateSolutionBuffer_ += processingTime6.count();

        std::chrono::steady_clock::time_point stop1 = std::chrono::steady_clock::now();
        std::chrono::duration<double> processingTime1 = std::chrono::duration_cast<std::chrono::duration<double>>(stop1 - start1);
        // std::cout << "submasterTime = " << processingTime1.count() << std::endl;
        time_submaster_ += processingTime1.count();
    }

    /*=============================calculates RR with CVODE use DLB=============================*/
    DynamicList<ChemistrySolution> CPUSolutionList;
    if (Pstream::myProcNo() % cores_) //for slave
    {
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        DynamicBuffer<ChemistrySolution> incomingSolutions;
        balancer_.updateState(CPUproblemList, cvodeComm);
        auto guestProblems = balancer_.balance(CPUproblemList, cvodeComm);
        auto ownProblems = balancer_.getRemaining(CPUproblemList, cvodeComm);
        auto ownSolutions = solveList(ownProblems);
        auto guestSolutions = solveBuffer(guestProblems);
        incomingSolutions = balancer_.unbalance(guestSolutions, cvodeComm);
        incomingSolutions.append(ownSolutions);
        updateReactionRates(incomingSolutions, CPUSolutionList);
        std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();
        std::chrono::duration<double> processingTime = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
        std::cout << "slaveTime = " << processingTime.count() << std::endl;
    }

    /*=============================send CPUSolutionList back to submaster=============================*/
    PstreamBuffers pBufs3(Pstream::commsTypes::nonBlocking);

    if ((Pstream::myProcNo() % cores_) == 1) // neighbour of submaster
    {
        UOPstream send((Pstream::myProcNo() - 1), pBufs3);
        send << CPUSolutionList;
    }
    pBufs3.finishedSends();
    if (!(Pstream::myProcNo() % cores_)) // submaster
    {
        UIPstream recv((Pstream::myProcNo() + 1), pBufs3);// resv CPUproblems from neighbour
        recv >> CPUSolutionList;
    }

    /*=============================send and recv GPUSolutions=============================*/
    std::chrono::steady_clock::time_point start4 = std::chrono::steady_clock::now();

    DynamicList<GpuSolution> finalList;
    PstreamBuffers pBufs2(Pstream::commsTypes::nonBlocking);
    if (!(Pstream::myProcNo() % cores_)) // submaster
    {
        finalList = solutionBuffer[0];
        for (label i = 1; i < cores_; i++)
        {

            UOPstream send(i + Pstream::myProcNo(), pBufs2);
            send << solutionBuffer[i];
        }
    }
    pBufs2.finishedSends();

    if (Pstream::myProcNo() % cores_) // slavers
    {
        UIPstream recv((Pstream::myProcNo()/cores_)*cores_, pBufs2);
        recv >> finalList;
    }

    std::chrono::steady_clock::time_point stop4 = std::chrono::steady_clock::now();
    std::chrono::duration<double> processingTime4 = std::chrono::duration_cast<std::chrono::duration<double>>(stop4 - start4);
    // std::cout << "SendRecvSolutionTime = " << processingTime4.count() << std::endl;
    time_sendRecvSolution_ += processingTime4.count();

    /*=============================update RR fields=============================*/
    for (int cellI = 0; cellI < finalList.size(); cellI++)
    {
        Qdot_[finalList[cellI].cellid] = 0;
        for (int speciID = 0; speciID < mixture_.nSpecies(); speciID++)
        {
            RR_[speciID][finalList[cellI].cellid] = finalList[cellI].RRi[speciID];
            Qdot_[finalList[cellI].cellid] -= hc_[speciID] * RR_[speciID][finalList[cellI].cellid];
        }
    }

    if (!(Pstream::myProcNo() % cores_)) // submaster
    {
        for (int cellI = 0; cellI < CPUSolutionList.size(); cellI++)
        {
            for (int speciID = 0; speciID < mixture_.nSpecies(); speciID++)
            {
                RR_[speciID][CPUSolutionList[cellI].cellid] = CPUSolutionList[cellI].RRi[speciID];
            }
            Qdot_[CPUSolutionList[cellI].cellid] = CPUSolutionList[cellI].Qdoti;
            cpuTimes_[CPUSolutionList[cellI].cellid] = CPUSolutionList[cellI].cpuTime;
        }
    }

    Info << "=== end solve_DNN === " << endl;

    std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();
    std::chrono::duration<double> processingTime = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
    // std::cout << "allSolveTime = " << processingTime.count() << std::endl;
    time_allsolve_ += processingTime.count();

    return deltaTMin;
}
#endif


#ifdef USE_PYTORCH
template <class ThermoType>
template <class DeltaTType>
Foam::scalar Foam::dfChemistryModel<ThermoType>::solve_DNN_GPU_oneCore(
    const DeltaTType &deltaT)
{
    scalar deltaTMin = great;

    if (!this->chemistry_)
    {
        return deltaTMin;
    }
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    Info << "=== begin torchCUDAoneCore-solve === " << endl;

    // set variables
    scalarList yPre_(mixture_.nSpecies());
    scalarList yBCT_(mixture_.nSpecies());
    scalarList u_(mixture_.nSpecies() + 2);
    Cantera::Reactor react;
    double lambda = 0.1;

    std::vector<int> torch_cell;
    label torch_cellname = 0;

    // obtain the number of DNN cells
    std::chrono::steady_clock::time_point start_0 = std::chrono::steady_clock::now();
    forAll(T_, cellI)
    {
        if (T_[cellI] >= Tact1_)
        {
            torch_cell.push_back(cellI);
        }
    }

    // generate GPU inputs and solve CVODE cells
    std::vector<double> inputs_;
    inputs_.reserve(torch_cell.size() * (mixture_.nSpecies() + 3));

    forAll(T_, cellI)
    {
        scalar Ti = T_[cellI];
        scalar pi = p_[cellI];
        scalar rhoi = rho_[cellI];

        if (Ti >= Tact1_)
        {
            Qdot_[cellI] = 0.0;

            // set inputs
            inputs_.push_back(rhoi);
            inputs_.push_back(Ti);
            inputs_.push_back(pi/101325);
            for (int i = 0; i < mixture_.nSpecies(); i++)
            {
                inputs_.push_back(Y_[i][cellI]);
            }
        }
        else
        {
            Qdot_[cellI] = 0.0;
            for (int i = 0; i < mixture_.nSpecies(); i++)
            {
                yPre_[i] = Y_[i][cellI];
            }

            CanteraGas_->setState_TPY(Ti, pi, yPre_.begin());
            react.insert(mixture_.CanteraSolution());
            react.setEnergy(0);

            Cantera::ReactorNet sim;
            sim.addReactor(react);
            setNumerics(sim);
            sim.advance(deltaT);

            CanteraGas_->getMassFractions(yTemp_.begin());

            for (int i = 0; i < mixture_.nSpecies(); i++)
            {
                RR_[i][cellI] = (yTemp_[i] - yPre_[i]) * rhoi / deltaT;
                Qdot_[cellI] -= hc_[i] * RR_[i][cellI];
            }
        }
    }
    std::chrono::steady_clock::time_point stop_0 = std::chrono::steady_clock::now();
    std::chrono::duration<double> processingTime_0 = std::chrono::duration_cast<std::chrono::duration<double>>(stop_0 - start_0);
    std::cout << "beforeCUDATime = " << processingTime_0.count() << std::endl;
    time_getDNNinputs_ += processingTime_0.count();

    // DNN

    std::chrono::steady_clock::time_point start_3 = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point start_4 = std::chrono::steady_clock::now();

    pybind11::array_t<double> vec = pybind11::cast(inputs_);

    std::chrono::steady_clock::time_point stop_4 = std::chrono::steady_clock::now();
    std::chrono::duration<double> processingTime_4 = std::chrono::duration_cast<std::chrono::duration<double>>(stop_4 - start_4);
    std::cout << "vec2ndarrayTime = " << processingTime_4.count() << std::endl;
    time_vec2ndarray_ += processingTime_4.count();

    pybind11::module_ call_torch = pybind11::module_::import("inference"); // import python file

    std::chrono::steady_clock::time_point start_5 = std::chrono::steady_clock::now();

    pybind11::object result = call_torch.attr("inference")(vec); // call function
    const double* star = result.cast<pybind11::array_t<double>>().data();

    std::vector<double> outputsVec(star, star + torch_cell.size() * 7);

    std::chrono::steady_clock::time_point stop_5 = std::chrono::steady_clock::now();
    std::chrono::duration<double> processingTime_5 = std::chrono::duration_cast<std::chrono::duration<double>>(stop_5 - start_5);
    std::cout << "pythonTime = " << processingTime_5.count() << std::endl;
    time_python_ += processingTime_5.count();

    std::chrono::steady_clock::time_point stop_3 = std::chrono::steady_clock::now();
    std::chrono::duration<double> processingTime_3 = std::chrono::duration_cast<std::chrono::duration<double>>(stop_3 - start_3);
    std::cout << "DNNinferenceTime = " << processingTime_3.count() << std::endl;
    time_DNNinference_ += processingTime_3.count();

    std::chrono::steady_clock::time_point start_2 = std::chrono::steady_clock::now();
    for (int cellI = 0; cellI < torch_cell.size(); cellI++)
    {
        // update y
        for (int i = 0; i < mixture_.nSpecies(); i++)
        {
            RR_[i][torch_cell[cellI]] = outputsVec[cellI * 7 + i];
            Qdot_[cellI] -= hc_[i] * RR_[i][cellI];
        }
    }
    std::chrono::steady_clock::time_point stop_2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> processingTime_2 = std::chrono::duration_cast<std::chrono::duration<double>>(stop_2 - start_2);
    std::cout << "afterCUDATime = " << processingTime_2.count() << std::endl;
    time_updateSolutionBuffer_ += processingTime_2.count();

    Info << "=== end torch&ode-solve === " << endl;
    std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();
    std::chrono::duration<double> processingTime = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
    std::cout << "allSolveTime = " << processingTime.count() << std::endl;
    time_allsolve_ += processingTime.count();

    return deltaTMin;
}

template <class ThermoType>
template <class DeltaTType>
Foam::scalar Foam::dfChemistryModel<ThermoType>::solve_DNN(
    const DeltaTType &deltaT)
{
    scalar deltaTMin = great;
    // set the cores slaved by a DCU
    if (!this->chemistry_)
    {
        return deltaTMin;
    }

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    Info << "=== begin solve_DNN === " << endl;
    if (gpu_)
    {
        Info << "now DNN inference is conducted on GPU" << endl;
    }
    else
    {
        Info << "now DNN inference is conducted on CPU" << endl;
    }


    /*=============================gather problems=============================*/
    std::chrono::steady_clock::time_point start10 = std::chrono::steady_clock::now();
    DynamicList<GpuProblem> GPUproblemList; //single core TODO:rename it
    DynamicList<ChemistryProblem> CPUproblemList;
    getGPUProblems(deltaT, GPUproblemList, CPUproblemList);
    label flag_mpi_init;
    MPI_Initialized(&flag_mpi_init);
    if(flag_mpi_init) MPI_Barrier(PstreamGlobals::MPI_COMM_FOAM);
    std::chrono::steady_clock::time_point stop10 = std::chrono::steady_clock::now();
    std::chrono::duration<double> processingTime10 = std::chrono::duration_cast<std::chrono::duration<double>>(stop10 - start10);
    time_getProblems_ += processingTime10.count();

    /*==============================send problems==============================*/
    std::chrono::steady_clock::time_point start2 = std::chrono::steady_clock::now();

    PstreamBuffers pBufs(Pstream::commsTypes::nonBlocking);
    if (Pstream::myProcNo() % cores_) //for slave
    {
        UOPstream send((Pstream::myProcNo()/cores_)*cores_, pBufs);// sending problem to master
        send << GPUproblemList;
    }
    pBufs.finishedSends();

    /*==============================send CVODE problems from submaster to neighbour==============================*/
    PstreamBuffers pBufs1(Pstream::commsTypes::nonBlocking);
    if (!(Pstream::myProcNo() % cores_)) // submaster
    {
        UOPstream send((Pstream::myProcNo() + 1), pBufs1);// sending CPUproblems to neighbour
        send << CPUproblemList;
    }
    pBufs1.finishedSends();
    if ((Pstream::myProcNo() % cores_) == 1) // neighbour of submaster
    {
        DynamicList<ChemistryProblem> CPUproblemList_submaster;
        UIPstream recv((Pstream::myProcNo() - 1), pBufs1);
        recv >> CPUproblemList_submaster;
        CPUproblemList.append(CPUproblemList_submaster);
    }

    /*========================================================================================================*/

    DynamicBuffer<GpuSolution> solutionBuffer;

    std::chrono::steady_clock::time_point stop2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> processingTime2 = std::chrono::duration_cast<std::chrono::duration<double>>(stop2 - start2);
    // std::cout << "sendProblemTime = " << processingTime2.count() << std::endl;
    time_sendProblem_ += processingTime2.count();

    /*=============================submaster work start=============================*/
    if (!(Pstream::myProcNo() % cores_))
    {
        std::chrono::steady_clock::time_point start1 = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point start3 = std::chrono::steady_clock::now();

        label problemSize = 0; // problemSize is defined to debug
        DynamicBuffer<GpuProblem> problemBuffer(cores_);//each submaster init a local problemBuffer TODO:rename it

        /*==============================gather problems==============================*/
        problemBuffer[0] = GPUproblemList; //problemList of submaster get index 0
        problemSize += problemBuffer[0].size();

        for (label i = 1; i < cores_; i++)
        {
            UIPstream recv(i + Pstream::myProcNo(), pBufs);
            recv >> problemBuffer[i];  //recv previous send problem and append to problemList
            problemSize += problemBuffer[i].size();
        }
        if (gpulog_)
        {
            Info << "problemSize = " << problemSize << endl;
        }

        std::chrono::steady_clock::time_point stop3 = std::chrono::steady_clock::now();
        std::chrono::duration<double> processingTime3 = std::chrono::duration_cast<std::chrono::duration<double>>(stop3 - start3);
        // std::cout << "RecvProblemTime = " << processingTime3.count() << std::endl;
        time_RecvProblem_ += processingTime3.count();

        /*==============================construct DNN inputs==============================*/
        std::vector<label> outputLength;
        std::vector<std::vector<double>> DNNinputs;     // vectors for the inference of DNN
        std::vector<DynamicBuffer<label>> cellIDBuffer; // Buffer contains the cell numbers
        std::vector<std::vector<label>> problemCounter; // evaluate the number of the problems of each subslave

        std::chrono::steady_clock::time_point start5 = std::chrono::steady_clock::now();
        getDNNinputs(problemBuffer, outputLength, DNNinputs, cellIDBuffer, problemCounter);
        std::chrono::steady_clock::time_point stop5 = std::chrono::steady_clock::now();
        std::chrono::duration<double> processingTime5 = std::chrono::duration_cast<std::chrono::duration<double>>(stop5 - start5);
        // std::cout << "getDNNinputsTime = " << processingTime5.count() << std::endl;
        time_getDNNinputs_ += processingTime5.count();

        /*=============================inference via pybind11=============================*/
        std::chrono::steady_clock::time_point start7 = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point start8 = std::chrono::steady_clock::now();

        pybind11::array_t<double> vec0 = pybind11::array_t<double>({DNNinputs[0].size()}, {8}, &DNNinputs[0][0]); // cast vector to np.array
        pybind11::array_t<double> vec1 = pybind11::array_t<double>({DNNinputs[1].size()}, {8}, &DNNinputs[1][0]);
        pybind11::array_t<double> vec2 = pybind11::array_t<double>({DNNinputs[2].size()}, {8}, &DNNinputs[2][0]);

        std::chrono::steady_clock::time_point stop8 = std::chrono::steady_clock::now();
        std::chrono::duration<double> processingTime8 = std::chrono::duration_cast<std::chrono::duration<double>>(stop8 - start8);
        // std::cout << "vec2ndarrayTime = " << processingTime8.count() << std::endl;
        time_vec2ndarray_ += processingTime8.count();

        pybind11::module_ call_torch = pybind11::module_::import("inference"); // import python file

        std::chrono::steady_clock::time_point start9 = std::chrono::steady_clock::now();

        pybind11::object result = call_torch.attr("inference")(vec0, vec1, vec2); // call python function
        const double* star = result.cast<pybind11::array_t<double>>().data();

        std::chrono::steady_clock::time_point stop9 = std::chrono::steady_clock::now();
        std::chrono::duration<double> processingTime9 = std::chrono::duration_cast<std::chrono::duration<double>>(stop9 - start9);
        // std::cout << "pythonTime = " << processingTime9.count() << std::endl;
        time_python_ += processingTime9.count();

        std::chrono::steady_clock::time_point stop7 = std::chrono::steady_clock::now();
        std::chrono::duration<double> processingTime7 = std::chrono::duration_cast<std::chrono::duration<double>>(stop7 - start7);
        // std::cout << "DNNinferenceTime = " << processingTime7.count() << std::endl;
        time_DNNinference_ += processingTime7.count();

        /*=============================construct solutions=============================*/
        std::chrono::steady_clock::time_point start6 = std::chrono::steady_clock::now();
        std::vector<double> outputsVec0(star, star+outputLength[0] * 7); //the float number is sample_length*sample_number
        std::vector<double> outputsVec1(star+outputLength[0] * 7, star+outputLength[1] * 7);
        std::vector<double> outputsVec2(star+outputLength[1] * 7, star+outputLength[2] * 7);
        std::vector<std::vector<double>> results = {outputsVec0, outputsVec1, outputsVec2};
        updateSolutionBuffer(solutionBuffer, results, cellIDBuffer, problemCounter);
        std::chrono::steady_clock::time_point stop6 = std::chrono::steady_clock::now();
        std::chrono::duration<double> processingTime6 = std::chrono::duration_cast<std::chrono::duration<double>>(stop6 - start6);
        // std::cout << "updateSolutionBufferTime = " << processingTime6.count() << std::endl;
        time_updateSolutionBuffer_ += processingTime6.count();

        std::chrono::steady_clock::time_point stop1 = std::chrono::steady_clock::now();
        std::chrono::duration<double> processingTime1 = std::chrono::duration_cast<std::chrono::duration<double>>(stop1 - start1);
        // std::cout << "submasterTime = " << processingTime1.count() << std::endl;
        time_submaster_ += processingTime1.count();
    }

    /*=============================calculates RR with CVODE use DLB=============================*/
    DynamicList<ChemistrySolution> CPUSolutionList;
    if (Pstream::myProcNo() % cores_) //for slave
    {
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        DynamicBuffer<ChemistrySolution> incomingSolutions;
        balancer_.updateState(CPUproblemList, cvodeComm);
        auto guestProblems = balancer_.balance(CPUproblemList, cvodeComm);
        auto ownProblems = balancer_.getRemaining(CPUproblemList, cvodeComm);
        auto ownSolutions = solveList(ownProblems);
        auto guestSolutions = solveBuffer(guestProblems);
        incomingSolutions = balancer_.unbalance(guestSolutions, cvodeComm);
        incomingSolutions.append(ownSolutions);
        updateReactionRates(incomingSolutions, CPUSolutionList);
        std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();
        std::chrono::duration<double> processingTime = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
        std::cout << "slaveTime = " << processingTime.count() << std::endl;
    }

    /*=============================send CPUSolutionList back to submaster=============================*/
    PstreamBuffers pBufs3(Pstream::commsTypes::nonBlocking);

    if ((Pstream::myProcNo() % cores_) == 1) // neighbour of submaster
    {
        UOPstream send((Pstream::myProcNo() - 1), pBufs3);
        send << CPUSolutionList;
    }
    pBufs3.finishedSends();
    if (!(Pstream::myProcNo() % cores_)) // submaster
    {
        UIPstream recv((Pstream::myProcNo() + 1), pBufs3);// resv CPUproblems from neighbour
        recv >> CPUSolutionList;
    }

    /*=============================send and recv GPUSolutions=============================*/
    std::chrono::steady_clock::time_point start4 = std::chrono::steady_clock::now();

    DynamicList<GpuSolution> finalList;
    PstreamBuffers pBufs2(Pstream::commsTypes::nonBlocking);
    if (!(Pstream::myProcNo() % cores_)) // submaster
    {
        finalList = solutionBuffer[0];
        for (label i = 1; i < cores_; i++)
        {
            UOPstream send(i + Pstream::myProcNo(), pBufs2);
            send << solutionBuffer[i];
        }
    }
    pBufs2.finishedSends();
    if (Pstream::myProcNo() % cores_) // slavers
    {
        UIPstream recv((Pstream::myProcNo()/cores_)*cores_, pBufs2);
        recv >> finalList;
    }

    std::chrono::steady_clock::time_point stop4 = std::chrono::steady_clock::now();
    std::chrono::duration<double> processingTime4 = std::chrono::duration_cast<std::chrono::duration<double>>(stop4 - start4);
    // std::cout << "SendRecvSolutionTime = " << processingTime4.count() << std::endl;
    time_sendRecvSolution_ += processingTime4.count();

    /*=============================update RR fields=============================*/
    for (int cellI = 0; cellI < finalList.size(); cellI++)
    {
        Qdot_[finalList[cellI].cellid] = 0;
        for (int speciID = 0; speciID < mixture_.nSpecies(); speciID++)
        {
            RR_[speciID][finalList[cellI].cellid] = finalList[cellI].RRi[speciID];
            Qdot_[finalList[cellI].cellid] -= hc_[speciID] * RR_[speciID][finalList[cellI].cellid];
        }
    }

    if (!(Pstream::myProcNo() % cores_)) // submaster
    {
        for (int cellI = 0; cellI < CPUSolutionList.size(); cellI++)
        {
            for (int speciID = 0; speciID < mixture_.nSpecies(); speciID++)
            {
                RR_[speciID][CPUSolutionList[cellI].cellid] = CPUSolutionList[cellI].RRi[speciID];
            }
            Qdot_[CPUSolutionList[cellI].cellid] = CPUSolutionList[cellI].Qdoti;
            cpuTimes_[CPUSolutionList[cellI].cellid] = CPUSolutionList[cellI].cpuTime;
        }
    }

    Info << "=== end solve_DNN === " << endl;

    std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();
    std::chrono::duration<double> processingTime = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
    // std::cout << "allSolveTime = " << processingTime.count() << std::endl;
    time_allsolve_ += processingTime.count();

    return deltaTMin;
}
#endif

// ************************************************************************* //
