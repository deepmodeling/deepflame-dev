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
    CanteraKinetics_(mixture_.CanteraKinetics()),
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
    // rho_(mesh_.objectRegistry::lookupObject<volScalarField>("rho")),
    rho_(const_cast<volScalarField&>(dynamic_cast<rhoThermo&>(thermo).rho())),
    // mu_(const_cast<volScalarField&>(dynamic_cast<psiThermo&>(thermo).mu()())),
    // psi_(const_cast<volScalarField&>(dynamic_cast<psiThermo&>(thermo).psi())),
    mu_(const_cast<volScalarField&>(dynamic_cast<rhoThermo&>(thermo).mu()())),
    psi_(const_cast<volScalarField&>(dynamic_cast<rhoThermo&>(thermo).psi())),
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
    torchModelName1_ = this->subDict("TorchSettings").lookupOrDefault("torchModel1", word(""));
    torchModelName2_ = this->subDict("TorchSettings").lookupOrDefault("torchModel2", word(""));
    torchModelName3_ = this->subDict("TorchSettings").lookupOrDefault("torchModel3", word(""));

    // set the number of cores slaved by each GPU card
    cores_ = this->subDict("TorchSettings").lookupOrDefault("coresPerGPU", 8);
    GPUsPerNode_ = this->subDict("TorchSettings").lookupOrDefault("GPUsPerNode", 4);

    // initialization the Inferencer (if use multi GPU)
    if(torchSwitch_)
    {
        if (gpu_)
        {
            if(!(Pstream::myProcNo() % cores_)) // Now is a master
            {
                torch::jit::script::Module torchModel1_ = torch::jit::load(torchModelName1_);
                torch::jit::script::Module torchModel2_ = torch::jit::load(torchModelName2_);
                torch::jit::script::Module torchModel3_ = torch::jit::load(torchModelName3_);
                std::string device_;
                int CUDANo = (Pstream::myProcNo() / cores_) % GPUsPerNode_;
                device_ = "cuda:" + std::to_string(CUDANo);
                DNNInferencer DNNInferencer(torchModel1_, torchModel2_, torchModel3_, device_);
                DNNInferencer_ = DNNInferencer;
            }
        }
        else
        {
            torch::jit::script::Module torchModel1_ = torch::jit::load(torchModelName1_);
            torch::jit::script::Module torchModel2_ = torch::jit::load(torchModelName2_);
            torch::jit::script::Module torchModel3_ = torch::jit::load(torchModelName3_);
            std::string device_;
            device_ = "cpu";
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
        for (int rank = 0; rank < Pstream::nProcs(); rank ++)
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

#if defined USE_LIBTORCH || defined USE_PYTORCH
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
    try
    {
        psi_.oldTime();

        forAll(T_, celli)
        {
            forAll(Y_, i)
            {
                yTemp_[i] = Y_[i][celli];
            }
            CanteraGas_->setState_PY(p_[celli], yTemp_.begin());
            if(mixture_.heName()=="ha")
            {
                CanteraGas_->setState_HP(thermo_.he()[celli], p_[celli]); // setState_HP needs (J/kg)
            }
            else if(mixture_.heName()=="ea")
            {
                scalar ha = thermo_.he()[celli] + p_[celli]/rho_[celli];
                CanteraGas_->setState_HP(ha, p_[celli]);
            }


            T_[celli] = CanteraGas_->temperature();

            psi_[celli] = mixture_.psi(p_[celli],T_[celli]);

            rho_[celli] = mixture_.rho(p_[celli],T_[celli]);

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

        volScalarField::Boundary& rhoBf = rho_.boundaryFieldRef();

        volScalarField::Boundary& TBf = T_.boundaryFieldRef();

        volScalarField::Boundary& psiBf = psi_.boundaryFieldRef();

        volScalarField::Boundary& hBf = thermo_.he().boundaryFieldRef();

        volScalarField::Boundary& muBf = mu_.boundaryFieldRef();

        volScalarField::Boundary& alphaBf = alpha_.boundaryFieldRef();

        forAll(T_.boundaryField(), patchi)
        {
            const fvPatchScalarField& pp = pBf[patchi];
            fvPatchScalarField& prho = rhoBf[patchi];
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

                    ppsi[facei] = mixture_.psi(pp[facei],pT[facei]);

                    prho[facei] = mixture_.rho(pp[facei],pT[facei]);

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
                    if(mixture_.heName()=="ha")
                    {
                        CanteraGas_->setState_HP(ph[facei], pp[facei]);
                    }
                    else if(mixture_.heName()=="ea")
                    {
                        scalar ha = ph[facei] + pp[facei]/prho[facei];
                        CanteraGas_->setState_HP(ha, pp[facei]);
                    }

                    pT[facei] = CanteraGas_->temperature();

                    ppsi[facei] = mixture_.psi(pp[facei],pT[facei]);

                    prho[facei] = mixture_.rho(pp[facei],pT[facei]);

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
    catch(Cantera::CanteraError& err)
    {
        std::cerr << err.what() << '\n';
        FatalErrorInFunction
            << abort(FatalError);
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
    // keep T const before and after sim.advance. this will give you a little improvement
    react.setEnergy(0);
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
                    RR_[j][solution.cellid] = solution.RRi[j];
                }
                Qdot_[solution.cellid] = solution.Qdoti;

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

template<class ThermoType>
Foam::tmp<Foam::DimensionedField<Foam::scalar, Foam::volMesh>>
Foam::dfChemistryModel<ThermoType>::calculateRR
(
    const label reactionI,
    const label speciei
) const
{
    tmp<volScalarField::Internal> tRR
    (
        volScalarField::Internal::New
        (
           "RR",
           mesh_,
           dimensionedScalar(dimMass/dimVolume/dimTime, 0)
        )
    );
  
    volScalarField::Internal& RR = tRR.ref();
	
    doublereal netRate[mixture_.nReactions()];
    doublereal X[mixture_.nSpecies()];

    forAll(rho_, celli)
    {
        const scalar rhoi = rho_[celli];
        const scalar Ti = T_[celli];
        const scalar pi = p_[celli];

        for (label i=0; i<mixture_.nSpecies(); i++)
        {
            const scalar Yi = Y_[i][celli];

            X[i] = rhoi*Yi/CanteraGas_->molecularWeight(i);
        }

	CanteraGas_->setState_TPX(Ti, pi, X);

	CanteraKinetics_->getNetRatesOfProgress(netRate);

	auto R = CanteraKinetics_->reaction(reactionI);

	for (const auto& sp : R->reactants)
        {
		if (speciei == static_cast<int>(CanteraGas_->speciesIndex(sp.first)))
		{
			RR[celli] -= sp.second*netRate[reactionI];
		}
			
	}			
	for (const auto& sp : R->products)
	{
		if (speciei == static_cast<int>(CanteraGas_->speciesIndex(sp.first)))
		{
			RR[celli] += sp.second*netRate[reactionI];
		}
	}					
		
        RR[celli] *= CanteraGas_->molecularWeight(speciei);
    }

    return tRR;
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

    if(!chemistry_)
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
#include "torchFunctions.H"
#endif


#ifdef USE_LIBTORCH
#include "libtorchFunctions.H"
#endif


#ifdef USE_PYTORCH
#include "pytorchFunctions.H"
#endif

// ************************************************************************* //
