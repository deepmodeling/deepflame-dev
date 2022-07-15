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
#include "loadBalancing/runtime_assert.H"


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
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh_,
        dimensionedScalar(dimEnergy/dimVolume/dimTime, 0)
    ),
    torchSwitch_(lookupOrDefault("torch", false)),
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
    if(torchSwitch_)
    {
        torchModelName_ = this->lookupOrDefault("torchModel", word(""));
        Xmu_ = scalarList(this->subDict("torchParameters").lookup("Xmu"));
        Xstd_ = scalarList(this->subDict("torchParameters").lookup("Xstd"));
        Ymu_ = scalarList(this->subDict("torchParameters").lookup("Ymu"));
        Ystd_ = scalarList(this->subDict("torchParameters").lookup("Ystd"));
        Tact_ = this->subDict("torchParameters").lookupOrDefault("Tact", 700);
        Qdotact_ = this->subDict("torchParameters").lookupOrDefault("Qdotact", 1e9);
    }

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
    if(torchSwitch_)
    {
        result = torchSolve(deltaT);
    }
    else
    {
        result = canteraSolve(deltaT);
    }
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
template<class DeltaTType>
Foam::scalar Foam::dfChemistryModel<ThermoType>::canteraSolve
(
    const DeltaTType& deltaT
)
{
    scalar deltaTMin = great;

    if (!this->chemistry_)
    {
        return deltaTMin;
    }

    Info<<"=== begin Cantera-solve === "<<endl;

    Cantera::Reactor react;
    //Cantera::IdealGasReactor react;  // Constant-UV, default, constant Volumn
    //Cantera::IdealGasConstPressureReactor react;  // Constant-HP, constant pressure

    scalarField c0(CanteraGas_->nSpecies());

    Qdot_ = Zero;

    forAll(T_, cellI)
    {
        scalar Ti = T_[cellI];
        scalar pi = p_[cellI];
        try
        {
            for (size_t i=0; i<CanteraGas_->nSpecies(); i++)
            {
                yTemp_[i] = Y_[i][cellI];
            }

            CanteraGas_->setState_TPY(Ti, pi, yTemp_.begin());
            CanteraGas_->getConcentrations(c0.begin()); // value --> c0

            react.insert(mixture_.CanteraSolution());
            react.setEnergy(0); // keep T const before and after sim.advance. this will give you a little improvement
            Cantera::ReactorNet sim;
            sim.addReactor(react);
            setNumerics(sim);


            sim.advance(deltaT[cellI]);


            CanteraGas_->getConcentrations(cTemp_.begin()); // value --> cTemp_

            for (size_t i=0; i<CanteraGas_->nSpecies(); i++)
            {
                RR_[i][cellI] = (cTemp_[i] - c0[i])*CanteraGas_->molecularWeight(i)/deltaT[cellI];
            }
            // CanteraGas_->molecularWeight(i)    kg/kmol

            forAll(Y_, i)
            {
                const scalar hc = CanteraGas_->Hf298SS(i)/CanteraGas_->molecularWeight(i); // J/kg
                Qdot_[cellI] -= hc*RR_[i][cellI];
            }
        }
        catch(Cantera::CanteraError& err)
        {
            // handle exceptions thrown by Cantera
            std::cout << err.what() << std::endl;

            FatalErrorIn("dfChemistryModel::solve")
                << " Cantera complained in cell " << cellI
                << " with a Cantera::CanteraError"  << endl
                << abort(FatalError) ;
        }
    }

    Info<<"=== end Cantera-solve === "<<endl;
    return deltaTMin;
}


template<class ThermoType>
void Foam::dfChemistryModel<ThermoType>::setNumerics(Cantera::ReactorNet &sim)
{
    sim.setTolerances(relTol_,absTol_);
}


template<class ThermoType>
template<class DeltaTType>
Foam::scalar Foam::dfChemistryModel<ThermoType>::torchSolve
(
    const DeltaTType& deltaT
)
{
    scalar deltaTMin = great;

    if (!this->chemistry_)
    {
        return deltaTMin;
    }

    Info<<"=== begin torch&ode-solve === "<<endl;

    // set variables
    scalarList yPre_(mixture_.nSpecies());
    scalarList yBCT_(mixture_.nSpecies());
    scalarList u_(mixture_.nSpecies()+2); //plus T and p
    Cantera::Reactor react;
    double lambda = 0.1;

    // load model
    torch::DeviceType device_type;
    torch::DeviceIndex index = -1;
    if (0)
    {
        device_type = torch::kCUDA;
        index = 0;
    }
    else
    {
        device_type = torch::kCPU;
        index = 0;
    }
    torch::Device device(device_type, index);
    if (!isFile(torchModelName_))
    {
        FatalErrorInFunction
            << torchModelName_
            << " doesn't exist!"
            << exit(FatalError);
    }
    torch::jit::script::Module torchModel_ = torch::jit::load(torchModelName_, device);

    std::vector<size_t> torch_cell;
    label torch_cellname= 0;

    // obtain the number of DNN cells
    forAll(T_, cellI)
    {
        if ((Qdot_[cellI] >= Qdotact_) && (T_[cellI] >= Tact_))
        {
            torch_cell.push_back(cellI);
        }
    }

    torch::Tensor inputs(torch::zeros({torch_cell.size(),mixture_.nSpecies()+2}));
    forAll(T_, cellI)
    {
        scalar Ti = T_[cellI];
        scalar pi = p_[cellI];
        scalar rhoi = rho_[cellI];

        if ((Qdot_[cellI] >= Qdotact_) && (Ti >= Tact_))
        {
            Qdot_[cellI] = 0.0;
            // Info<<"Now is DNN "<<endl;

            // set inputs
            std::vector<double> inputs_;
            inputs_.push_back((Ti - Xmu_[0])/Xstd_[0]);
            inputs_.push_back((pi / 101325 - Xmu_[1])/Xstd_[1]);
            for (size_t i=0; i<CanteraGas_->nSpecies(); i++)
            {
                yPre_[i] = Y_[i][cellI];
                yBCT_[i] = (pow(yPre_[i],lambda) - 1) / lambda; // function BCT
            }
            for (size_t i=0; i<CanteraGas_->nSpecies(); i++)
            {
                inputs_.push_back((yBCT_[i] - Xmu_[i+2]) / Xstd_[i+2]);
            }
            inputs[torch_cellname] = torch::tensor(inputs_);
            torch_cellname ++;
            // Info << "cell_name = "<< cellI <<endl;
        }
        else
        {
            Qdot_[cellI] = 0.0;
            for (size_t i=0; i<CanteraGas_->nSpecies(); i++)
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

            for (size_t i=0; i<CanteraGas_->nSpecies(); i++)
            {
                RR_[i][cellI] = (yTemp_[i] - yPre_[i])*rhoi/deltaT;
                const scalar hc = CanteraGas_->Hf298SS(i)/CanteraGas_->molecularWeight(i); // J/kg
                Qdot_[cellI] -= hc*RR_[i][cellI];
            }
        }
    }
    // DNN
    std::vector<torch::jit::IValue> INPUTS{inputs};
    auto outputs_raw = torchModel_.forward(INPUTS);
    auto outputs = outputs_raw.toTensor();

    for(size_t cellI = 0; cellI<torch_cell.size(); cellI ++)
    {
        // update y
        scalar Yt = 0;
        for (size_t i=0; i<CanteraGas_->nSpecies(); i++)
        {
            yPre_[i] = Y_[i][torch_cell[cellI]];
            yTemp_[i] = Y_[i][torch_cell[cellI]];
            yBCT_[i] = (pow(yPre_[i],lambda) - 1) / lambda; // function BCT
        }
        for (size_t i=0; i<(CanteraGas_->nSpecies()); i++)//
        {
            u_[i+2] = outputs[cellI][i+2].item().to<double>()*Ystd_[i+2]+Ymu_[i+2];
            yTemp_[i] = pow((yBCT_[i] + u_[i+2]*deltaT)*lambda+1,1/lambda);
            Yt += yTemp_[i];
        }
        for (size_t i=0; i<CanteraGas_->nSpecies(); i++)
        {
            yTemp_[i] = yTemp_[i] / Yt;
            RR_[i][torch_cell[cellI]] = (yTemp_[i] - Y_[i][torch_cell[cellI]])*rho_[torch_cell[cellI]]/deltaT;
            const scalar hc = CanteraGas_->Hf298SS(i)/CanteraGas_->molecularWeight(i); // J/kg
            Qdot_[torch_cell[cellI]] -= hc*RR_[i][torch_cell[cellI]];
        }
    }

    Info<<"=== end torch&ode-solve === "<<endl;
    return deltaTMin;
}

template<class ThermoType>
void Foam::dfChemistryModel<ThermoType>::correctThermo()
{
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
                        rhoD_[i].boundaryFieldRef()[patchi][facei] = dTemp_[i];

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
    // scalarList c0 = problem.c;

    // Timer begins
    clockTime time;
    time.timeIncrement();

    Cantera::Reactor react;
    const scalar Ti = problem.Ti;
    const scalar pi = problem.pi;
    const scalar rhoi = problem.rhoi;
    const scalarList yPre_ = problem.Y;

    CanteraGas_->setState_TPY(Ti, pi, yPre_.begin());

    react.insert(mixture_.CanteraSolution());
    react.setEnergy(0); // keep T const before and after sim.advance. this will give you a little improvement
    Cantera::ReactorNet sim;
    sim.addReactor(react);
    setNumerics(sim);

    sim.advance(problem.deltaT);

    // get new concentrations
    CanteraGas_->getMassFractions(yTemp_.begin());
    // problem.c = cTemp_;

    for (size_t i=0; i<CanteraGas_->nSpecies(); i++)
    {
        solution.RRi[i] = (yTemp_[i] - yPre_[i]) / problem.deltaT * rhoi;
    }

    // Timer ends
    solution.cpuTime = time.timeIncrement();

    solution.cellid = problem.cellid;
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
    const scalarField& rho = rho_;


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
            // problem.c = cTemp_;
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
    const RecvBuffer<ChemistrySolution>& solutions
)
{
    scalar deltaTMin = great;

    for(const auto& array : solutions)
    {
        for(const auto& solution : array)
        {

            for(label j = 0; j < mixture_.nSpecies(); j++)
            {
                this->RR_[j][solution.cellid] = solution.RRi[j];
            }

            cpuTimes_[solution.cellid] = solution.cpuTime;
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
Foam::scalar Foam::dfChemistryModel<ThermoType>::solve_loadBalance
(
    const DeltaTType& deltaT
)
{
    Info<<"=== begin DLB-solve === "<<endl;
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

    Info<<"=== end DLB-solve === "<<endl;
    return updateReactionRates(incomingSolutions);
}

// ************************************************************************* //
