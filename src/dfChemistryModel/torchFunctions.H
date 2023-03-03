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

template <class ThermoType>
template<class DeltaTType>
void Foam::dfChemistryModel<ThermoType>::getGPUProblems_libtorch
(
    const DeltaTType &deltaT, DynamicBuffer<label>& cellIDBuffer,
    Foam::DynamicList<ChemistryProblem>& CPUproblemList
)
{
    DynamicList<GpuProblem> problemList; //single core TODO:rename it
    std::vector<double> NN0List, NN1List, NN2List;
    DynamicList<label> cellIDList0;
    DynamicList<label> cellIDList1;
    DynamicList<label> cellIDList2;
    nNN0Local = 0;
    nNN1Local = 0;
    nNN2Local = 0;

    // get cuda problemList, for all cell
    // each get problem
    forAll(T_, cellI)
    {
        double Ti = T_[cellI];
        double pi = p_[cellI]/101325;
        double rhoi = rho_[cellI];

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
        ChemistryProblem ode_problem(mixture_.nSpecies());

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
            NN0List.push_back(Ti);
            NN0List.push_back(pi);
            for (int i = 0; i < mixture_.nSpecies(); i++)
            {
                NN0List.push_back(Y_[i][cellI]);
            }
            NN0List.push_back(rhoi);
            cellIDList0.append(cellI);
            nNN0Local ++;
            
            continue;
        }
        if(((Qdot_[cellI] >= 3e7) && (T_[cellI] < 2000)&&(T_[cellI] >= 700))||((Qdot_[cellI] > 7e8) && T_[cellI] > 2000)) //choose2
        {
            NN1List.push_back(Ti);
            NN1List.push_back(pi);
            for (int i = 0; i < mixture_.nSpecies(); i++)
            {
                NN1List.push_back(Y_[i][cellI]);
            }
            NN1List.push_back(rhoi);
            cellIDList1.append(cellI);
            nNN1Local ++;

            continue;
        }
        if  ((Qdot_[cellI] < 7e8) && (T_[cellI] >= 2000) && (Qdot_[cellI]!=0)) //choose3
        {
            NN2List.push_back(Ti);
            NN2List.push_back(pi);
            for (int i = 0; i < mixture_.nSpecies(); i++)
            {
                NN2List.push_back(Y_[i][cellI]);
            }
            NN2List.push_back(rhoi);
            cellIDList2.append(cellI);
            nNN2Local ++;

            continue;
        }
    }
    MPI_Request reqs0[3] = { MPI_REQUEST_NULL};
    MPI_Iallgather(&nNN0Local, 1, MPI_INT, nNN0InDevWorld.data(), 1, MPI_INT, devWorld, &reqs0[0]);
    MPI_Iallgather(&nNN1Local, 1, MPI_INT, nNN1InDevWorld.data(), 1, MPI_INT, devWorld, &reqs0[1]);
    MPI_Iallgather(&nNN2Local, 1, MPI_INT, nNN2InDevWorld.data(), 1, MPI_INT, devWorld, &reqs0[2]);

    // fill buffers
    cellIDBuffer.append(cellIDList0);
    cellIDBuffer.append(cellIDList1);
    cellIDBuffer.append(cellIDList2);

    // calculate NN number in devWorld
    MPI_Waitall(3, reqs0, MPI_STATUSES_IGNORE);

    nNN0Dev = std::accumulate(nNN0InDevWorld.begin(), nNN0InDevWorld.end(), 0);
    nNN1Dev = std::accumulate(nNN1InDevWorld.begin(), nNN1InDevWorld.end(), 0);
    nNN2Dev = std::accumulate(nNN2InDevWorld.begin(), nNN2InDevWorld.end(), 0);

    // Calculate displacements
    nNN0InputDispls.resize(devWorldSize + 1, 0);
    nNN1InputDispls.resize(devWorldSize + 1, 0);
    nNN2InputDispls.resize(devWorldSize + 1, 0);
    std::partial_sum(nNN0InDevWorld.begin(), nNN0InDevWorld.end(), nNN0InputDispls.begin() + 1);
    std::partial_sum(nNN1InDevWorld.begin(), nNN1InDevWorld.end(), nNN1InputDispls.begin() + 1);
    std::partial_sum(nNN2InDevWorld.begin(), nNN2InDevWorld.end(), nNN2InputDispls.begin() + 1);
    int scale = mixture_.nSpecies() + 3;
    std::transform(nNN0InputDispls.begin(), nNN0InputDispls.end(), nNN0InputDispls.begin(), [=](int x) { return x * scale; });
    std::transform(nNN1InputDispls.begin(), nNN1InputDispls.end(), nNN1InputDispls.begin(), [=](int x) { return x * scale; });
    std::transform(nNN2InputDispls.begin(), nNN2InputDispls.end(), nNN2InputDispls.begin(), [=](int x) { return x * scale; });
    
    // fill NN's input on GPU
    CHECK(cudaMemcpy(d_NN0 + nNN0InputDispls[devWorldRank], NN0List.data(), NN0List.size() * sizeof(scalar), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_NN1 + nNN1InputDispls[devWorldRank], NN1List.data(), NN1List.size() * sizeof(scalar), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_NN2 + nNN2InputDispls[devWorldRank], NN2List.data(), NN2List.size() * sizeof(scalar), cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());

    return;
}

template <class ThermoType>
void Foam::dfChemistryModel<ThermoType>::assignResults
(
    const std::vector<std::vector<double>>& results,
    std::vector<double*>& localResults
)
{
    double *sharedResult0 = nullptr;
    double *sharedResult1 = nullptr;
    double *sharedResult2 = nullptr;
    int disp_unit  = sizeof(double);
    int size0 = results[0].size();
    int size1 = results[1].size();
    int size2 = results[2].size();

    MPI_Win win0, win1, win2;

    // allocate results
    MPI_Win_allocate_shared(size0, disp_unit, MPI_INFO_NULL, devWorld, &sharedResult0, &win0);
    MPI_Win_allocate_shared(size1, disp_unit, MPI_INFO_NULL, devWorld, &sharedResult1, &win1);
    MPI_Win_allocate_shared(size2, disp_unit, MPI_INFO_NULL, devWorld, &sharedResult2, &win2);

    // subMaster write data to shared Memory
    // - lock the window for exclusive access
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win0);
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win1);
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win2);

    // - write data
    if (!devWorldRank) // now is master
    {
        MPI_Put(results[0].data(), size0, MPI_DOUBLE, 0, 0, size0, MPI_DOUBLE, win0);
        MPI_Put(results[1].data(), size1, MPI_DOUBLE, 0, 0, size1, MPI_DOUBLE, win1);
        MPI_Put(results[2].data(), size2, MPI_DOUBLE, 0, 0, size2, MPI_DOUBLE, win2);
    }
    
    // - wait for master to finish writing
    MPI_Win_sync(win0);
    MPI_Win_sync(win1);
    MPI_Win_sync(win2);

    // - release the lock on the window
    MPI_Win_unlock(0, win0);
    MPI_Win_unlock(0, win1);
    MPI_Win_unlock(0, win2);

    // other processes read data from shared memory simultaneously
    // - lock the window for shared access
    MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win0);
    MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win1);
    MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win2);

    // - calculate the data offset for each MPI rank
    nNN0OutputDispls.resize(devWorldSize + 1, 0);
    nNN1OutputDispls.resize(devWorldSize + 1, 0);
    nNN2OutputDispls.resize(devWorldSize + 1, 0);
    std::partial_sum(nNN0InDevWorld.begin(), nNN0InDevWorld.end(), nNN0OutputDispls.begin() + 1);
    std::partial_sum(nNN1InDevWorld.begin(), nNN1InDevWorld.end(), nNN1OutputDispls.begin() + 1);
    std::partial_sum(nNN2InDevWorld.begin(), nNN2InDevWorld.end(), nNN2OutputDispls.begin() + 1);
    int scale = mixture_.nSpecies();
    std::transform(nNN0OutputDispls.begin(), nNN0OutputDispls.end(), nNN0OutputDispls.begin(), [=](int x) {return x * scale;});
    std::transform(nNN1OutputDispls.begin(), nNN1OutputDispls.end(), nNN1OutputDispls.begin(), [=](int x) {return x * scale;});
    std::transform(nNN2OutputDispls.begin(), nNN2OutputDispls.end(), nNN2OutputDispls.begin(), [=](int x) {return x * scale;});

    // - get results from shared memory
    MPI_Get(localResults[0], nNN0Local*scale, MPI_DOUBLE, 0, nNN0OutputDispls[devWorldRank], nNN0Local*scale, MPI_DOUBLE, win0);
    MPI_Get(localResults[1], nNN1Local*scale, MPI_DOUBLE, 0, nNN1OutputDispls[devWorldRank], nNN1Local*scale, MPI_DOUBLE, win1);
    MPI_Get(localResults[2], nNN2Local*scale, MPI_DOUBLE, 0, nNN2OutputDispls[devWorldRank], nNN2Local*scale, MPI_DOUBLE, win2);

    return;
}