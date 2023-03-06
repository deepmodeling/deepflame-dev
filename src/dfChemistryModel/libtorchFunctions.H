template <class ThermoType>
template<class DeltaTType>
void Foam::dfChemistryModel<ThermoType>::constructNNInput
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

    if (gpu_)
    {
        /*=============================gather NN inputs for GPU=============================*/
        std::chrono::steady_clock::time_point start10 = std::chrono::steady_clock::now();
        DynamicBuffer<label> cellIDBuffer;
        DynamicList<ChemistryProblem> CPUproblemList;
        constructNNInput(deltaT, cellIDBuffer, CPUproblemList);
        label flag_mpi_init;
        MPI_Initialized(&flag_mpi_init);
        if(flag_mpi_init) MPI_Barrier(PstreamGlobals::MPI_COMM_FOAM);
        std::chrono::steady_clock::time_point stop10 = std::chrono::steady_clock::now();
        std::chrono::duration<double> processingTime10 = std::chrono::duration_cast<std::chrono::duration<double>>(stop10 - start10);
        time_getProblems_ += processingTime10.count();

        /*==============================send CVODE problems from submaster to neighbour==============================*/
        PstreamBuffers pBufs1(Pstream::commsTypes::nonBlocking);
        if (!devWorldRank) // submaster
        {
            UOPstream send((Pstream::myProcNo() + 1), pBufs1);// sending CPUproblems to neighbour
            send << CPUproblemList;
        }
        pBufs1.finishedSends();
        if (devWorldRank == 1) // neighbour of submaster
        {
            DynamicList<ChemistryProblem> CPUproblemList_submaster;
            UIPstream recv((Pstream::myProcNo() - 1), pBufs1);
            recv >> CPUproblemList_submaster;
            CPUproblemList.append(CPUproblemList_submaster);
        }
        /*============================allocate shared memory to recieve inference result===========================*/
        std::chrono::steady_clock::time_point start3 = std::chrono::steady_clock::now();
        int sizeResults0 = nNN0Dev * mixture_.nSpecies();
        int sizeResults1 = nNN1Dev * mixture_.nSpecies();
        int sizeResults2 = nNN2Dev * mixture_.nSpecies();
        double *sharedResult0, *sharedResult1, *sharedResult2;
        int disp_unit  = sizeof(double);
        MPI_Win win0, win1, win2;
        // allocate results
        MPI_Win_allocate_shared(devWorldRank==0 ? sizeResults0*disp_unit : 0, disp_unit, MPI_INFO_NULL, devWorld, &sharedResult0, &win0);
        MPI_Win_allocate_shared(devWorldRank==0 ? sizeResults1*disp_unit : 0, disp_unit, MPI_INFO_NULL, devWorld, &sharedResult1, &win1);
        MPI_Win_allocate_shared(devWorldRank==0 ? sizeResults2*disp_unit : 0, disp_unit, MPI_INFO_NULL, devWorld, &sharedResult2, &win2);
        if (devWorldRank) // slavers
        {
            MPI_Aint winsize;
            int windisp;
            MPI_Win_shared_query(win0, 0, &winsize, &windisp, &sharedResult0);
            MPI_Win_shared_query(win1, 0, &winsize, &windisp, &sharedResult1);
            MPI_Win_shared_query(win2, 0, &winsize, &windisp, &sharedResult2);
        }
        MPI_Win_fence(0, win0);
        MPI_Win_fence(0, win1);
        MPI_Win_fence(0, win2);
        std::chrono::steady_clock::time_point stop3 = std::chrono::steady_clock::now();
        std::chrono::duration<double> processingTime3 = std::chrono::duration_cast<std::chrono::duration<double>>(stop3 - start3);
        // std::cout << "RecvProblemTime = " << processingTime3.count() << std::endl;
        time_RecvProblem_ += processingTime3.count();

        /*================================================Inference===============================================*/
        if (!devWorldRank) // submaster
        {
            // -Inference
            std::chrono::steady_clock::time_point start7 = std::chrono::steady_clock::now();
            std::vector<std::vector<double>> results;
            results = DNNInferencer_.Inference_multiDNNs_new(d_NN0, d_NN1, d_NN2, mixture_.nSpecies() + 3,
            nNN0Dev, nNN1Dev, nNN2Dev);
            std::chrono::steady_clock::time_point stop7 = std::chrono::steady_clock::now();
            std::chrono::duration<double> processingTime7 = std::chrono::duration_cast<std::chrono::duration<double>>(stop7 - start7);
            time_DNNinference_ += processingTime7.count();
            // -Copy results to shared region
            std::chrono::steady_clock::time_point start2 = std::chrono::steady_clock::now();
            std::copy(results[0].begin(), results[0].end(), sharedResult0);
            std::copy(results[1].begin(), results[1].end(), sharedResult1);
            std::copy(results[2].begin(), results[2].end(), sharedResult2);
            std::chrono::steady_clock::time_point stop2 = std::chrono::steady_clock::now();
            std::chrono::duration<double> processingTime2 = std::chrono::duration_cast<std::chrono::duration<double>>(stop2 - start2);
            time_sendProblem_ += processingTime2.count();
        }
        /*====================================calculates RR with CVODE use DLB====================================*/
        DynamicList<ChemistrySolution> CPUSolutionList;
        if (devWorldRank) // slave
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
        }

        /*====================================construct and assign NN solutions===================================*/
        // allocate pointer to local results
        MPI_Barrier(devWorld);
        std::chrono::steady_clock::time_point start4 = std::chrono::steady_clock::now();
        double * localResults0 = new double[nNN0Local*mixture_.nSpecies()]; // remember to destory
        double * localResults1 = new double[nNN1Local*mixture_.nSpecies()];
        double * localResults2 = new double[nNN2Local*mixture_.nSpecies()];
        std::vector<double*> localResults = {localResults0, localResults1, localResults2};

        // wait for master to finish writing
        MPI_Win_fence(0, win0);
        MPI_Win_fence(0, win1);
        MPI_Win_fence(0, win2);

        // processes read data from shared memory simultaneously
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
        std::copy(sharedResult0+nNN0OutputDispls[devWorldRank], sharedResult0+nNN0OutputDispls[devWorldRank]+nNN0Local*mixture_.nSpecies(), localResults0);
        std::copy(sharedResult1+nNN1OutputDispls[devWorldRank], sharedResult1+nNN1OutputDispls[devWorldRank]+nNN1Local*mixture_.nSpecies(), localResults1);
        std::copy(sharedResult2+nNN2OutputDispls[devWorldRank], sharedResult2+nNN2OutputDispls[devWorldRank]+nNN2Local*mixture_.nSpecies(), localResults2);
        std::chrono::steady_clock::time_point stop4 = std::chrono::steady_clock::now();
        std::chrono::duration<double> processingTime4 = std::chrono::duration_cast<std::chrono::duration<double>>(stop4 - start4);
        time_sendRecvSolution_ += processingTime4.count();

        /*==================================send CPUSolutionList back to submaster================================*/
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

        /*=============================update RR & Qdot fields=============================*/
        int offset;
        for (int NNID = 0; NNID < 3; NNID++)
        {
            offset = 0;
            for (int cellI = 0; cellI < cellIDBuffer[NNID].size(); cellI++)
            {
                Qdot_[cellIDBuffer[NNID][cellI]] = 0;
                for (int speciID = 0; speciID < mixture_.nSpecies(); speciID++)
                {
                    RR_[speciID][cellIDBuffer[NNID][cellI]] = localResults[NNID][offset];
                    Qdot_[cellIDBuffer[NNID][cellI]] -= hc_[speciID] * RR_[speciID][cellIDBuffer[NNID][cellI]];
                    offset ++;
                }
            }
        }

        if (!devWorldRank) // submaster
        {
            std::chrono::steady_clock::time_point start6 = std::chrono::steady_clock::now();
            for (int cellI = 0; cellI < CPUSolutionList.size(); cellI++)
            {
                for (int speciID = 0; speciID < mixture_.nSpecies(); speciID++)
                {
                    RR_[speciID][CPUSolutionList[cellI].cellid] = CPUSolutionList[cellI].RRi[speciID];
                }
                Qdot_[CPUSolutionList[cellI].cellid] = CPUSolutionList[cellI].Qdoti;
                cpuTimes_[CPUSolutionList[cellI].cellid] = CPUSolutionList[cellI].cpuTime;
            }
            std::chrono::steady_clock::time_point stop6 = std::chrono::steady_clock::now();
            std::chrono::duration<double> processingTime6 = std::chrono::duration_cast<std::chrono::duration<double>>(stop6 - start6);
            // std::cout << "updateSolutionBufferTime = " << processingTime6.count() << std::endl;
            time_updateSolutionBuffer_ += processingTime6.count();
        }
        // free memory
        delete[] localResults0;
        delete[] localResults1;
        delete[] localResults2;
        MPI_Win_free(&win0);
        MPI_Win_free(&win1);
        MPI_Win_free(&win2);
    }
    else
    {
        DynamicList<GpuProblem> GPUproblemList; //single core TODO:rename it
        DynamicList<ChemistryProblem> CPUproblemList;
        getGPUProblems(deltaT, GPUproblemList, CPUproblemList);
        cores_ = 1;
        // solve CPU problem with cvode
        DynamicBuffer<ChemistrySolution> incomingSolutions;
        DynamicList<ChemistrySolution> CPUSolutionList;
        balancer_.updateState(CPUproblemList, cvodeComm);
        auto guestProblems = balancer_.balance(CPUproblemList, cvodeComm);
        auto ownProblems = balancer_.getRemaining(CPUproblemList, cvodeComm);
        auto ownSolutions = solveList(ownProblems);
        auto guestSolutions = solveBuffer(guestProblems);
        incomingSolutions = balancer_.unbalance(guestSolutions, cvodeComm);
        incomingSolutions.append(ownSolutions);
        updateReactionRates(incomingSolutions, CPUSolutionList);

        // solve other problems with NN
        DynamicBuffer<GpuProblem> problemBuffer;
        DynamicBuffer<GpuSolution> solutionBuffer;
        std::vector<label> outputLength;
        std::vector<std::vector<double>> DNNinputs;     // tensors for the inference of DNN
        std::vector<DynamicBuffer<label>> cellIDBuffer; // Buffer contains the cell numbers
        std::vector<std::vector<label>> problemCounter; // evaluate the number of the problems of each subslave
        problemBuffer.append(GPUproblemList);
        getDNNinputs(problemBuffer, outputLength, DNNinputs, cellIDBuffer, problemCounter);
        auto results = DNNInferencer_.Inference_multiDNNs(DNNinputs, mixture_.nSpecies() + 3);
        updateSolutionBuffer(solutionBuffer, results, cellIDBuffer, problemCounter);
        DynamicList<GpuSolution> finalList;
        finalList = solutionBuffer[0];
        for (int cellI = 0; cellI < finalList.size(); cellI++)
        {
            Qdot_[finalList[cellI].cellid] = 0;
            for (int speciID = 0; speciID < mixture_.nSpecies(); speciID++)
            {
                RR_[speciID][finalList[cellI].cellid] = finalList[cellI].RRi[speciID];
                Qdot_[finalList[cellI].cellid] -= hc_[speciID] * RR_[speciID][finalList[cellI].cellid];
            }
        }
    }

    Info << "=== end solve_DNN === " << endl;

    std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();
    std::chrono::duration<double> processingTime = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
    // std::cout << "allSolveTime = " << processingTime.count() << std::endl;
    time_allsolve_ += processingTime.count();

    return deltaTMin;
}