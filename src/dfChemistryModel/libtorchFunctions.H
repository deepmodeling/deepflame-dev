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

    if (gpu_)
    {
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
    }
    else
    {
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

template <class ThermoType>
template <class DeltaTType>
Foam::scalar Foam::dfChemistryModel<ThermoType>::solve_DNN_new(
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
        DynamicBuffer<label> cellIDBuffer;
        DynamicList<ChemistryProblem> CPUproblemList;
        getGPUProblems_libtorch(deltaT, cellIDBuffer, CPUproblemList);

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
        /*============================prepare shared region to recieve inference result===========================*/
        int sizeResults0 = nNN0Dev * mixture_.nSpecies();
        int sizeResults1 = nNN1Dev * mixture_.nSpecies();
        int sizeResults2 = nNN2Dev * mixture_.nSpecies();
        double *sharedResult0 = nullptr;
        double *sharedResult1 = nullptr;
        double *sharedResult2 = nullptr;
        int disp_unit  = sizeof(double);
        MPI_Win win0, win1, win2;
        // allocate results
        MPI_Win_allocate_shared(devWorldRank==0 ? sizeResults0*disp_unit : 0, disp_unit, MPI_INFO_NULL, devWorld, &sharedResult0, &win0);
        MPI_Win_allocate_shared(devWorldRank==0 ? sizeResults1*disp_unit : 0, disp_unit, MPI_INFO_NULL, devWorld, &sharedResult1, &win1);
        MPI_Win_allocate_shared(devWorldRank==0 ? sizeResults2*disp_unit : 0, disp_unit, MPI_INFO_NULL, devWorld, &sharedResult2, &win2);
        if (devWorldRank)
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

        /*================================================Inference===============================================*/
        if (!devWorldRank) // submaster
        {
            // Inference
            std::vector<std::vector<double>> results;
            results = DNNInferencer_.Inference_multiDNNs_new(d_NN0, d_NN1, d_NN2, mixture_.nSpecies() + 3,
            nNN0Dev, nNN1Dev, nNN2Dev);
            double* test = new double[80];
            cudaMemcpy(test, d_NN0, 80 * sizeof(double),cudaMemcpyDeviceToHost);
            // printf("pointerResults0\n");
            // for (int i = 0; i < 80; i++)
            // {
            //     printf("%.5e ", *(test + i));
            // }
            // write data to shared block
            // MPI_Put(results[0].data(), sizeResults0, MPI_DOUBLE, 0, 0, sizeResults0, MPI_DOUBLE, win0);
            // MPI_Put(results[1].data(), sizeResults1, MPI_DOUBLE, 0, 0, sizeResults1, MPI_DOUBLE, win1);
            // MPI_Put(results[2].data(), sizeResults2, MPI_DOUBLE, 0, 0, sizeResults2, MPI_DOUBLE, win2);
            std::copy(results[0].begin(), results[0].end(), sharedResult0);
            std::copy(results[1].begin(), results[1].end(), sharedResult1);
            std::copy(results[2].begin(), results[2].end(), sharedResult2);
        }
        /*====================================calculates RR with CVODE use DLB====================================*/
        DynamicList<ChemistrySolution> CPUSolutionList;
        if (devWorldRank) //for slave
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
        // MPI_Get(localResults[0], nNN0Local*scale, MPI_DOUBLE, 0, nNN0OutputDispls[devWorldRank], nNN0Local*scale, MPI_DOUBLE, win0);
        // MPI_Get(localResults[1], nNN1Local*scale, MPI_DOUBLE, 0, nNN1OutputDispls[devWorldRank], nNN1Local*scale, MPI_DOUBLE, win1);
        // MPI_Get(localResults[2], nNN2Local*scale, MPI_DOUBLE, 0, nNN2OutputDispls[devWorldRank], nNN2Local*scale, MPI_DOUBLE, win2);
        std::copy(sharedResult0+nNN0OutputDispls[devWorldRank], sharedResult0+nNN0OutputDispls[devWorldRank]+nNN0Local*mixture_.nSpecies(), localResults0);
        std::copy(sharedResult1+nNN1OutputDispls[devWorldRank], sharedResult1+nNN1OutputDispls[devWorldRank]+nNN1Local*mixture_.nSpecies(), localResults1);
        std::copy(sharedResult2+nNN2OutputDispls[devWorldRank], sharedResult2+nNN2OutputDispls[devWorldRank]+nNN2Local*mixture_.nSpecies(), localResults2);
        
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
        // free memory
        delete[] localResults0;
        delete[] localResults1;
        delete[] localResults2;
    }
    else
    {
        // cores_ = 1;
        // // solve CPU problem with cvode
        // DynamicBuffer<ChemistrySolution> incomingSolutions;
        // DynamicList<ChemistrySolution> CPUSolutionList;
        // balancer_.updateState(CPUproblemList, cvodeComm);
        // auto guestProblems = balancer_.balance(CPUproblemList, cvodeComm);
        // auto ownProblems = balancer_.getRemaining(CPUproblemList, cvodeComm);
        // auto ownSolutions = solveList(ownProblems);
        // auto guestSolutions = solveBuffer(guestProblems);
        // incomingSolutions = balancer_.unbalance(guestSolutions, cvodeComm);
        // incomingSolutions.append(ownSolutions);
        // updateReactionRates(incomingSolutions, CPUSolutionList);

        // // solve other problems with NN
        // DynamicBuffer<GpuProblem> problemBuffer;
        // DynamicBuffer<GpuSolution> solutionBuffer;
        // std::vector<label> outputLength;
        // std::vector<std::vector<double>> DNNinputs;     // tensors for the inference of DNN
        // std::vector<DynamicBuffer<label>> cellIDBuffer; // Buffer contains the cell numbers
        // std::vector<std::vector<label>> problemCounter; // evaluate the number of the problems of each subslave
        // problemBuffer.append(GPUproblemList);
        // getDNNinputs(problemBuffer, outputLength, DNNinputs, cellIDBuffer, problemCounter);
        // auto results = DNNInferencer_.Inference_multiDNNs(DNNinputs, mixture_.nSpecies() + 3);
        // updateSolutionBuffer(solutionBuffer, results, cellIDBuffer, problemCounter);
        // DynamicList<GpuSolution> finalList;
        // finalList = solutionBuffer[0];
        // for (int cellI = 0; cellI < finalList.size(); cellI++)
        // {
        //     Qdot_[finalList[cellI].cellid] = 0;
        //     for (int speciID = 0; speciID < mixture_.nSpecies(); speciID++)
        //     {
        //         RR_[speciID][finalList[cellI].cellid] = finalList[cellI].RRi[speciID];
        //         Qdot_[finalList[cellI].cellid] -= hc_[speciID] * RR_[speciID][finalList[cellI].cellid];
        //     }
        // }
    }

    Info << "=== end solve_DNN === " << endl;

    std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();
    std::chrono::duration<double> processingTime = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
    // std::cout << "allSolveTime = " << processingTime.count() << std::endl;
    time_allsolve_ += processingTime.count();

    return deltaTMin;
}