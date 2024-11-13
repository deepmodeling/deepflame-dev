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