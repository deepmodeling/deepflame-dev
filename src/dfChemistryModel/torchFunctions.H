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

        problem.DNNid = 0;
        GPUproblemList.append(problem);
        selectDNN_[cellI]=0;
        continue;

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

    //if (gpulog_)
    //{
    //    std::cout<<"inputsDNN0 = "<<inputsDNN0.size()/(mixture_.nSpecies()+3) << "\n";
    //    std::cout<<"inputsDNN1 = "<<inputsDNN1.size()/(mixture_.nSpecies()+3) << "\n";
    //    std::cout<<"inputsDNN2 = "<<inputsDNN2.size()/(mixture_.nSpecies()+3) << "\n";
    //}

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
