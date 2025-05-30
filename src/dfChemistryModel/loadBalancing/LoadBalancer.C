/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | DLBFoam: Dynamic Load Balancing 
   \\    /   O peration     | for fast reactive simulations
    \\  /    A nd           | 
     \\/     M anipulation  | 2020, Aalto University, Finland 
-------------------------------------------------------------------------------
License
    This file is part of DLBFoam library, derived from OpenFOAM.
    https://github.com/blttkgl/DLBFoam

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

#include "LoadBalancer.H"

void
Foam::LoadBalancer::updateState(
    const DynamicList<ChemistryProblem>& problems, const label comm)
{
    if (Pstream::myProcNo(comm) == -1) return;
    auto myLoad = computeLoad(problems, comm);
    auto allLoads = allGather(myLoad, comm);
    std::vector<Foam::LoadBalancer::Operation> operations;
    if (algorithm_ == "allAverage")
    {
        operations = getOperations(allLoads, myLoad);
        if (log_)
        {
            Info << "now perform load balance with allAverage (DLB) algorithm" << endl;
        }
    }
    else 
    {
        operations = getOperationsRedezVous(allLoads, myLoad);
        if (log_)
        {
            Info << "now perform load balance with headTail (RedezVous) algorithm" << endl;
        }
    }
    auto info = operationsToInfo(operations, problems, myLoad);

    setState(info);
}

Foam::LoadBalancerBase::BalancerState
Foam::LoadBalancer::operationsToInfo(
    const std::vector<Operation>&        operations,
    const DynamicList<ChemistryProblem>& problems,
    const ChemistryLoad&                 myLoad)
{
    BalancerState info;

    if(isSender(operations, myLoad.rank))
    {
        double sum = 0.0;
        std::vector<double> times;
        for(const auto& op : operations)
        {
            info.destinations.push_back(op.to);
            sum += op.value;
            times.push_back(op.value);
        }
        info.nProblems = timesToProblemCounts(times, problems);

        label total = std::accumulate(info.nProblems.begin(), info.nProblems.end(), 0); // counts of problems sent elsewhere
        info.nRemaining = problems.size() - total;
    }

    // receiver
    else
    {
        for(const auto& op : operations)
        {
            info.sources.push_back(op.from);
        }
        info.nProblems = {};
        info.nRemaining = problems.size();
    }


    return info;
}

std::vector<Foam::label>
Foam::LoadBalancer::timesToProblemCounts(
    const std::vector<scalar>&           times,
    const DynamicList<ChemistryProblem>& problems)
{

    std::vector<int> counts;
    counts.reserve(times.size() + 1);
    auto begin = problems.begin();

    for(const auto& time : times)
    {
        scalar sum(0);
        auto operation = [&](const ChemistryProblem& problem) 
        {
            sum += problem.cpuTime;
            return sum <= time;
        };
        auto count = count_while(begin, problems.end(), operation);
        begin += count;
        counts.push_back(count);
    }


    return counts;
}

std::vector<Foam::LoadBalancer::Operation>
Foam::LoadBalancer::getOperations(
    DynamicList<ChemistryLoad>& loads, const ChemistryLoad& myLoad)
{

    double globalMean = getMean(loads);

    std::vector<Operation> operations;

    std::sort(loads.begin(), loads.end());

    auto sender = loads.end() - 1;
    auto receiver = loads.begin();

    while(sender != receiver)
    {
        double send_value = std::min(
            sender->value - globalMean, globalMean - receiver->value);

        Operation operation{sender->rank, receiver->rank, send_value};
        if(sender->rank == myLoad.rank || receiver->rank == myLoad.rank)
        {
            operations.push_back(operation);
        }
        sender->value -= send_value;
        receiver->value += send_value;

        if(std::abs(sender->value - globalMean) < SMALL)
        {
            sender--;
        }
        else
        {
            receiver++;
        }
    }

    // explicitly filter very small operations
    std::vector<Operation> large;
    for(const auto& op : operations)
    {
        if(op.value > 0.01 * globalMean)
        {
            large.push_back(op);
        }
    }

    runtime_assert(
        !((isSender(operations, myLoad.rank) &&
           isReceiver(operations, myLoad.rank))),
        "Only sender or receiver should be possible.");

    runtime_assert(
        std::abs(getMean(loads) - globalMean) < 1E-7, "Vanishing load");

    return large;
}

//perform load balance with Redez-vous algorithm
std::vector<Foam::LoadBalancer::Operation>
Foam::LoadBalancer::getOperationsRedezVous(
        DynamicList<ChemistryLoad>& loads, const ChemistryLoad& myLoad)
{
    double globalMean = getMean(loads); //calculate the mean load

    std::vector<Operation> operations;

    std::sort(loads.begin(), loads.end()); //sort the loads

    auto sender = loads.end() - 1; //the last load greater than the mean load
    auto receiver = loads.begin();

    for (int i = 0; i < int(loads.size()/2); ++i)
    {
        //std::cout << "sender: " << sender->rank << " receiver: " << receiver->rank << std::endl;

        double send_value = (sender->value - receiver->value) / 2; // calculate the send value

        Operation operation{sender->rank, receiver->rank, send_value};
        if(sender->rank == myLoad.rank || receiver->rank == myLoad.rank)
        {
            operations.push_back(operation); // if send or recv rank related to my rank, add to operations
        }
        sender->value -= send_value; // update the load
        receiver->value += send_value; // update the load

        sender--;
        receiver++;
    }

    // explicitly filter very small operations
    std::vector<Operation> large;
    for(const auto& op:operations)
    {
        if(op.value > 0.01 * globalMean) //if send value is larger than 1% of global mean
        {
            large.push_back(op); // add to large operations
        }
    }
    runtime_assert(
            !((isSender(operations, myLoad.rank) &&
            isReceiver(operations, myLoad.rank))),
            "Only sender or receiver should be possible.");

    runtime_assert(
            std::abs(getMean(loads) - globalMean) < 1E-7, "Vanishing load");

    return large; //return large operations
}

bool
Foam::LoadBalancer::isSender(
    const std::vector<Operation>& operations, int rank)
{
    if(operations.size() == 0)
    {
        return false;
    }

    for(const auto& op : operations)
    {
        if(op.from != rank)
        {
            return false;
        }
    }
    return true;
}

bool
Foam::LoadBalancer::isReceiver(
    const std::vector<Operation>& operations, int rank)
{
    for(const auto& op : operations)
    {
        if(op.to != rank)
        {
            return false;
        }
    }
    return true;
}

