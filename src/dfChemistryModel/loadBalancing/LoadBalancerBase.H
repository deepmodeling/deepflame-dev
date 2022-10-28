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

Class
    Foam::LoadBalancerBase

Description
    An abstract base class for all balancing methods.

SourceFiles
    LoadBalancerBase.C

\*---------------------------------------------------------------------------*/

#ifndef LoadBalancerBase_H
#define LoadBalancerBase_H

#include "ChemistryLoad.H"
#include "ChemistryProblem.H"
#include "ChemistrySolution.H"
#include "RecvBuffer.H"
#include "SendBuffer.H"
#include "runtime_assert.H"

#include <algorithm> //std::min/max element
#include <numeric>   //std::accumulate
#include <vector>    //std::vector

namespace Foam
{

class LoadBalancerBase
{

public:
    struct BalancerState
    {
        std::vector<label> sources;      // ranks which send to this process
        std::vector<label> destinations; // ranks to which this process sends to
        std::vector<label> nProblems; // counts which this process sends/receivs
        label nRemaining;             // own problems / solutions
    };


private:
    BalancerState state_; // the current state of the object


public:
    LoadBalancerBase() = default;

    virtual ~LoadBalancerBase() = default;

    //- Check if load balancing is active
    virtual bool active() const = 0;

    //- The load balancing algorithm which each derived class must define
    virtual void updateState(const DynamicList<ChemistryProblem>& problems) = 0;

    //- Calculate the mean of the given list of loads
    static double getMean(const DynamicList<ChemistryLoad>& loads);

    //- Find the minimum value of the given list of loads
    static ChemistryLoad getMin(const DynamicList<ChemistryLoad>& loads);

    //- Find the maximum value of the given list of loads
    static ChemistryLoad getMax(const DynamicList<ChemistryLoad>& loads);

    //- Compute the load based on the given list of chemistry problems
    static ChemistryLoad
    computeLoad(const DynamicList<ChemistryProblem>& problems);

    //- Gather the data from all ranks
    template <class T>
    static DynamicList<T> allGather(const T& myData);

    //- Set the current state of the rank
    void setState(const BalancerState& state);

    //- Get the current state of the rank
    const BalancerState& getState() const
    {
        return state_;
    }

    //- Given a list of values, split them evenly between MPI processes
    template <class T>
    RecvBuffer<T> balance(const DynamicList<T>& values) const;

    //- Given a buffer of values, send the values back to their owner ranks
    template <class T>
    RecvBuffer<T> unbalance(const RecvBuffer<T>& values) const;

    //- Print the current state information
    void printState() const;

    //- Convert a vector of std::string for printing purposes
    template <class T>
    static std::string vectorToString(const std::vector<T>& vec);

    //- Find the index of the load having a load.rank = rank
    label
    rankToLoadIdx(const DynamicList<ChemistryLoad>& loads, label rank) const;

    //- Check if the load state is valid
    bool validState() const;

    //- Send the split send_buffer to sources and receive everything from
    //  destinations
    template <class ET, class Indexable>
    static RecvBuffer<ET> sendRecv(
        const Indexable&          send_buffer,
        const std::vector<label>& sources,
        const std::vector<label>& destinations);

    //- Slice an nRemaining size portion from the _end_ of the values
    template <class T>
    SubList<T> getRemaining(const DynamicList<T>& values)
    {

        return SubList<T>(
            values, state_.nRemaining, values.size() - state_.nRemaining);
    }
};

template <class T>
DynamicList<T> LoadBalancerBase::allGather(const T& myData)
{

    DynamicList<T> ret(Pstream::nProcs(), myData);
    ret[Pstream::myProcNo()] = myData;

    label tag = 1;
    Pstream::gatherList(ret, tag);
    Pstream::scatterList(ret, tag);

    return ret;
}

template <class T>
RecvBuffer<T> LoadBalancerBase::balance(const DynamicList<T>& values) const
{

    return sendRecv<T, SendBuffer<T>>(
        SendBuffer<T>(values, state_.nProblems),
        state_.sources,
        state_.destinations);
}

template <class T>
RecvBuffer<T> LoadBalancerBase::unbalance(const RecvBuffer<T>& values) const
{

    return sendRecv<T, RecvBuffer<T>>(
        values, state_.destinations, state_.sources);
}

template <class T>
std::string LoadBalancerBase::vectorToString(const std::vector<T>& vec)
{

    std::string ret = "(";

    for(const auto& v : vec)
    {
        ret += std::to_string(v) + ' ';
    }
    ret += ")";
    return ret;
}

template <class ET, class Indexable>
RecvBuffer<ET> LoadBalancerBase::sendRecv(
    const Indexable&          send_buffer,
    const std::vector<label>& sources,
    const std::vector<label>& destinations)
{

    RecvBuffer<ET> ret;

    if(Pstream::parRun())
    {

        PstreamBuffers pBufs(Pstream::commsTypes::nonBlocking);

        for(label i = 0; i < label(destinations.size()); ++i)
        {
            UOPstream send(destinations[i], pBufs);
            send << send_buffer[i];
        }

        pBufs.finishedSends();

        ret.setSize(sources.size());
        for(label i = 0; i < label(sources.size()); ++i)
        {
            UIPstream recv(sources[i], pBufs);
            recv >> ret[i];
        }
    }

    return ret;
}

} // namespace Foam

#endif

// ************************************************************************* //
