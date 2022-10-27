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
    Foam::LoadBalancer

Description
    Extends the base class LoadBalancerBase by implementing a
    balancing algorithm which tries to set the global mean load to each rank.
    
SourceFiles
    LoadBalancer.C
\*---------------------------------------------------------------------------*/

#ifndef LoadBalancer_H
#define LoadBalancer_H

#include "ChemistryProblem.H"
#include "IOdictionary.H"
#include "LoadBalancerBase.H"
#include "Switch.H"
#include "algorithms_DLB.H"
#include "runTimeSelectionTables.H"
#include "scalarField.H"

#include <algorithm>
#include <vector>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

class LoadBalancer : public LoadBalancerBase
{


public:
    struct Operation
    {
        int from, to; // ranks
        double value;
    };

    LoadBalancer() = default;

    LoadBalancer(const dictionary& dict)
        : LoadBalancerBase(), dict_(dict),
          coeffsDict_(dict.subDict("loadbalancing")),
          active_(coeffsDict_.lookupOrDefault<Switch>("active", true)),
          log_(coeffsDict_.lookupOrDefault<Switch>("log", false))
    {
    }

    // Destructor
    virtual ~LoadBalancer() = default;

    //- Given a list of problems, update the balancer state member
    virtual void updateState(const DynamicList<ChemistryProblem>& problems);

    //- Is load balancing active?
    bool active() const
    {
        return active_;
    }

    //- Is load balancing logged?
    bool log() const
    {
        return log_;
    }



protected:

    //- Get the operations for this rank that would minimize the load to
    //  global mean
    static std::vector<LoadBalancer::Operation> getOperations(
        DynamicList<ChemistryLoad>& loads, const ChemistryLoad& myLoad);

    //- Convert the operations to send and receive info to handle balancing
    static BalancerState operationsToInfo(
        const std::vector<Operation>& operations,
        const DynamicList<ChemistryProblem>& problems,
        const ChemistryLoad& myLoad);


    //- Convert the vector of cpu times to number of problems for the rank
    static std::vector<label> timesToProblemCounts(
        const std::vector<scalar>& times,
        const DynamicList<ChemistryProblem>& problems);


private:

    const dictionary dict_;

    const dictionary coeffsDict_;

    // Is load balancing active?
    Switch active_;

    // Is load balancing logged?
    Switch log_;

    //- Check if the rank is a sender
    static bool isSender(const std::vector<Operation>& operations, int rank);

    //- Check if the rank is a receiver. Zero operations marked as receiver.
    static bool isReceiver(const std::vector<Operation>& operations, int rank);
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // namespace Foam

#endif