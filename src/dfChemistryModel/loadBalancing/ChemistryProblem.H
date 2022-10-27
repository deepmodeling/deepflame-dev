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
    Foam::ChemistryProblem

Description
    A small object containing everything required for updating the reaction rate
    and the chemistry time step. These are passed around in the load balancer.

Note
    2022-Oct: Modified to adapt with the DeepFlame chemistry class.

\*---------------------------------------------------------------------------*/

#ifndef ChemistryProblem_H
#define ChemistryProblem_H

#include "volFields.H"

namespace Foam
{

struct ChemistryProblem
{

    ChemistryProblem() = default;
    ChemistryProblem(label nSpecie)
        : Y(nSpecie), Ti(0), pi(0), rhoi(0), deltaT(0), cellid(0)
    {
    }

    scalarList Y;
    scalar Ti;
    scalar pi;
    scalar rhoi;
    scalar deltaT;
    scalar cpuTime;
    label cellid;

    // TODO: implement!
    bool operator==(const ChemistryProblem& rhs) const
    {
        return false;
    }

    bool operator!=(const ChemistryProblem& rhs) const
    {
        return !(*this == rhs);
    }
};

//- Serialization for send
static inline Ostream& operator<<(Ostream& os, const ChemistryProblem& p)
{

    os << p.Y;
    os << p.Ti;
    os << p.pi;
    os << p.rhoi;
    os << p.deltaT;
    os << p.cpuTime;
    os << p.cellid;

    return os;
}

//- Get a serialized problem from IStream
static inline Istream& operator>>(Istream& is, ChemistryProblem& p)
{

    is >> p.Y;
    is >> p.Ti;
    is >> p.pi;
    is >> p.rhoi;
    is >> p.deltaT;
    is >> p.cpuTime;
    is >> p.cellid;

    return is;
}

} // namespace Foam

#endif