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
    Foam::GpuSolution
    
Description
    A small object containing everything required for updating the reaction rate
    and the chemistry time step. These are passed around in the load balancer.

\*---------------------------------------------------------------------------*/

#ifndef GpuSolution_H
#define GpuSolution_H

#include "volFields.H"

namespace Foam
{

struct GpuSolution
{

    GpuSolution() = default;

    GpuSolution(label nspecie)
        : Ti(0), RRi(nspecie, 0.0), cellid(0)
    {
    }

    bool operator==(const GpuSolution& rhs) const
    {
        return false;
    }

    bool operator!=(const GpuSolution& rhs) const
    {
        return !(*this == rhs);
    }

    scalar Ti;
    scalarList RRi;
    label cellid;
};

//- Serialization for send
static inline Ostream& operator<<(Ostream& os, const GpuSolution& s)
{
    os << s.Ti;
    os << s.RRi;
    os << s.cellid;
    return os;
}

//- Get a serialized solution from IStream
static inline Istream& operator>>(Istream& is, GpuSolution& s)
{
    is >> s.Ti;
    is >> s.RRi;
    is >> s.cellid;
    return is;
}

} // namespace Foam

#endif