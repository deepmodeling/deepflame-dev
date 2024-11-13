/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2018 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

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
    Foam::GpuProblem

Description
    A small object containing everything required for solving the reaction rate
    using the ODE solver. These are passed around in the load balancer.

\*---------------------------------------------------------------------------*/

#ifndef GpuProblem_H
#define GpuProblem_H

#include "volFields.H"

namespace Foam
{

struct GpuProblem
{

    GpuProblem() = default;
    GpuProblem(label nSpecie)
        : Y(nSpecie), Ti(0), pi(0), rhoi(0), DNNid(0), cellid(0)
    {
    }

    scalarList Y;
    scalar Ti;
    scalar pi;
    scalar rhoi;
    label DNNid;
    label cellid;

    // TODO: implement!
    bool operator==(const GpuProblem& rhs) const
    {
        return false;
    }

    bool operator!=(const GpuProblem& rhs) const
    {
        return !(*this == rhs);
    }
};

//- Serialization for send
static inline Ostream& operator<<(Ostream& os, const GpuProblem& p)
{

    os << p.Y;
    os << p.Ti;
    os << p.pi;
    os << p.rhoi;
    os << p.DNNid;
    os << p.cellid;

    return os;
}

//- Get a serialized problem from IStream
static inline Istream& operator>>(Istream& is, GpuProblem& p)
{

    is >> p.Y;
    is >> p.Ti;
    is >> p.pi;
    is >> p.rhoi;
    is >> p.DNNid;
    is >> p.cellid;

    return is;
}

} // namespace Foam

#endif