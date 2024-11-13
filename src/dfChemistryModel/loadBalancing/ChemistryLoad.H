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
    Foam::ChemistryLoad

Description
    A generic load object that stores a rank and a value and enables comparison
    as well as serialization through the OStream and IStream operators.

\*---------------------------------------------------------------------------*/

#ifndef ChemistryLoad_H
#define ChemistryLoad_H

#include "Istream.H"
#include "Ostream.H"

namespace Foam
{

// A struct containing rank and load value of each processor
struct ChemistryLoad
{
    // rank ID of the process
    label rank; 

    // the load value indicating how busy the processor is
    scalar value;

    ChemistryLoad() = default;

    ChemistryLoad(label _rank, scalar _value) : rank(_rank), value(_value)
    {
    }

    // overload comparisons for sorting
    bool operator==(const ChemistryLoad& rhs) const
    {
        return value == rhs.value;
    }
    bool operator!=(const ChemistryLoad& rhs) const
    {
        return !(value == rhs.value);
    }
    bool operator<=(const ChemistryLoad& rhs) const
    {
        return value <= rhs.value;
    }
    bool operator>=(const ChemistryLoad& rhs) const
    {
        return value >= rhs.value;
    }
    bool operator<(const ChemistryLoad& rhs) const
    {
        return value < rhs.value;
    }
    bool operator>(const ChemistryLoad& rhs) const
    {
        return value > rhs.value;
    }
};

//- Serialization for send OStream
static inline Ostream& operator<<(Ostream& os, const ChemistryLoad& l)
{

    os << l.rank;
    os << l.value;

    return os;
}

//- Get a serialized load from Istream
static inline Istream& operator>>(Istream& is, ChemistryLoad& l)
{

    is >> l.rank;
    is >> l.value;

    return is;
}

} // namespace Foam

#endif
