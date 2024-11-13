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
    Foam::SendBuffer

Description
    Wrapper around Foam::DynamicList which is used to slice subsets to avoid
    unnecessary copies when sending data.

\*---------------------------------------------------------------------------*/

#ifndef SendBuffer_H
#define SendBuffer_H

#include "DynamicList.H"

#include <numeric> //std::accumulate
#include <vector>

namespace Foam
{

template <class T>
struct SendBuffer
{
    SendBuffer(const DynamicList<T>& values, const std::vector<label>& counts)
        : 
            m_values(values), m_counts(counts)
    {
    }

    SubList<T> operator[](label i) const
    {
        label start =
            std::accumulate(m_counts.begin(), m_counts.begin() + i, 0);
        return SubList<T>(m_values, m_counts[i], start);
    }

    const DynamicList<T>& m_values;
    std::vector<label> m_counts;
};

} // namespace Foam

#endif