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
    Foam::algorithms_DLB.H

Description
    Converts an iterator to a reverse iterator, since UList:: reverse iterator
    does not behave like one.
    
    See bug report on: https://bugs.openfoam.org/view.php?id=3534

    The issue is resolved in OpenFOAM-dev in commit 17c40d2aa57b1f922cdddc315d8cf3471c9bc600 
    
\*---------------------------------------------------------------------------*/

#pragma once

#include <iterator>
namespace Foam
{

//- Convert an iterator to a reverse iterator
template <class InputIt>
std::reverse_iterator<InputIt> make_reverse(InputIt i)
{
    return std::reverse_iterator<InputIt>(i);
}

//- Compute the number of sequential elements for which the predicate is true
template <class InputIt, class UnaryPredicate>
typename std::iterator_traits<InputIt>::difference_type
count_while(InputIt first, InputIt last, UnaryPredicate predicate)
{

    typename std::iterator_traits<InputIt>::difference_type ret = 0;
    for(; first != last; ++first)
    {
        if(!predicate(*first))
        {
            break;
        }
        ret++;
    }
    return ret;
}

} // namespace Foam