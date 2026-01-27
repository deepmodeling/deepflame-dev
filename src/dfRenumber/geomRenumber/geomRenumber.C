/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
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

\*---------------------------------------------------------------------------*/

#include "geomRenumber.H"
#include "addToRunTimeSelectionTable.H"
#include "Random.H"
#include <vector>
#include <algorithm>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(geomRenumber, 0);

    addToRunTimeSelectionTable
    (
        renumberMethod,
        geomRenumber,
        dictionary
    );
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::geomRenumber::geomRenumber(const dictionary& renumberDict)
:
    renumberMethod(renumberDict)
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::labelList Foam::geomRenumber::renumber
(
    const pointField& points
) const
{
    // Info << "In Foam::geomRenumber::renumber" << endl;
    // Info << "before renumber : " << endl;
    // Info << "points.size() : " << points.size() << endl;
    // for(label i = 0; i < points.size(); ++i){
    //     Info << i << ", " << points[i].x() << ", " << points[i].y() << ", " << points[i].z() << endl;
    // }

    std::vector<Point> Points(points.size());
    forAll(points, i){
        Points[i].id_ = i;
        Points[i].x_ = static_cast<label>(points[i][0] * 1e6);
        Points[i].y_ = static_cast<label>(points[i][1] * 1e6);
        Points[i].z_ = static_cast<label>(points[i][2] * 1e6);
    }    
    
    std::sort(Points.begin(), Points.end(), [](const auto& a, const auto& b){
        return a.z_ < b.z_;
    });

    std::vector<label> z_split_ptr;
    z_split_ptr.push_back(0);
    for(size_t i = 1; i < Points.size(); ++i){
        if(Points[i].z_ != Points[i-1].z_){
            z_split_ptr.push_back(i);
        }
    }
    z_split_ptr.push_back(points.size());

    #pragma omp parallel for
    for(size_t z_split_index = 0; z_split_index < z_split_ptr.size() - 1; ++z_split_index){
        label z_start_index = z_split_ptr[z_split_index];
        label z_end_index = z_split_ptr[z_split_index+1];
        std::sort(Points.begin() + z_start_index, Points.begin() + z_end_index, [](const auto& a, const auto& b){
            return a.y_ < b.y_;
        });
        std::vector<label> y_split_ptr;
        y_split_ptr.push_back(z_start_index);
        for(size_t i = z_start_index + 1; i < z_end_index; ++i){
            if(Points[i].y_ != Points[i-1].y_){
                y_split_ptr.push_back(i);
            }
        }
        y_split_ptr.push_back(z_end_index);

        for(size_t y_split_index = 0; y_split_index < y_split_ptr.size() - 1; ++y_split_index){
            label y_start_index = y_split_ptr[y_split_index];
            label y_end_index = y_split_ptr[y_split_index+1];   
            std::sort(Points.begin() + y_start_index, Points.begin() + y_end_index, [](const auto& a, const auto& b){
                return a.x_ < b.x_;
            });
        }
    }

    // for(size_t i = 0; i < Points.size(); ++i){
    //     Info << "(" << Points[i].x_ << ", " << Points[i].y_ << ", " << Points[i].z_ << ")" << endl;
    // }

    labelList newToOld(points.size());
    for(size_t i = 0; i < Points.size(); ++i){
        newToOld[i] = Points[i].id_;
    }
    return newToOld;
}



Foam::labelList Foam::geomRenumber::renumber
(
    const polyMesh& mesh,
    const pointField& points
) const
{
    return renumber(points);
}


Foam::labelList Foam::geomRenumber::renumber
(
    const labelListList& cellCells,
    const pointField& points
) const
{
    return renumber(points);
}


// ************************************************************************* //
