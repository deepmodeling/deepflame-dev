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

#include "blockRenumber.H"
#include "addToRunTimeSelectionTable.H"
#include "Random.H"
#include <vector>
#include <algorithm>
#include <limits>
#include <mpi.h>
#include <cassert>
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(blockRenumber, 0);

    addToRunTimeSelectionTable
    (
        renumberMethod,
        blockRenumber,
        dictionary
    );
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::blockRenumber::blockRenumber(const dictionary& renumberDict)
:
    renumberMethod(renumberDict)
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::labelList Foam::blockRenumber::renumber
(
    const pointField& points
) const
{
    Info << "Foam::blockRenumber::renumber begin" << endl;
    scalar x_min = std::numeric_limits<scalar>::max();
    scalar y_min = std::numeric_limits<scalar>::max();
    scalar z_min = std::numeric_limits<scalar>::max();
    scalar x_max = std::numeric_limits<scalar>::lowest();
    scalar y_max = std::numeric_limits<scalar>::lowest();
    scalar z_max = std::numeric_limits<scalar>::lowest();

    Info << "x_min : " << x_min << endl; 
    Info << "y_min : " << y_min << endl; 
    Info << "z_min : " << z_min << endl; 
    Info << "x_max : " << x_max << endl; 
    Info << "y_max : " << y_max << endl; 
    Info << "z_max : " << z_max << endl; 

    for(label i = 0; i < points.size(); ++i){
        x_min = std::min(x_min, points[i].x());
        y_min = std::min(y_min, points[i].y());
        z_min = std::min(z_min, points[i].z());
        x_max = std::max(x_max, points[i].x());
        y_max = std::max(y_max, points[i].y());
        z_max = std::max(z_max, points[i].z());
    }

    Info << "x_min : " << x_min << endl; 
    Info << "y_min : " << y_min << endl; 
    Info << "z_min : " << z_min << endl; 
    Info << "x_max : " << x_max << endl; 
    Info << "y_max : " << y_max << endl; 
    Info << "z_max : " << z_max << endl; 
    scalar box_x = x_max + x_min;
    scalar box_y = y_max + y_min;
    scalar box_z = z_max + z_min;

    // block partition

    label block_dim_x = 16;
    label block_dim_y = 1;
    label block_dim_z = 1;
    label block_count = block_dim_x * block_dim_y * block_dim_z;
    scalar block_unit_x = (box_x) / block_dim_x;
    scalar block_unit_y = (box_y) / block_dim_y;
    scalar block_unit_z = (box_z) / block_dim_z;

    Info << "block_unit_x : " << block_unit_x << endl; 
    Info << "block_unit_y : " << block_unit_y << endl; 
    Info << "block_unit_z : " << block_unit_z << endl; 

    std::vector<std::vector<label>> blocks(block_count);

    for(label i = 0; i < points.size(); ++i){
        label block_id_x = (points[i].x()) / block_unit_x;
        label block_id_y = (points[i].y()) / block_unit_y;
        label block_id_z = (points[i].z()) / block_unit_z;
        assert(block_id_x >= 0 && block_id_x < block_dim_x);
        assert(block_id_y >= 0 && block_id_y < block_dim_y);
        assert(block_id_z >= 0 && block_id_z < block_dim_z);
        label block_id = block_id_z * block_dim_x * block_dim_y + block_id_y * block_dim_x + block_id_x;
        assert(block_id >= 0 && block_id < block_count);
        blocks[block_id].push_back(i);
    }

    block_size_vector_.resize(block_count);
    for(size_t i = 0; i < blocks.size(); ++i){
        block_size_vector_[i] = blocks[i].size();
        Info << "block " << i << " : " << blocks[i].size() << endl;
    }

    labelList newToOld(points.size());
    label new_id = 0;
    for(size_t bid = 0; bid < blocks.size(); ++bid){
        for(auto& pid : blocks[bid]){
            newToOld[new_id] = pid;
            new_id += 1;
        }
    }
    
    Info << "Foam::blockRenumber::renumber end" << endl;
    return newToOld;
}



Foam::labelList Foam::blockRenumber::renumber
(
    const polyMesh& mesh,
    const pointField& points
) const
{
    return renumber(points);
}


Foam::labelList Foam::blockRenumber::renumber
(
    const labelListList& cellCells,
    const pointField& points
) const
{
    return renumber(points);
}


// ************************************************************************* //
