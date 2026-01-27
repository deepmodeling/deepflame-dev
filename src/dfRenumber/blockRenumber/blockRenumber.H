/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2019 OpenFOAM Foundation
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
    Foam::blockRenumber

Description
    Random renumber. Just to see effect of renumbering.

SourceFiles
    blockRenumber.C

\*---------------------------------------------------------------------------*/

#ifndef blockRenumber_H
#define blockRenumber_H

#include "renumberMethod.H"
#include <vector>

namespace Foam
{

/*---------------------------------------------------------------------------*\
                      Class blockRenumber Declaration
\*---------------------------------------------------------------------------*/

class blockRenumber
:
    public renumberMethod
{
private:
    mutable std::vector<label> block_size_vector_;

public:

    //- Runtime type information
    TypeName("block");


    // Constructors

        //- Construct given the renumber dictionary
        blockRenumber(const dictionary& renumberDict);

        //- Disallow default bitwise copy construction
        blockRenumber(const blockRenumber&) = delete;


    //- Destructor
    virtual ~blockRenumber()
    {}


    // Member Functions

        //- Return the order in which cells need to be visited, i.e.
        //  from ordered back to original cell label.
        //  This is only defined for blocketric renumberMethods.
        virtual labelList renumber(const pointField&) const;

        //- Return the order in which cells need to be visited, i.e.
        //  from ordered back to original cell label.
        //  Use the mesh connectivity (if needed)
        virtual labelList renumber
        (
            const polyMesh& mesh,
            const pointField& cc
        ) const;

        //- Return the order in which cells need to be visited, i.e.
        //  from ordered back to original cell label.
        //  The connectivity is equal to mesh.cellCells() except
        //  - the connections are across coupled patches
        virtual labelList renumber
        (
            const labelListList& cellCells,
            const pointField& cc
        ) const;


    // Member Operators

        //- Disallow default bitwise assignment
        void operator=(const blockRenumber&) = delete;

    const std::vector<label>& block_size_vector() const {
        return this->block_size_vector_;
    }

};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
