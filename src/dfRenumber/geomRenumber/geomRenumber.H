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
    Foam::geomRenumber

Description
    Random renumber. Just to see effect of renumbering.

SourceFiles
    geomRenumber.C

\*---------------------------------------------------------------------------*/

#ifndef geomRenumber_H
#define geomRenumber_H

#include "renumberMethod.H"

namespace Foam
{

/*---------------------------------------------------------------------------*\
                      Class geomRenumber Declaration
\*---------------------------------------------------------------------------*/

class geomRenumber
:
    public renumberMethod
{
public:

    //- Runtime type information
    TypeName("geom");


    // Constructors

        //- Construct given the renumber dictionary
        geomRenumber(const dictionary& renumberDict);

        //- Disallow default bitwise copy construction
        geomRenumber(const geomRenumber&) = delete;


    //- Destructor
    virtual ~geomRenumber()
    {}


    // Member Functions

        //- Return the order in which cells need to be visited, i.e.
        //  from ordered back to original cell label.
        //  This is only defined for geometric renumberMethods.
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
        void operator=(const geomRenumber&) = delete;
    class Point{
    public:
        label id_;
        label x_;
        label y_;
        label z_;
        Point(){}
        Point(label id,scalar x, scalar y, scalar z):id_(id), x_(x), y_(y), z_(z){}
        ~Point(){}
        bool operator <(const Point& other) const{
            if(z_ < other.z_){
                return true;
            }
            if(z_ > other.z_){
                return false;
            }
            if(y_ < other.y_){
                return true;
            }
            if(y_ > other.y_){
                return false;
            }
            if(x_ < other.x_){
                return true;
            }
            if(x_ > other.x_){
                return false;
            }
            return id_ < other.id_;
        }
        bool operator >(const Point& other) const {
            if(z_ > other.z_){
                return true;
            }
            if(z_ < other.z_){
                return false;
            }
            if(y_ > other.y_){
                return true;
            }
            if(y_ < other.y_){
                return false;
            }
            if(x_ > other.x_){
                return true;
            }
            if(x_ < other.x_){
                return false;
            }
            return id_ > other.id_;
        }
    };
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
