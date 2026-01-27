/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2019 OpenCFD Ltd.
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
    Foam::linearSolverContextTable

Description
    A mesh-based collection linear solver contexts for external solvers

SourceFiles
    linearSolverContextTable.C

\*---------------------------------------------------------------------------*/

#ifndef foamLinearSolverContextTable_H
#define foamLinearSolverContextTable_H

#include "fvMesh.H"
#include "MeshObject.H"
#include "HashPtrTable.H"
// #include "petscLinearSolverContext.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                  Class linearSolverContextTable Declaration
\*---------------------------------------------------------------------------*/

template <class T>
class linearSolverContextTable
:
    public MeshObject
    <
        fvMesh,
        TopologicalMeshObject,
        linearSolverContextTable<T>
    >
{
    // Private Data

        //- Linear solver contexts
        mutable HashPtrTable<T> contexts_;


public:

    //- Runtime information
    TypeName("linearSolverContextTable");


    // Constructors

        //- Construct from mesh for given field
        explicit linearSolverContextTable(const fvMesh& mesh)
        :
            MeshObject
            <
                fvMesh,
                Foam::TopologicalMeshObject,
                linearSolverContextTable<T>
            >(mesh),
            contexts_()
        {}


    //- Destructor
    virtual ~linearSolverContextTable() = default;


    // Member Functions

        //- Linear solver contexts (mutable)
        HashPtrTable<T>& contexts() const
        {
            return contexts_;
        }


        //- Test if named context exists
        bool haveContext(const word& name) const
        {
            return contexts_.found(name);
        }


        //- Get existing or a new context by name
        T& getContext(const word& name) const
        {
            auto iter = contexts().find(name);

            const bool cached = (iter!=HashTableCore::end());   //NOTE!!!!
            

            if (iter==HashTableCore::end())
            {
                contexts().set(name, new T{name, this->mesh().thisDb().time().system()/T::packageName + name + "Options"});
                iter = contexts().find(name);
            }

            DebugInfo
                << "cache-" << (cached ? "hit" : "miss")
                << T::packageName << " Generic linear solver context "
                << name << " mesh: " << this->mesh().name() << nl;

            return *(*iter);
        }


        //- Remove context by name
        bool removeContext(const word& name) const
        {
            return contexts().erase(name);
        }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //



