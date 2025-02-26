/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2018-2023 OpenCFD Ltd.
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
    Foam::meshObjects::gravity

Description
    Gravitational acceleration vector
    Although termed a \em MeshObject it is registered on Time only
    and thus identical for all regions.

SourceFiles
    gravityMeshObject.C

\*---------------------------------------------------------------------------*/

#ifndef Foam_meshObjects_gravity_H
#define Foam_meshObjects_gravity_H

#include "uniformDimensionedFields.H"
#include "MeshObject.H"
#include "Time.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace meshObjects
{

/*---------------------------------------------------------------------------*\
                           Class gravity Declaration
\*---------------------------------------------------------------------------*/

class gravity
:
    public MeshObject
    <
        Time,
        TopologicalMeshObject,
        meshObjects::gravity
    >,
    public uniformDimensionedVectorField
{
    // Private Typedefs

        typedef MeshObject
        <
            Time,
            TopologicalMeshObject,
            meshObjects::gravity
        > MeshObject_type;

public:

    //- Run-time type information
    TypeNameNoDebug("g");


    // Constructors

        //- Construct with given name on Time from \c constant (READ_MODIFIED)
        gravity(const word& name, const Time& runTime);

        //- Construct "g" field on Time from \c constant (READ_MODIFIED)
        explicit gravity(const Time& runTime)
        :
            gravity("g", runTime)
        {}

        //- Return named gravity field cached or construct on Time
        static const gravity& New(const word& name, const Time& runTime)
        {
            //return MeshObject_type::New(name, runTime);
            return MeshObject_type::New(runTime);
        }

        //- Return gravity "g" field cached or construct on Time
        static const gravity& New(const Time& runTime)
        {
            return MeshObject_type::New(runTime);
        }


    //- Destructor
    virtual ~gravity() = default;


    // Static Lookups

        //- Return const pointer to the gravity "g" object field
        static const uniformDimensionedVectorField*
        findObject(const Time& runTime)
        {
            //return runTime.findObject<uniformDimensionedVectorField>("g");
            return &(runTime.lookupObject<uniformDimensionedVectorField>("g"));
        }

        //- Return non-const pointer to the gravity "g" object field
        //- using a const-cast to have it behave like a mutable.
        static uniformDimensionedVectorField*
        getObjectPtr(const Time& runTime)
        {
            //return runTime.getObjectPtr<uniformDimensionedVectorField>("g");
            return const_cast<uniformDimensionedVectorField*>(&(runTime.lookupObject<uniformDimensionedVectorField>("g")));
        }

        //- Lookup and return const reference to the gravity "g" object field
        static const uniformDimensionedVectorField&
        lookupObject(const Time& runTime)
        {
            return runTime.lookupObject<uniformDimensionedVectorField>("g");
        }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace meshObjects
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
