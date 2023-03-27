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
    Foam::linearSolverContext

Description
    The base class for a generic linearSolverContext

SourceFiles
    linearSolverContext.C

\*---------------------------------------------------------------------------*/

#ifndef foamLinearSolverContext_H
#define foamLinearSolverContext_H

#include "solverPerformance.H"
#include "cacheManager.H"


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                     Class linearSolverContext Declaration
\*---------------------------------------------------------------------------*/

class linearSolverContext
{
    // Private Data

        bool init_;

public:

    // Public Members

        solverPerformance performance;
        cacheManager caching;

        const word eqName;

    // Constructors

        //- Default construct
        linearSolverContext(const word eqName)
        :
            init_(false),
            eqName(eqName)
        {}


    //- Destructor
    virtual ~linearSolverContext()
    {
    }


    // Member Functions

        //- Return value of initialized
        bool initialized() const
        {
            return init_;
        }

        //- Return value of initialized
        bool& initialized()
        {
            return init_;
        }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
