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
    Foam::amgxLinearSolverContext

Description
    A struct containing a AmgXWrapper and Foam2CSR for each equation to be solved

SourceFiles
    amgxLinearSolverContext.C

\*---------------------------------------------------------------------------*/

#ifndef amgxFoamLinearSolverContext_H
#define amgxFoamLinearSolverContext_H

#include "linearSolverContext.H"
#include "AmgXSolver.H"
#include "AmgXCSRMatrix.H"
#include "uncollatedFileOperation.H"


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                     Class amgxLinearSolverContext Declaration
\*---------------------------------------------------------------------------*/

class amgxLinearSolverContext 
: 
    public linearSolverContext
{
    // Private Data

        bool loaded_;

public:

    //- Runtime information
        TypeName("amgxLinearSolverContext");

    // Public Members

        AmgXSolver amgx;
        AmgXCSRMatrix Amat; 


        static const word packageName;


    // Constructors

        //- Default construct
        amgxLinearSolverContext(const word eqName, const fileName& optionsFile)
        :
            linearSolverContext(eqName),
            loaded_(false)
        {
            int err = 0;

            if (isFile(optionsFile))
            {
                // AMGx initialize
                amgx.initialize(MPI_COMM_WORLD, "dDDI", optionsFile);
            
                MPI_Barrier(MPI_COMM_WORLD);
            }
            else
            {
                // AMGx initialize no arguments
                err = 1;
                loaded_ = 0;
                Info<< "Error: AmgX-" << eqName << " cannot be initialized without a valid config file" << nl;
                Info<< optionsFile << " cannot be found" << nl;
            }

            if (!err)
            {
                Info<< "Initializing AmgX-" << eqName << nl;
                loaded_ = 1;
            }
            else
            {
                Info<< "Could not initialize AmgX-" << eqName << nl;
            }
        }

    //- Destructor
    virtual ~amgxLinearSolverContext()
    {
        if (loaded_ > 0)
        {
            Info<< "Finalizing AmgX-" << eqName << nl;
            amgx.finalize();
            Amat.finalise();
            loaded_ = 0;
        }
        else if (!loaded_)
        {
            Info<< "AmgX-" << eqName << " already finalized" << nl;
        }
    }


    // Member Functions

        bool loaded() const
        {
            return loaded_;
        }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
