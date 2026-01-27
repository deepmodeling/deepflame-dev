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

Class
    Foam::dfGAMGPreconditioner

Description
    Geometric agglomerated algebraic multigrid preconditioner.

See also
    dfGAMGSolver for more details.

SourceFiles
    dfGAMGPreconditioner.C

\*---------------------------------------------------------------------------*/

#ifndef dfGAMGPreconditioner_H
#define dfGAMGPreconditioner_H

#include "dfGAMGSolver.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                           Class dfGAMGPreconditioner Declaration
\*---------------------------------------------------------------------------*/

class dfGAMGPreconditioner
:
    public dfGAMGSolver,
    public dfMatrix::preconditioner
{
protected:
    // Protected data

        //- Number of V-cycles to perform
        label nVcycles_;

        //- Read the control parameters from the controlDict_
        virtual void readControls();

public:

    //- Runtime type information
    TypeName("GAMG");


    // Constructors

        //- Construct from matrix components and preconditioner solver controls
        dfGAMGPreconditioner
        (
            const dfMatrix::solver&,
            const dictionary& solverControls
        );


    //- Destructor
    virtual ~dfGAMGPreconditioner();


    // Member Functions

        //- Return wA the preconditioned form of residual rA
        virtual void precondition
        (
            scalarField& wA,
            const scalarField& rA,
            const direction cmpt=0
        ) const;
    
    
    mutable double procondition_time = 0.;
    mutable double initVcycle_time = 0.;
    mutable double Vcycle_time = 0.;

    virtual void print_time() const {
        double other_time = procondition_time - initVcycle_time - spmv_time - smooth_time - scale_time - interpolate_time - restrictField_time - prolongField_time - solveCoarsest_time;
        Info << "dfGAMGPreconditioner procondition profiling -------------------------------------------" << endl;
        Info << "procondition time " << procondition_time << endl;
        Info << "initVcycle time : " << initVcycle_time << ", " << initVcycle_time / procondition_time * 100 << "%" << endl;
        Info << "spmv time : " << spmv_time << ", " << spmv_time / procondition_time * 100 << "%" << endl;
        Info << "smooth time : " << smooth_time << ", " << smooth_time / procondition_time * 100 << "%" << endl;
        Info << "scale time : " << scale_time << ", " << scale_time / procondition_time * 100 << "%" << endl;
        Info << "interpolate time : " << interpolate_time << ", " << interpolate_time / procondition_time * 100 << "%" << endl;
        Info << "restrictField time : " << restrictField_time << ", " << restrictField_time / procondition_time * 100 << "%" << endl;
        Info << "prolongField time : " << prolongField_time << ", " << prolongField_time / procondition_time * 100 << "%" << endl;
        Info << "solveCoarsest time : " << solveCoarsest_time << ", " << solveCoarsest_time / procondition_time * 100 << "%" << endl;
        Info << "other time : " << other_time << ", " << other_time / procondition_time * 100 << "%" << endl;
        Info << "----------------------------------------------------------------------------------------" << endl;
        Info << "Level " << 0 << " : " << endl; 
        matrix_.print_time();
        forAll(dfMatrixLevels_, leveli){
            Info << "Level " << leveli + 1 << " : " << endl; 
            dfMatrixLevels_[leveli].print_time();
        }
        Info << "----------------------------------------------------------------------------------------" << endl;
    }

};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
