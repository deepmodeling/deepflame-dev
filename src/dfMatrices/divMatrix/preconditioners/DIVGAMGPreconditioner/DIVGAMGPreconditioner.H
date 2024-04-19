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
    Foam::DIVGAMGPreconditioner

Description
    Geometric agglomerated algebraic multigrid preconditioner.

See also
    DIVGAMGSolver for more details.

SourceFiles
    DIVGAMGPreconditioner.C

\*---------------------------------------------------------------------------*/

#ifndef DIVGAMGPreconditioner_H
#define DIVGAMGPreconditioner_H

#include "DIVGAMGSolver.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                           Class DIVGAMGPreconditioner Declaration
\*---------------------------------------------------------------------------*/

class DIVGAMGPreconditioner
:
    public DIVGAMGSolver,
    public divMatrix::preconditioner
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
        DIVGAMGPreconditioner
        (
            const divMatrix::solver&,
            const dictionary& solverControls
        );


    //- Destructor
    virtual ~DIVGAMGPreconditioner();


    // Member Functions

        //- Return wA the preconditioned form of residual rA
        virtual void precondition
        (
            scalarField& wA,
            const scalarField& rA,
            const direction cmpt=0
        ) const;
    
    
    mutable double procondition_time = 0.;
    mutable double procondition_initVcycle_time = 0.;
    mutable double procondition_Vcycle_time = 0.;
    mutable double procondition_spmv_time = 0.;
    mutable double procondition_misc_time = 0.;

    virtual void print_time() const {
        double procondition_other_time = procondition_time - procondition_initVcycle_time - procondition_Vcycle_time - procondition_spmv_time - procondition_misc_time;
        Info << "DIVGAMGPreconditioner procondition profiling -------------------------------------------" << endl;
        Info << "procondition time " << procondition_time << endl;
        Info << "procondition initVcycle time : " << procondition_initVcycle_time << ", " << procondition_initVcycle_time / procondition_time * 100 << "%" << endl;
        Info << "procondition Vcycle time : " << procondition_Vcycle_time << ", " << procondition_Vcycle_time / procondition_time * 100 << "%" << endl;
        Info << "procondition spmv time : " << procondition_spmv_time << ", " << procondition_spmv_time / procondition_time * 100 << "%" << endl;
        Info << "procondition misc time : " << procondition_misc_time << ", " << procondition_misc_time / procondition_time * 100 << "%" << endl;
        Info << "procondition other time : " << procondition_other_time << ", " << procondition_other_time / procondition_time * 100 << "%" << endl;
        Info << "----------------------------------------------------------------------------------------" << endl;
        Info << "DIVGAMGSolver Vcycle profiling -------------------------------------------" << endl;
        Info << "Vcycle spmv time : " << Vcycle_spmv_time << ", " << Vcycle_spmv_time / Vcycle_total_time * 100 << "%" << endl;
        Info << "Vcycle smooth time : " << Vcycle_smooth_time << ", " << Vcycle_smooth_time / Vcycle_total_time * 100 << "%" << endl;
        Info << "Vcycle scale time : " << Vcycle_scale_time << ", " << Vcycle_scale_time / Vcycle_total_time * 100 << "%" << endl;
        Info << "Vcycle interpolate time : " << Vcycle_interpolate_time << ", " << Vcycle_interpolate_time / Vcycle_total_time * 100 << "%" << endl;
        Info << "Vcycle restrictField time : " << Vcycle_restrictField_time << ", " << Vcycle_restrictField_time / Vcycle_total_time * 100 << "%" << endl;
        Info << "Vcycle prolongField time : " << Vcycle_prolongField_time << ", " << Vcycle_prolongField_time / Vcycle_total_time * 100 << "%" << endl;
        Info << "Vcycle solveCoarsest time : " << Vcycle_misc_time << ", " << Vcycle_misc_time / Vcycle_total_time * 100 << "%" << endl;
        Info << "Vcycle misc time : " << Vcycle_solveCoarsest_time << ", " << Vcycle_solveCoarsest_time / Vcycle_total_time * 100 << "%" << endl;
        double Vcycle_other_time = Vcycle_total_time - Vcycle_spmv_time - Vcycle_smooth_time
        - Vcycle_scale_time - Vcycle_interpolate_time - Vcycle_restrictField_time - Vcycle_prolongField_time
        - Vcycle_solveCoarsest_time - Vcycle_misc_time;
        Info << "Vcycle other time : " << Vcycle_other_time << ", " << Vcycle_other_time / Vcycle_total_time * 100 << "%" << endl;
        Info << "----------------------------------------------------------------------------------------" << endl;    
        Info << "DIVGAMGSolver scale profiling -------------------------------------------" << endl;
        Info << "scale total time : " << scale_total_time << ", " << scale_total_time / scale_total_time * 100 << "%" << endl;
        Info << "scale spmv time : " << scale_spmv_time << ", " << scale_spmv_time / scale_total_time * 100 << "%" << endl;
        Info << "scale norm time : " << scale_norm_time << ", " << scale_norm_time / scale_total_time * 100 << "%" << endl;
        Info << "scale vector2D_reduce time : " << scale_vector2D_reduce_time << ", " << scale_vector2D_reduce_time / scale_total_time * 100 << "%" << endl;
        Info << "scale sf time : " << scale_sf_time << ", " << scale_sf_time / scale_total_time * 100 << "%" << endl;
        Info << "scale field time : " << scale_field_time << ", " << scale_field_time / scale_total_time * 100 << "%" << endl;
        double scale_other_time = scale_total_time - scale_spmv_time - scale_norm_time - scale_vector2D_reduce_time - scale_sf_time - scale_field_time;
        Info << "scale other time : " << scale_other_time << ", " << scale_other_time / scale_total_time * 100 << "%" << endl;
        Info << "----------------------------------------------------------------------------------------" << endl;    
        // Info << "Level " << 0 << " : " << endl; 
        // matrix_.print_time();
        // forAll(divMatrixLevels_, leveli){
        //     Info << "Level " << leveli + 1 << " : " << endl; 
        //     divMatrixLevels_[leveli].print_time();
        // }
        // Info << "----------------------------------------------------------------------------------------" << endl;
    }

};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
