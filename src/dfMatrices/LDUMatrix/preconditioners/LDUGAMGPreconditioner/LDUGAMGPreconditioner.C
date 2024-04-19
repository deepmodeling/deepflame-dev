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

#include "LDUGAMGPreconditioner.H"
#include <mpi.h>

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(LDUGAMGPreconditioner, 0);

    LDUMatrix::preconditioner::addsymMatrixConstructorToTable
    <LDUGAMGPreconditioner> addLDUGAMGPreconditionerSymMatrixConstructorToTable_;

    LDUMatrix::preconditioner::addasymMatrixConstructorToTable
    <LDUGAMGPreconditioner> addLDUGAMGPreconditionerAsymMatrixConstructorToTable_;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::LDUGAMGPreconditioner::LDUGAMGPreconditioner
(
    const LDUMatrix::solver& sol,
    const dictionary& solverControls
)
:
    LDUGAMGSolver
    (
        sol.fieldName(),
        sol.matrix(),
        sol.interfaceBouCoeffs(),
        sol.interfaceIntCoeffs(),
        sol.interfaces(),
        solverControls
    ),
    LDUMatrix::preconditioner(sol),
    nVcycles_(2)
{
    readControls();
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::LDUGAMGPreconditioner::~LDUGAMGPreconditioner()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::LDUGAMGPreconditioner::readControls()
{
    LDUGAMGSolver::readControls();
    nVcycles_ = controlDict_.lookupOrDefault<label>("nVcycles", 2);
}


void Foam::LDUGAMGPreconditioner::precondition
(
    scalarField& wA,
    const scalarField& rA,
    const direction cmpt
) const
{
    double start ,end;

    double procondition_start = MPI_Wtime();
    wA = 0.0;
    scalarField AwA(wA.size());
    scalarField finestCorrection(wA.size());
    scalarField finestResidual(rA);

    // Create coarse grid correction fields
    PtrList<scalarField> coarseCorrFields;

    // Create coarse grid sources
    PtrList<scalarField> coarseSources;

    // Create the smoothers for all levels
    PtrList<LDUMatrix::smoother> smoothers;

    // Scratch fields if processor-agglomerated coarse level meshes
    // are bigger than original. Usually not needed
    scalarField ApsiScratch;
    scalarField finestCorrectionScratch;

    // Initialise the above data structures
    start = MPI_Wtime();
    initVcycle
    (
        coarseCorrFields,
        coarseSources,
        smoothers,
        ApsiScratch,
        finestCorrectionScratch
    );
    end = MPI_Wtime();
    initVcycle_time += end - start;


    start = MPI_Wtime();
    for (label cycle=0; cycle<nVcycles_; cycle++)
    {
        Vcycle
        (
            smoothers,
            wA,
            rA,
            AwA,
            finestCorrection,
            finestResidual,

            (ApsiScratch.size() ? ApsiScratch : AwA),
            (
                finestCorrectionScratch.size()
              ? finestCorrectionScratch
              : finestCorrection
            ),

            coarseCorrFields,
            coarseSources,
            cmpt
        );

        if (cycle < nVcycles_-1)
        {
            // Calculate finest level residual field
            matrix_.Amul(AwA, wA, interfaceBouCoeffs_, interfaces_, cmpt);
            finestResidual = rA;
            finestResidual -= AwA;
        }
    }
    end = MPI_Wtime();
    Vcycle_time += end - start;

    double procondition_end = MPI_Wtime();
    procondition_time += procondition_end - procondition_start;
}


// ************************************************************************* //
