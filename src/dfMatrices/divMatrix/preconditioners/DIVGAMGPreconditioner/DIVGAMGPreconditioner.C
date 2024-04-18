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

#include "DIVGAMGPreconditioner.H"
#include <mpi.h>
#include <clockTime.H>

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(DIVGAMGPreconditioner, 0);

    divMatrix::preconditioner::addsymMatrixConstructorToTable
    <DIVGAMGPreconditioner> addDIVGAMGPreconditionerSymMatrixConstructorToTable_;

    divMatrix::preconditioner::addasymMatrixConstructorToTable
    <DIVGAMGPreconditioner> addDIVGAMGPreconditionerAsymMatrixConstructorToTable_;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::DIVGAMGPreconditioner::DIVGAMGPreconditioner
(
    const divMatrix::solver& sol,
    const dictionary& solverControls
)
:
    DIVGAMGSolver
    (
        sol.fieldName(),
        sol.matrix(),
        sol.interfaceBouCoeffs(),
        sol.interfaceIntCoeffs(),
        sol.interfaces(),
        solverControls
    ),
    divMatrix::preconditioner(sol),
    nVcycles_(2)
{
    readControls();
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::DIVGAMGPreconditioner::~DIVGAMGPreconditioner()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::DIVGAMGPreconditioner::readControls()
{
    DIVGAMGSolver::readControls();
    nVcycles_ = controlDict_.lookupOrDefault<label>("nVcycles", 2);
}


void Foam::DIVGAMGPreconditioner::precondition
(
    scalarField& wA,
    const scalarField& rA,
    const direction cmpt
) const
{
    clockTime clock;

    wA = 0.0;
    scalarField AwA(wA.size());
    scalarField finestCorrection(wA.size());
    scalarField finestResidual(rA);

    // Create coarse grid correction fields
    PtrList<scalarField> coarseCorrFields;

    // Create coarse grid sources
    PtrList<scalarField> coarseSources;

    // Create the smoothers for all levels
    PtrList<divMatrix::smoother> smoothers;

    // Scratch fields if processor-agglomerated coarse level meshes
    // are bigger than original. Usually not needed
    scalarField ApsiScratch;
    scalarField finestCorrectionScratch;

    procondition_misc_time += clock.timeIncrement();

    // Initialise the above data structures
    initVcycle
    (
        coarseCorrFields,
        coarseSources,
        smoothers,
        ApsiScratch,
        finestCorrectionScratch
    );

    procondition_initVcycle_time += clock.timeIncrement();

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

        procondition_Vcycle_time += clock.timeIncrement();

        if (cycle < nVcycles_-1)
        {
            // Calculate finest level residual field
            matrix_.Amul(AwA, wA, interfaceBouCoeffs_, interfaces_, cmpt);
            finestResidual = rA;
            finestResidual -= AwA;
        }
        procondition_spmv_time += clock.timeIncrement();
    }

    procondition_time += clock.elapsedTime();
}


// ************************************************************************* //
