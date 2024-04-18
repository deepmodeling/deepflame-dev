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

#include "DIVGAMGSolver.H"
#include "vector2D.H"
#include <mpi.h>
#include "clockTime.H"

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

void Foam::DIVGAMGSolver::scale
(
    scalarField& field,
    scalarField& Acf,
    const divMatrix& A,
    const FieldField<Field, scalar>& interfaceLevelBouCoeffs,
    const lduInterfaceFieldPtrsList& interfaceLevel,
    const scalarField& source,
    const direction cmpt
) const
{
    scalar* fieldPtr = field.begin();
    scalar* AcfPtr = Acf.begin();
    const scalar* sourcePtr = source.begin();
    const scalar* diagPtr = A.diag().begin();

    const label nCells = source.size();

    clockTime clock;

    A.Amul
    (
        Acf,
        field,
        interfaceLevelBouCoeffs,
        interfaceLevel,
        cmpt
    );

    scale_spmv_time += clock.timeIncrement();

    scalar scalingFactorNum = 0.0;
    scalar scalingFactorDenom = 0.0;

#ifdef _OPENMP
    #pragma omp parallel for reduction(+:scalingFactorNum) reduction(+:scalingFactorDenom)
#endif
    for(label i = 0; i < nCells; ++i){
        scalingFactorNum += sourcePtr[i] * fieldPtr[i];
        scalingFactorDenom += AcfPtr[i] * fieldPtr[i];
    }

    scale_norm_time += clock.timeIncrement();

    vector2D scalingVector(scalingFactorNum, scalingFactorDenom);

    reduce(scalingVector, sumOp<vector2D>());

    scale_vector2D_reduce_time += clock.timeIncrement();

    const scalar sf = scalingVector.x()/stabilise(scalingVector.y(), vSmall);

    if (debug >= 2)
    {
        Pout<< sf << " ";
    }

    scale_sf_time += clock.timeIncrement();

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for(label i = 0; i < nCells; ++i){
        field[i] = sf * fieldPtr[i] + (sourcePtr[i] - sf * AcfPtr[i]) / diagPtr[i];
    }

    scale_field_time += clock.timeIncrement();
    scale_total_time += clock.elapsedTime();

    A.flops_ += nCells * (2 + 9);
}



// ************************************************************************* //
