/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2012-2018 OpenFOAM Foundation
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

#include "csrJacobiSmoother.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(csrJacobiSmoother, 0);

    csrMatrix::smoother::addsymMatrixConstructorToTable<csrJacobiSmoother>
        addcsrJacobiSmootherSymMatrixConstructorToTable_;

    csrMatrix::smoother::addasymMatrixConstructorToTable<csrJacobiSmoother>
        addcsrJacobiSmootherAsymMatrixConstructorToTable_;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::csrJacobiSmoother::csrJacobiSmoother
(
    const word& fieldName,
    const csrMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces
)
:
    csrMatrix::smoother
    (
        fieldName,
        matrix,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces
    )
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::csrJacobiSmoother::smooth
(
    const word& fieldName_,
    scalarField& psi,
    const csrMatrix& matrix_,
    const scalarField& source,
    const FieldField<Field, scalar>& interfaceBouCoeffs_,
    const lduInterfaceFieldPtrsList& interfaces_,
    const direction cmpt,
    const label nSweeps
)
{
    const label nCells = psi.size();
    scalarField bPrime(nCells);
    scalarField psiCopy(nCells);

    scalar* __restrict__ psiPtr = psi.begin();

    scalar* __restrict__ bPrimePtr = bPrime.begin();
    scalar* __restrict__ psiCopyPtr = psiCopy.begin();
    const scalar* const __restrict__ diagPtr = matrix_.diag().begin();


    // Parallel boundary initialisation.  The parallel boundary is treated
    // as an effective jacobi interface in the boundary.
    // Note: there is a change of sign in the coupled
    // interface update.  The reason for this is that the
    // internal coefficients are all located at the l.h.s. of
    // the matrix whereas the "implicit" coefficients on the
    // coupled boundaries are all created as if the
    // coefficient contribution is of a source-kind (i.e. they
    // have a sign as if they are on the r.h.s. of the matrix.
    // To compensate for this, it is necessary to turn the
    // sign of the contribution.

    FieldField<Field, scalar>& mBouCoeffs =
        const_cast<FieldField<Field, scalar>&>
        (
            interfaceBouCoeffs_
        );

    forAll(mBouCoeffs, patchi)
    {
        if (interfaces_.set(patchi))
        {
            mBouCoeffs[patchi].negate();
        }
    }


    for (label sweep=0; sweep<nSweeps; sweep++)
    {
        bPrime = source;

        matrix_.initMatrixInterfaces
        (
            mBouCoeffs,
            interfaces_,
            psi,
            bPrime,
            cmpt
        );

        matrix_.updateMatrixInterfaces
        (
            mBouCoeffs,
            interfaces_,
            psi,
            bPrime,
            cmpt
        );

        const label* const __restrict__ off_diag_rowptr_Ptr = matrix_.off_diag_rowptr().begin();
        const label* const __restrict__ off_diag_colidx_Ptr = matrix_.off_diag_colidx().begin();
        const scalar* const __restrict__ off_diag_value_Ptr = matrix_.off_diag_value().begin();

        #pragma omp parallel 
        {
            #pragma omp for
            for(label r = 0; r < nCells; ++r){
                psiCopyPtr[r] = psi[r];
            }
            #pragma omp for
            for(label r = 0; r < nCells; ++r){
                scalar sum = bPrimePtr[r];
                for(label index = off_diag_rowptr_Ptr[r]; index < off_diag_rowptr_Ptr[r+1]; ++index){
                    sum -= off_diag_value_Ptr[index] * psiCopyPtr[off_diag_colidx_Ptr[index]];
                }
                psiPtr[r] = sum / diagPtr[r];
            }
        }
    }

    // Restore interfaceBouCoeffs_
    forAll(mBouCoeffs, patchi)
    {
        if (interfaces_.set(patchi))
        {
            mBouCoeffs[patchi].negate();
        }
    }
}


void Foam::csrJacobiSmoother::smooth
(
    scalarField& psi,
    const scalarField& source,
    const direction cmpt,
    const label nSweeps
) const
{
    smooth
    (
        fieldName_,
        psi,
        matrix_,
        source,
        interfaceBouCoeffs_,
        interfaces_,
        cmpt,
        nSweeps
    );
}


// ************************************************************************* //
