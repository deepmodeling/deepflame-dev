/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2016-2018 OpenFOAM Foundation
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

#include "NBPBiCGStab.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(NBPBiCGStab, 0);

    lduMatrix::solver::addsymMatrixConstructorToTable<NBPBiCGStab>
        addNBPBiCGStabSymMatrixConstructorToTable_;

    lduMatrix::solver::addasymMatrixConstructorToTable<NBPBiCGStab>
        addNBPBiCGStabAsymMatrixConstructorToTable_;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::NBPBiCGStab::NBPBiCGStab
(
    const word& fieldName,
    const lduMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const dictionary& solverControls
)
:
    lduMatrix::solver
    (
        fieldName,
        matrix,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces,
        solverControls
    )
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::solverPerformance Foam::NBPBiCGStab::solve
(
    scalarField& psi,
    const scalarField& source,
    const direction cmpt
) const
{
    // --- Setup class containing solver performance data
    solverPerformance solverPerf
    (
        lduMatrix::preconditioner::getName(controlDict_) + typeName,
        fieldName_
    );

    const label nCells = psi.size();

    scalar* __restrict__ psiPtr = psi.begin();

    scalarField pA(nCells,0);

    scalarField yA(nCells,0);
    scalar* __restrict__ yAPtr = yA.begin();

    scalarField rM(nCells,0);
    scalar* __restrict__ rMPtr = rM.begin();

    scalarField wA(nCells,0);
    scalar* __restrict__ wAPtr = wA.begin();

    scalarField wM(nCells,0);
    scalar* __restrict__ wMPtr = wM.begin();

    scalarField tA(nCells,0);
    scalar* __restrict__ tAPtr = tA.begin();

    // --- Calculate A.psi
    matrix_.Amul(yA, psi, interfaceBouCoeffs_, interfaces_, cmpt);

    // --- Calculate initial residual field
    scalarField rA(source - yA);
    scalar* __restrict__ rAPtr = rA.begin();

    scalarField r0(rA);

    // --- Calculate normalisation factor
    const scalar normFactor = this->normFactor(psi, source, yA, pA);

    // --- Calculate normalised residual norm
    solverPerf.initialResidual() =
        gSumMag(rA, matrix().mesh().comm())
       /normFactor;
    solverPerf.finalResidual() = solverPerf.initialResidual();

    // --- Select and construct the preconditioner
    autoPtr<lduMatrix::preconditioner> preconPtr =
    lduMatrix::preconditioner::New
    (
        *this,
        controlDict_
    );
    preconPtr->precondition(rM, rA, cmpt);
    matrix_.Amul(wA, rM, interfaceBouCoeffs_, interfaces_, cmpt);
    preconPtr->precondition(wM, wA, cmpt);
    matrix_.Amul(tA, wM, interfaceBouCoeffs_, interfaces_, cmpt);

    // initialize alpha & beta
    const scalar rArA = gSumSqr(rA, matrix().mesh().comm());
    scalar alpha = rArA/gSumProd(rA, wA, matrix().mesh().comm());
    scalar beta = 0;
    scalar omega = 0;
    scalar r0rA = rArA;


    // --- Check convergence, solve if not converged
    if
    (
        minIter_ > 0
     || !solverPerf.checkConvergence(tolerance_, relTol_)
    )
    {
        // Initialize parameters
        scalarField pM(nCells,0);
        scalar* __restrict__ pMPtr = pM.begin();

        scalarField sM(nCells,0);
        scalar* __restrict__ sMPtr = sM.begin();

        scalarField sA(nCells,0);
        scalar* __restrict__ sAPtr = sA.begin();

        scalarField zA(nCells,0);
        scalar* __restrict__ zAPtr = zA.begin();

        scalarField zM(nCells,0);
        scalar* __restrict__ zMPtr = zM.begin();

        scalarField vA(nCells,0);
        scalar* __restrict__ vAPtr = vA.begin();

        scalarField qA(nCells,0);
        scalar* __restrict__ qAPtr = qA.begin();

        scalarField qM(nCells,0);
        scalar* __restrict__ qMPtr = qM.begin();

        // --- Solver iteration
        do
        {
            for (label cell=0; cell<nCells; cell++)
            {
                pMPtr[cell] = rMPtr[cell] + beta*(pMPtr[cell] - omega*sMPtr[cell]);
                sAPtr[cell] = wAPtr[cell] + beta*(sAPtr[cell] - omega*zAPtr[cell]);
                sMPtr[cell] = wMPtr[cell] + beta*(sMPtr[cell] - omega*zMPtr[cell]);
                zAPtr[cell] = tAPtr[cell] + beta*(zAPtr[cell] - omega*vAPtr[cell]);
                qAPtr[cell] = rAPtr[cell] - alpha*sAPtr[cell];
                qMPtr[cell] = rMPtr[cell] - alpha*sMPtr[cell];
                yAPtr[cell] = wAPtr[cell] - alpha*zAPtr[cell];
            }

            // --- overlap comm and comp
            // 1. no blocking reduction 
            double allredSend1[3];
            double allredRecv1[3];
            allredSend1[0] = sumProd(qA, yA);
            allredSend1[1] = sumSqr(yA);
            allredSend1[2] = sumMag(qA);
            MPI_Request reqs1;
            MPI_Iallreduce
            (
                &allredSend1,
                &allredRecv1,
                3,
                MPI_DOUBLE,
                MPI_SUM,
                PstreamGlobals::MPICommunicators_[0],
                &reqs1
            );
            // 2. SPMV
            preconPtr->precondition(zM, zA, cmpt);
            matrix_.Amul(vA, zM, interfaceBouCoeffs_, interfaces_, cmpt);
            // 3. MPI_wait
            MPI_Wait(&reqs1, MPI_STATUS_IGNORE);

            // check convergence
            solverPerf.finalResidual() = allredRecv1[2] / normFactor;
            if (solverPerf.checkConvergence(tolerance_, relTol_))
            {
                for (label cell=0; cell<nCells; cell++)
                {
                    psiPtr[cell] += alpha*pMPtr[cell];
                }

                solverPerf.nIterations()++;

                return solverPerf;
            }
            
            omega = allredRecv1[0] / allredRecv1[1];

            for (label cell=0; cell<nCells; cell++)
            {
                psiPtr[cell] += alpha*pMPtr[cell] + omega*qMPtr[cell];
                rAPtr[cell] = qAPtr[cell] - omega*yAPtr[cell];
                rMPtr[cell] = qMPtr[cell] - omega*(wMPtr[cell] - alpha*zMPtr[cell]);
                wAPtr[cell] = yAPtr[cell] - omega*(tAPtr[cell] - alpha*vAPtr[cell]);
            }

            // --- overlap comm and comp
            // 1. no blocking reduction 
            double allredSend2[5];
            double allredRecv2[5];
            allredSend2[0] = sumProd(r0, rA);
            allredSend2[1] = sumProd(r0, wA);
            allredSend2[2] = sumProd(r0, sA);
            allredSend2[3] = sumProd(r0, zA);
            allredSend2[4] = sumMag(rA);
            MPI_Request reqs2;
            MPI_Iallreduce
            (
                &allredSend2,
                &allredRecv2,
                5,
                MPI_DOUBLE,
                MPI_SUM,
                PstreamGlobals::MPICommunicators_[0],
                &reqs2
            );
            // 2. SPMV
            preconPtr->precondition(wM, wA, cmpt);
            matrix_.Amul(tA, wM, interfaceBouCoeffs_, interfaces_, cmpt);
            // 3. MPI_wait
            MPI_Wait(&reqs2, MPI_STATUS_IGNORE);

            // check convergence
            solverPerf.finalResidual() = allredRecv2[4]/normFactor;

            if (solverPerf.checkConvergence(tolerance_, relTol_))
            {
                solverPerf.nIterations()++;

                return solverPerf;
            }

            beta = (alpha/omega)*allredRecv2[0]/r0rA;
            alpha = allredRecv2[0]/(allredRecv2[1] + beta*allredRecv2[2] - beta*omega*allredRecv2[3]);
            r0rA = allredRecv2[0];

        } while
        (
            ++solverPerf.nIterations() < maxIter_ || solverPerf.nIterations() < minIter_
        );
    }

    return solverPerf;
}


// ************************************************************************* //
