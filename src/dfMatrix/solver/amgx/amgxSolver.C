/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2018-2020 OpenCFD Ltd.
    Copyright (C) 2019-2020 Simone Bna
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

#include "fvMesh.H"
#include "fvMatrices.H"
#include "globalIndex.H"
//#include "PrecisionAdaptor.H"
#include "cyclicLduInterface.H"
#include "cyclicAMILduInterface.H"
#include "addToRunTimeSelectionTable.H"
#include "PstreamGlobals.H"

#include "amgxSolver.H"
#include "amgxLinearSolverContext.H"
#include "linearSolverContextTable.H"
//#include "petscControls.H"
//#include "petscWrappedVector.H"


// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(amgxSolver, 1);

    lduMatrix::solver::addsymMatrixConstructorToTable<amgxSolver>
        addamgxSolverSymMatrixConstructorToTable_;

    lduMatrix::solver::addasymMatrixConstructorToTable<amgxSolver>
        addamgxSolverAsymMatrixConstructorToTable_;
}


// * * * * * * * * * * * * * * * Local Functions * * * * * * * * * * * * * * //


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::amgxSolver::amgxSolver
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
    ),
    amgxDict_(solverControls.subDict("amgx")),
    eqName_(fieldName)
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //


Foam::solverPerformance Foam::amgxSolver::solve
(
    scalarField& psi,
    const scalarField& source,
    const direction cmpt
) const
{
    const fvMesh& fvm = dynamicCast<const fvMesh>(matrix_.mesh().thisDb());

    // Ensure PETSc is initialized
    //const petscControls& pcontrols = petscControls::New
    //(
    //    fvm
    //);

    //if (!pcontrols.valid())
    //{
    //   FatalErrorInFunction
    //        << "Could not initialize PETSc" << nl << abort(FatalError);
    //}

    dictionary amgxDictCaching = amgxDict_.subOrEmptyDict("caching");
   
    //const fvMesh& fvm = dynamicCast<const fvMesh>(matrix_.mesh().thisDb());

    const linearSolverContextTable<amgxLinearSolverContext>& contexts =
        linearSolverContextTable<amgxLinearSolverContext>::New(fvm);

    amgxLinearSolverContext& ctx = contexts.getContext(amgxDict_.lookup("dict"));

    if (!ctx.loaded())
    {
        FatalErrorInFunction
            << "Could not initialize AMGx" << nl << abort(FatalError);
    }

    ctx.caching.init(amgxDictCaching);
    ctx.caching.eventBegin();

    AmgXSolver& amgx = ctx.amgx;
    AmgXCSRMatrix& Amat = ctx.Amat;
    int nRowsLocal = 0;
    int nRowsGlobal = 0;
    int numberNonZeroes = 0;

    bool needsCacheUpdate = true;

    if (!ctx.initialized())
    {
        DebugInfo<< "Initializing AmgX Linear Solver " << eqName_ << nl;

        ctx.initialized() = true;
        
        needsCacheUpdate = false;

  
       
        amgx.initialiseMatrixComms(Amat);

        offloadMatrixArrays
        (
            Amat, 
            nRowsLocal, 
            nRowsGlobal, 
            numberNonZeroes
        );
    
       

        amgx.setOperator
        (
            nRowsLocal, 
            nRowsGlobal, 
            numberNonZeroes,
	    Amat 
        );


    }

    const bool matup = ctx.caching.needsMatrixUpdate();
    // const bool pcup = ctx.caching.needsPrecondUpdate();
    if (matup && needsCacheUpdate)
    {

    
        offloadMatrixValues
        (
            Amat, 
            nRowsLocal, 
            numberNonZeroes
        );


        
        amgx.updateOperator
        (
            nRowsLocal, 
            numberNonZeroes, 
	    Amat
        );
        

    }

    const word solverName("AMGx");

    // Setup class containing solver performance data
    solverPerformance solverPerf
    (
        solverName,
        fieldName_
    );

    // Retain copy of solverPerformance
    ctx.performance = solverPerf;

    // Create solution and rhs vectors for PETSc
    //PetscWrappedVector petsc_psi(psi); // Amat
    //PetscWrappedVector petsc_source(source); // Amat
    
    // Add monitor to record initial residual
    ctx.performance.initialResidual() = 0;

    // Solve A x = b
    //AMGX_vector_handle psi;
    //AMGX_vector_handle source;

    amgx.solve(nRowsLocal, &psi[0], &source[0], Amat);


    double irnorm = 0.;
    amgx.getResidual(0, irnorm);
    ctx.performance.initialResidual() = irnorm;

    // Set nIterations and final residual (from AMGx)
    int nIters = 0;
    amgx.getIters(nIters);
    ctx.performance.nIterations() = nIters - 1;

    double rnorm = 0.;
    amgx.getResidual(nIters - 1, rnorm);
    ctx.performance.finalResidual() = rnorm;

    ctx.caching.eventEnd();

    // Return solver performance to OpenFOAM
    return ctx.performance;
}

//Foam::solverPerformance Foam::amgxSolver::solve
//(
//    scalarField& psi_s,
//    const scalarField& source,
//    const direction cmpt
//) const
//{
//    PrecisionAdaptor<solveScalar, scalar> tpsi(psi_s);
//    return scalarSolve
//    (
//        tpsi.ref(),
//        ConstPrecisionAdaptor<solveScalar, scalar>(source)(),
//       cmpt
//    );
//}

void Foam::amgxSolver::offloadMatrixArrays
(
    AmgXCSRMatrix& Amat,
    int& nRowsLocal,
    int& nRowsGlobal,
    int& numberNonZeroes
) const
{
    const lduAddressing& lduAddr = matrix_.mesh().lduAddr();
    const lduInterfacePtrsList interfaces(matrix_.mesh().interfaces());

    const labelUList& upp = lduAddr.upperAddr();
    const labelUList& low = lduAddr.lowerAddr();

    const scalarField& diagVal = matrix_.diag();
    const scalarField& uppVal = matrix_.upper();
    const scalarField& lowVal =
    (
        matrix_.hasLower()
      ? matrix_.lower()
      : matrix_.upper()
    );

    // Local degrees-of-freedom i.e. number of local rows
    const label nrows_ = lduAddr.size();
    nRowsLocal = nrows_;
    nRowsGlobal = returnReduce(nRowsLocal, sumOp<label>());

    // Number of internal faces (connectivity)
    const label nIntFaces_ = upp.size();    

    const globalIndex globalNumbering_(nrows_);

    label diagIndexGlobal = globalNumbering_.toGlobal(0);
    label lowOffGlobal = globalNumbering_.toGlobal(low[0]) - low[0];
    label uppOffGlobal = globalNumbering_.toGlobal(upp[0]) - upp[0];

    labelList globalCells(nrows_);
    forAll(globalCells, celli)
    {
      globalCells[celli] = globalNumbering_.toGlobal(Pstream::myProcNo(), celli);
    }

    //    labelList globalCells
    //    (
    //        identity
    //        (
    //            globalNumbering_.localSize(),
    //            globalNumbering_.localStart()
    //        )
    //    );

    // Connections to neighbouring processors
    const label nReq = Pstream::nRequests();

    label nProcValues = 0;

    // Initialise transfer of global cells
    forAll(interfaces, patchi)
    {
        if (interfaces.set(patchi))
        {
            nProcValues += lduAddr.patchAddr(patchi).size();

            interfaces[patchi].initInternalFieldTransfer
            (
                Pstream::commsTypes::nonBlocking,
                globalCells
            );
        }
    }

    if (Pstream::parRun())
    {
        Pstream::waitRequests(nReq);
    }

    labelField procRows(nProcValues, 0);
    labelField procCols(nProcValues, 0);
    scalarField procVals(nProcValues, 0);
    nProcValues = 0;

    forAll(interfaces, patchi)
    {
        if (interfaces.set(patchi))
        {
            // Processor-local values
            const labelUList& faceCells = lduAddr.patchAddr(patchi);
            const scalarField& bCoeffs = interfaceBouCoeffs_[patchi];
            const label len = faceCells.size();

            labelField nbrCells
            (
                interfaces[patchi].internalFieldTransfer
                (
                    Pstream::commsTypes::nonBlocking,
                    globalCells
                )
            );

            if (faceCells.size() != nbrCells.size())
            {
                FatalErrorInFunction
                    << "Mismatch in interface sizes (AMI?)" << nl
                    << "Have " << faceCells.size() << " != "
                    << nbrCells.size() << nl
                    << exit(FatalError);
            }

            // for AMGx: Local rows, Global columns
            SubList<label>(procRows, len, nProcValues) = faceCells;
            SubList<label>(procCols, len, nProcValues) = nbrCells;
            SubList<scalar>(procVals, len, nProcValues) = bCoeffs;
            nProcValues += len;
        }
    }

    procVals.negate();  // Change sign for entire field (see previous note)
    
    // Determine the local non-zeros from the internal faces
    int localNonZeroes = nrows_ + 2 * nIntFaces_;

    // Add external non-zeros (communicated halo entries)
    numberNonZeroes = localNonZeroes + nProcValues;

    Amat.setValuesLDU
    (
        nrows_,
        nIntFaces_,
        diagIndexGlobal,
        lowOffGlobal,
        uppOffGlobal,
        &upp[0],
        &low[0],
        nProcValues,
        &procRows[0],
        &procCols[0],
        &diagVal[0],
        &uppVal[0],
        &lowVal[0],
        &procVals[0]
    );

    DebugInfo<< "Offloaded LDU matrix arrays on CUDA device and converted to CSR" << nl;
}


void Foam::amgxSolver::offloadMatrixValues
(
   AmgXCSRMatrix& Amat,
   int& nRowsLocal,
   int& numberNonZeroes
) const
{
    const lduAddressing& lduAddr = matrix_.mesh().lduAddr();
    const lduInterfacePtrsList interfaces(matrix_.mesh().interfaces());

    const labelUList& upp = lduAddr.upperAddr();
    // const labelUList& low = lduAddr.lowerAddr();

    const scalarField& diagVal = matrix_.diag();
    const scalarField& uppVal = matrix_.upper();
    const scalarField& lowVal =
    (
        matrix_.hasLower()
      ? matrix_.lower()
      : matrix_.upper()
    );

    // Local degrees-of-freedom i.e. number of local rows
    const label nrows_ = lduAddr.size();
    nRowsLocal = nrows_;   
 
    // Number of internal faces (connectivity)
    const label nIntFaces_ = upp.size();
    
    const globalIndex globalNumbering_(nrows_);

    labelList globalCells(nrows_);
    forAll(globalCells, celli)
    {
      globalCells[celli] = globalNumbering_.toGlobal(Pstream::myProcNo(), celli);
    }

    // Connections to neighbouring processors
    const label nReq = Pstream::nRequests();
    
    label nProcValues = 0;
    
    // Initialise transfer of global cells
    forAll(interfaces, patchi)
    {
        if (interfaces.set(patchi))
        {
            nProcValues += lduAddr.patchAddr(patchi).size();

            interfaces[patchi].initInternalFieldTransfer
            (
                Pstream::commsTypes::nonBlocking,
                globalCells
            );
        }
    }

    if (Pstream::parRun())
    {
        Pstream::waitRequests(nReq);
    }

    scalarField procVals(nProcValues, 0);
    nProcValues = 0;

    forAll(interfaces, patchi)
    {
        if (interfaces.set(patchi))
        {
            // Processor-local values
            const labelUList& faceCells = lduAddr.patchAddr(patchi);
            const scalarField& bCoeffs = interfaceBouCoeffs_[patchi];
            const label len = faceCells.size();

            SubList<scalar>(procVals, len, nProcValues) = bCoeffs;
            nProcValues += len;
        }
    }

    procVals.negate();  // Change sign for entire field (see previous note)

    // Determine the local non-zeros from the internal faces
    int localNonZeroes = nrows_ + 2 * nIntFaces_;
    
    // Add external non-zeros (communicated halo entries)
    numberNonZeroes = localNonZeroes + nProcValues;

    Amat.updateValues
    (
        nrows_,
        nIntFaces_,
        nProcValues,
        &diagVal[0],
        &uppVal[0],
        &lowVal[0],
        &procVals[0]
    );

    DebugInfo<< "Offloaded LDU matrix values (only) on CUDA device and converted to CSR" << nl;
}

