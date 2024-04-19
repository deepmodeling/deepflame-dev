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

#include "CSRGAMGSolver.H"
#include "CSRGAMGInterface.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(CSRGAMGSolver, 0);

    csrMatrix::solver::addsymMatrixConstructorToTable<CSRGAMGSolver>
        addCSRGAMGSolverMatrixConstructorToTable_;

    csrMatrix::solver::addasymMatrixConstructorToTable<CSRGAMGSolver>
        addCSRGAMGAsymSolverMatrixConstructorToTable_;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::CSRGAMGSolver::CSRGAMGSolver
(
    const word& fieldName,
    const csrMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const dictionary& solverControls
)
:
    csrMatrix::solver
    (
        fieldName,
        matrix,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces,
        solverControls
    ),

    // Default values for all controls
    // which may be overridden by those in controlDict
    cacheAgglomeration_(true),
    nPreSweeps_(0),
    preSweepsLevelMultiplier_(1),
    maxPreSweeps_(4),
    nPostSweeps_(2),
    postSweepsLevelMultiplier_(1),
    maxPostSweeps_(4),
    nFinestSweeps_(2),
    nCoarsestSweeps_(0),
    interpolateCorrection_(false),
    scaleCorrection_(matrix.symmetric()),
    solveCoarsest_(true),
    directSolveCoarsest_(false),
    agglomeration_(CSRGAMGAgglomeration::New(matrix_.ldu(), controlDict_)),
    matrixLevels_(agglomeration_.size()),
    csrMatrixLevels_(agglomeration_.size()),
    primitiveInterfaceLevels_(agglomeration_.size()),
    interfaceLevels_(agglomeration_.size()),
    interfaceLevelsBouCoeffs_(agglomeration_.size()),
    interfaceLevelsIntCoeffs_(agglomeration_.size())
{
    readControls();

    Info << "CSRGAMGSolver settings :" << endl
        << "    Agglomeration Level:" << agglomeration_.size() << endl
        << "    cacheAgglomeration:" << cacheAgglomeration_ << endl
        << "    nPreSweeps:" << nPreSweeps_ << endl
        << "    preSweepsLevelMultiplier:" << preSweepsLevelMultiplier_ << endl
        << "    maxPreSweeps:" << maxPreSweeps_ << endl
        << "    nPostSweeps:" << nPostSweeps_ << endl
        << "    postSweepsLevelMultiplier:" << postSweepsLevelMultiplier_ << endl
        << "    maxPostSweeps:" << maxPostSweeps_ << endl
        << "    nFinestSweeps:" << nFinestSweeps_ << endl
        << "    nCoarsestSweeps:" << nCoarsestSweeps_ << endl
        << "    interpolateCorrection:" << interpolateCorrection_ << endl
        << "    scaleCorrection:" << scaleCorrection_ << endl
        << "    solveCoarsest:" << solveCoarsest_ << endl
        << "    directSolveCoarsest:" << directSolveCoarsest_ << endl
        << endl;

    if (agglomeration_.processorAgglomerate())
    {
        forAll(agglomeration_, fineLevelIndex)
        {
            if (agglomeration_.hasMeshLevel(fineLevelIndex))
            {
                if
                (
                    (fineLevelIndex+1) < agglomeration_.size()
                 && agglomeration_.hasProcMesh(fineLevelIndex+1)
                )
                {
                    // Construct matrix without referencing the coarse mesh so
                    // construct a dummy mesh instead. This will get overwritten
                    // by the call to procAgglomerateMatrix so is only to get
                    // it through agglomerateMatrix


                    const lduInterfacePtrsList& fineMeshInterfaces =
                        agglomeration_.interfaceLevel(fineLevelIndex);

                    PtrList<CSRGAMGInterface> dummyPrimMeshInterfaces
                    (
                        fineMeshInterfaces.size()
                    );
                    lduInterfacePtrsList dummyMeshInterfaces
                    (
                        dummyPrimMeshInterfaces.size()
                    );
                    forAll(fineMeshInterfaces, intI)
                    {
                        if (fineMeshInterfaces.set(intI))
                        {
                            OStringStream os;
                            refCast<const CSRGAMGInterface>
                            (
                                fineMeshInterfaces[intI]
                            ).write(os);
                            IStringStream is(os.str());

                            dummyPrimMeshInterfaces.set
                            (
                                intI,
                                CSRGAMGInterface::New
                                (
                                    fineMeshInterfaces[intI].type(),
                                    intI,
                                    dummyMeshInterfaces,
                                    is
                                )
                            );
                        }
                    }

                    forAll(dummyPrimMeshInterfaces, intI)
                    {
                        if (dummyPrimMeshInterfaces.set(intI))
                        {
                            dummyMeshInterfaces.set
                            (
                                intI,
                                &dummyPrimMeshInterfaces[intI]
                            );
                        }
                    }

                    // So:
                    // - pass in incorrect mesh (= fine mesh instead of coarse)
                    // - pass in dummy interfaces
                    agglomerateMatrix
                    (
                        fineLevelIndex,
                        agglomeration_.meshLevel(fineLevelIndex),
                        dummyMeshInterfaces
                    );


                    const labelList& procAgglomMap =
                        agglomeration_.procAgglomMap(fineLevelIndex+1);
                    const List<label>& procIDs =
                        agglomeration_.agglomProcIDs(fineLevelIndex+1);

                    procAgglomerateMatrix
                    (
                        procAgglomMap,
                        procIDs,
                        fineLevelIndex
                    );
                }
                else
                {
                    agglomerateMatrix
                    (
                        fineLevelIndex,
                        agglomeration_.meshLevel(fineLevelIndex + 1),
                        agglomeration_.interfaceLevel(fineLevelIndex + 1)
                    );
                }
            }
            else
            {
                // No mesh. Not involved in calculation anymore
            }
        }
    }
    else
    {
        forAll(agglomeration_, fineLevelIndex)
        {
            // Agglomerate on to coarse level mesh
            agglomerateMatrix
            (
                fineLevelIndex,
                agglomeration_.meshLevel(fineLevelIndex + 1),
                agglomeration_.interfaceLevel(fineLevelIndex + 1)
            );
        }
    }


    if (debug)
    {
        for
        (
            label fineLevelIndex = 0;
            fineLevelIndex <= matrixLevels_.size();
            fineLevelIndex++
        )
        {
            if (fineLevelIndex == 0 || matrixLevels_.set(fineLevelIndex-1))
            {
                const lduMatrix& matrix = matrixLevel(fineLevelIndex);
                const lduInterfaceFieldPtrsList& interfaces =
                    interfaceLevel(fineLevelIndex);

                Pout<< "level:" << fineLevelIndex << nl
                    << "    nCells:" << matrix.diag().size() << nl
                    << "    nFaces:" << matrix.lower().size() << nl
                    << "    nInterfaces:" << interfaces.size()
                    << endl;

                forAll(interfaces, i)
                {
                    if (interfaces.set(i))
                    {
                        Pout<< "        " << i
                            << "\ttype:" << interfaces[i].type()
                            << endl;
                    }
                }
            }
            else
            {
                Pout<< "level:" << fineLevelIndex << " : no matrix" << endl;
            }
        }
        Pout<< endl;
    }


    if (matrixLevels_.size())
    {
        if (directSolveCoarsest_)
        {
            const label coarsestLevel = matrixLevels_.size() - 1;

            if (matrixLevels_.set(coarsestLevel))
            {
                coarsestLUMatrixPtr_.set
                (
                    new LUscalarMatrix
                    (
                        matrixLevels_[coarsestLevel],
                        interfaceLevelsBouCoeffs_[coarsestLevel],
                        interfaceLevels_[coarsestLevel]
                    )
                );
            }
        }
    }
    else
    {
        FatalErrorInFunction
            << "No coarse levels created, either matrix too small for GAMG"
               " or nCellsInCoarsestLevel too large.\n"
               "    Either choose another solver of reduce "
               "nCellsInCoarsestLevel."
            << exit(FatalError);
    }

}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::CSRGAMGSolver::~CSRGAMGSolver()
{
    if (!cacheAgglomeration_)
    {
        delete &agglomeration_;
    }
}


// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

void Foam::CSRGAMGSolver::readControls()
{
    csrMatrix::solver::readControls();

    controlDict_.readIfPresent("cacheAgglomeration", cacheAgglomeration_);
    controlDict_.readIfPresent("nPreSweeps", nPreSweeps_);
    controlDict_.readIfPresent
    (
        "preSweepsLevelMultiplier",
        preSweepsLevelMultiplier_
    );
    controlDict_.readIfPresent("maxPreSweeps", maxPreSweeps_);
    controlDict_.readIfPresent("nPostSweeps", nPostSweeps_);
    controlDict_.readIfPresent
    (
        "postSweepsLevelMultiplier",
        postSweepsLevelMultiplier_
    );
    controlDict_.readIfPresent("maxPostSweeps", maxPostSweeps_);
    controlDict_.readIfPresent("nFinestSweeps", nFinestSweeps_);
    controlDict_.readIfPresent("nCoarsestSweeps", nCoarsestSweeps_);
    controlDict_.readIfPresent("interpolateCorrection", interpolateCorrection_);
    controlDict_.readIfPresent("scaleCorrection", scaleCorrection_);
    controlDict_.readIfPresent("solveCoarsest", solveCoarsest_);
    controlDict_.readIfPresent("directSolveCoarsest", directSolveCoarsest_);

    if (debug)
    {
        Pout<< "CSRGAMGSolver settings :" << endl
            << "    cacheAgglomeration:" << cacheAgglomeration_ << endl
            << "    nPreSweeps:" << nPreSweeps_ << endl
            << "    preSweepsLevelMultiplier:" << preSweepsLevelMultiplier_ << endl
            << "    maxPreSweeps:" << maxPreSweeps_ << endl
            << "    nPostSweeps:" << nPostSweeps_ << endl
            << "    postSweepsLevelMultiplier:" << postSweepsLevelMultiplier_ << endl
            << "    maxPostSweeps:" << maxPostSweeps_ << endl
            << "    nFinestSweeps:" << nFinestSweeps_ << endl
            << "    nCoarsestSweeps:" << nCoarsestSweeps_ << endl
            << "    interpolateCorrection:" << interpolateCorrection_ << endl
            << "    scaleCorrection:" << scaleCorrection_ << endl
            << "    solveCoarsest:" << solveCoarsest_ << endl
            << "    directSolveCoarsest:" << directSolveCoarsest_ << endl
            << endl;
    }
}


const Foam::lduMatrix& Foam::CSRGAMGSolver::matrixLevel(const label i) const
{
    if (i == 0)
    {
        return matrix_.ldu();
    }
    else
    {
        return matrixLevels_[i - 1];
    }
}


const Foam::lduInterfaceFieldPtrsList& Foam::CSRGAMGSolver::interfaceLevel
(
    const label i
) const
{
    if (i == 0)
    {
        return interfaces_;
    }
    else
    {
        return interfaceLevels_[i - 1];
    }
}


const Foam::FieldField<Foam::Field, Foam::scalar>&
Foam::CSRGAMGSolver::interfaceBouCoeffsLevel
(
    const label i
) const
{
    if (i == 0)
    {
        return interfaceBouCoeffs_;
    }
    else
    {
        return interfaceLevelsBouCoeffs_[i - 1];
    }
}


const Foam::FieldField<Foam::Field, Foam::scalar>&
Foam::CSRGAMGSolver::interfaceIntCoeffsLevel
(
    const label i
) const
{
    if (i == 0)
    {
        return interfaceIntCoeffs_;
    }
    else
    {
        return interfaceLevelsIntCoeffs_[i - 1];
    }
}


// ************************************************************************* //
