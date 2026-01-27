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

#include "dfMatrix.H"
#include <cassert>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include "PstreamGlobals.H"
#include "Residuals.H"
#include <vector>
#include "env.H"
#include "dfLduMatrix.H"
#include "dfCSRMatrix.H"
#include "dfBlockMatrix.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(dfMatrix, 1);
}

const Foam::label Foam::dfMatrix::solver::defaultMaxIter_ = 1000;

Foam::InnerMatrixFormat Foam::dfMatrix::getInnerMatrixTypeFromEnv(){
    if(env::DFMATRIX_INNERMATRIX_TYPE == "LDU"){
        return InnerMatrixFormat::DFMATRIX_INNERMATRIX_FORMAT_LDU;
    }else if(env::DFMATRIX_INNERMATRIX_TYPE == "CSR"){
        return InnerMatrixFormat::DFMATRIX_INNERMATRIX_FORMAT_CSR;
    }else if(env::DFMATRIX_INNERMATRIX_TYPE == "BLOCK_CSR"){
        return InnerMatrixFormat::DFMATRIX_INNERMATRIX_FORMAT_BLOCK_CSR;
    }else{
        SeriousError << "Invalid DFMATRIX_INNERMATRIX_TYPE: " << env::DFMATRIX_INNERMATRIX_TYPE << endl << flush;
        std::exit(1);
    }
}

Foam::dfMatrix::dfMatrix(const lduMatrix& ldu): lduMatrixPtr_(&ldu)
{
    InnerMatrixFormat format = getInnerMatrixTypeFromEnv();
    switch(format){
        case InnerMatrixFormat::DFMATRIX_INNERMATRIX_FORMAT_LDU:
            Info << "Building LDU matrix" << endl;
            innerMatrixPtr_ = std::make_shared<dfLduMatrix>(ldu);
            break;
        case InnerMatrixFormat::DFMATRIX_INNERMATRIX_FORMAT_CSR:
            Info << "Building CSR matrix" << endl;
            innerMatrixPtr_ = std::make_shared<dfCSRMatrix>(ldu);
            break;
        default:
            assert(false);
            break;
    }
}

Foam::dfMatrix::dfMatrix(const lduMatrix& ldu, const labelList& regionPtr): lduMatrixPtr_(&ldu)
{
    InnerMatrixFormat format = getInnerMatrixTypeFromEnv();
    switch(format){
        case InnerMatrixFormat::DFMATRIX_INNERMATRIX_FORMAT_LDU:
            Info << "Building LDU matrix" << endl;
            innerMatrixPtr_ = std::make_shared<dfLduMatrix>(ldu);
            break;
        case InnerMatrixFormat::DFMATRIX_INNERMATRIX_FORMAT_CSR:
            Info << "Building CSR matrix" << endl;
            innerMatrixPtr_ = std::make_shared<dfCSRMatrix>(ldu);
            break;
        case InnerMatrixFormat::DFMATRIX_INNERMATRIX_FORMAT_BLOCK_CSR:
            Info << "Building Block CSR matrix" << endl;
            innerMatrixPtr_ = std::make_shared<dfBlockMatrix>(ldu, regionPtr);
            break;
        default:
            assert(false);
            break;
    }
}

Foam::dfMatrix::dfMatrix(const lduMesh& mesh): lduMatrixPtr_(nullptr)
{
    InnerMatrixFormat format = getInnerMatrixTypeFromEnv();
    switch(format){
        case InnerMatrixFormat::DFMATRIX_INNERMATRIX_FORMAT_LDU:
            Info << "Building LDU matrix" << endl;
            innerMatrixPtr_ = std::make_shared<dfLduMatrix>(mesh);
            break;
        case InnerMatrixFormat::DFMATRIX_INNERMATRIX_FORMAT_CSR:
            Info << "Building CSR matrix" << endl;
            innerMatrixPtr_ = std::make_shared<dfCSRMatrix>(mesh);
            break;
        default:
            assert(false);
            break;
    }
}

Foam::dfMatrix::dfMatrix(const lduMesh& mesh, const labelList& regionPtr): lduMatrixPtr_(nullptr)
{
    InnerMatrixFormat format = getInnerMatrixTypeFromEnv();
    switch(format){
        case InnerMatrixFormat::DFMATRIX_INNERMATRIX_FORMAT_LDU:
            Info << "Building LDU matrix" << endl;
            innerMatrixPtr_ = std::make_shared<dfLduMatrix>(mesh);
            break;
        case InnerMatrixFormat::DFMATRIX_INNERMATRIX_FORMAT_CSR:
            Info << "Building CSR matrix" << endl;
            innerMatrixPtr_ = std::make_shared<dfCSRMatrix>(mesh);
            break;
        case InnerMatrixFormat::DFMATRIX_INNERMATRIX_FORMAT_BLOCK_CSR:
            Info << "Building Block CSR matrix" << endl;
            innerMatrixPtr_ = std::make_shared<dfBlockMatrix>(mesh, regionPtr);
            break;
        default:
            assert(false);
            break;
    }
}

// Foam::dfMatrix::dfMatrix(const lduMesh& courseLduMesh, const labelList& fineRowBlockPtr, const labelList& fineToCoarse): lduMatrixPtr_(nullptr)
// {
//     InnerMatrixFormat format = getInnerMatrixTypeFromEnv();
//     switch(format){
//         case InnerMatrixFormat::DFMATRIX_INNERMATRIX_FORMAT_LDU:
//             Info << "Building LDU matrix" << endl;
//             innerMatrixPtr_ = std::make_shared<dfLduMatrix>(courseLduMesh);
//             break;
//         case InnerMatrixFormat::DFMATRIX_INNERMATRIX_FORMAT_CSR:
//             Info << "Building CSR matrix" << endl;
//             innerMatrixPtr_ = std::make_shared<dfCSRMatrix>(courseLduMesh);
//             break;
//         case InnerMatrixFormat::DFMATRIX_INNERMATRIX_FORMAT_BLOCK_CSR:
//             Info << "Building Block CSR matrix" << endl;
//             innerMatrixPtr_ = std::make_shared<dfBlockMatrix>(courseLduMesh, fineRowBlockPtr, fineToCoarse);
//             break;
//         default:
//             // error:
//             SeriousError << "Invalid InnerMatrixFormat: " << format << endl << flush;
//             std::exit(1);
//     }
// }

void Foam::dfMatrix::valueCopy(lduMatrix& ldu){
    lduMatrixPtr_ = &ldu;
    innerMatrixPtr_->valueCopy(ldu);
}

