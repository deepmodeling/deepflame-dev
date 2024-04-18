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

#include "LDUGAMGAgglomeration.H"
#include "lduMesh.H"
#include "lduMatrix.H"
#include "Time.H"
#include "LDUGAMGInterface.H"
#include "LDUGAMGProcAgglomeration.H"
#include "pairLDUGAMGAgglomeration.H"
#include "IOmanip.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(LDUGAMGAgglomeration, 0);
    defineRunTimeSelectionTable(LDUGAMGAgglomeration, lduMesh);
    defineRunTimeSelectionTable(LDUGAMGAgglomeration, lduMatrix);
    defineRunTimeSelectionTable(LDUGAMGAgglomeration, geometry);
}


// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

void Foam::LDUGAMGAgglomeration::compactLevels(const label nCreatedLevels)
{
    nCells_.setSize(nCreatedLevels);
    restrictAddressing_.setSize(nCreatedLevels);
    nFaces_.setSize(nCreatedLevels);
    faceRestrictAddressing_.setSize(nCreatedLevels);
    faceFlipMap_.setSize(nCreatedLevels);
    nPatchFaces_.setSize(nCreatedLevels);
    patchFaceRestrictAddressing_.setSize(nCreatedLevels);
    meshLevels_.setSize(nCreatedLevels);

    // Have procCommunicator_ always, even if not procAgglomerating
    procCommunicator_.setSize(nCreatedLevels + 1);
    if (processorAgglomerate())
    {
        procAgglomMap_.setSize(nCreatedLevels);
        agglomProcIDs_.setSize(nCreatedLevels);
        procCellOffsets_.setSize(nCreatedLevels);
        procFaceMap_.setSize(nCreatedLevels);
        procBoundaryMap_.setSize(nCreatedLevels);
        procBoundaryFaceMap_.setSize(nCreatedLevels);

        procAgglomeratorPtr_().agglomerate();


    }

    // Print a bit
    if (processorAgglomerate() && debug)
    {
        Info<< "LDUGAMGAgglomeration:" << nl
            << "    local agglomerator     : " << type() << nl;
        if (processorAgglomerate())
        {
            Info<< "    processor agglomerator : "
                << procAgglomeratorPtr_().type() << nl
                << nl;
        }

        Info<< setw(36) << "nCells"
            << setw(20) << "nFaces/nCells"
            << setw(20) << "nInterfaces"
            << setw(20) << "nIntFaces/nCells"
            << setw(12) << "profile"
            << nl
            << setw(8) << "Level"
            << setw(8) << "nProcs"
            << "    "
            << setw(8) << "avg"
            << setw(8) << "max"
            << "    "
            << setw(8) << "avg"
            << setw(8) << "max"
            << "    "
            << setw(8) << "avg"
            << setw(8) << "max"
            << "    "
            << setw(8) << "avg"
            << setw(8) << "max"
            //<< "    "
            << setw(12) << "avg"
            << nl
            << setw(8) << "-----"
            << setw(8) << "------"
            << "    "
            << setw(8) << "---"
            << setw(8) << "---"
            << "    "
            << setw(8) << "---"
            << setw(8) << "---"
            << "    "
            << setw(8) << "---"
            << setw(8) << "---"
            << "    "
            << setw(8) << "---"
            << setw(8) << "---"
            //<< "    "
            << setw(12) << "---"
            //<< "    "
            << nl;

        for (label levelI = 0; levelI <= size(); levelI++)
        {
            label nProcs = 0;
            label nCells = 0;
            scalar faceCellRatio = 0;
            label nInterfaces = 0;
            label nIntFaces = 0;
            scalar ratio = 0.0;
            scalar profile = 0.0;

            if (hasMeshLevel(levelI))
            {
                nProcs = 1;

                const lduMesh& fineMesh = meshLevel(levelI);
                nCells = fineMesh.lduAddr().size();
                faceCellRatio =
                    scalar(fineMesh.lduAddr().lowerAddr().size())/nCells;

                const lduInterfacePtrsList interfaces =
                    fineMesh.interfaces();
                forAll(interfaces, i)
                {
                    if (interfaces.set(i))
                    {
                        nInterfaces++;
                        nIntFaces += interfaces[i].faceCells().size();
                    }
                }
                ratio = scalar(nIntFaces)/nCells;

                profile = fineMesh.lduAddr().band().second();
            }

            label totNprocs = returnReduce(nProcs, sumOp<label>());

            label maxNCells = returnReduce(nCells, maxOp<label>());
            label totNCells = returnReduce(nCells, sumOp<label>());

            scalar maxFaceCellRatio =
                returnReduce(faceCellRatio, maxOp<scalar>());
            scalar totFaceCellRatio =
                returnReduce(faceCellRatio, sumOp<scalar>());

            label maxNInt = returnReduce(nInterfaces, maxOp<label>());
            label totNInt = returnReduce(nInterfaces, sumOp<label>());

            scalar maxRatio = returnReduce(ratio, maxOp<scalar>());
            scalar totRatio = returnReduce(ratio, sumOp<scalar>());

            scalar totProfile = returnReduce(profile, sumOp<scalar>());

            int oldPrecision = Info().precision(4);

            Info<< setw(8) << levelI
                << setw(8) << totNprocs
                << "    "
                << setw(8) << totNCells/totNprocs
                << setw(8) << maxNCells
                << "    "
                << setw(8) << totFaceCellRatio/totNprocs
                << setw(8) << maxFaceCellRatio
                << "    "
                << setw(8) << scalar(totNInt)/totNprocs
                << setw(8) << maxNInt
                << "    "
                << setw(8) << totRatio/totNprocs
                << setw(8) << maxRatio
                << setw(12) << totProfile/totNprocs
                << nl;

            Info().precision(oldPrecision);
        }
        Info<< endl;
    }
}


bool Foam::LDUGAMGAgglomeration::continueAgglomerating
(
    const label nFineCells,
    const label nCoarseCells
) const
{
    const label nTotalCoarseCells = returnReduce(nCoarseCells, sumOp<label>());
    if (nTotalCoarseCells < Pstream::nProcs()*nCellsInCoarsestLevel_)
    {
        return false;
    }
    else
    {
        const label nTotalFineCells = returnReduce(nFineCells, sumOp<label>());
        return nTotalCoarseCells < nTotalFineCells;
    }
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::LDUGAMGAgglomeration::LDUGAMGAgglomeration
(
    const lduMesh& mesh,
    const dictionary& controlDict
)
:
    MeshObject<lduMesh, Foam::GeometricMeshObject, LDUGAMGAgglomeration>(mesh),
    maxLevels_(
        controlDict.lookupOrDefault<label>("maxLevels", 50)
    ),
    nCellsInCoarsestLevel_
    (
        controlDict.lookupOrDefault<label>("nCellsInCoarsestLevel", 10)
    ),
    meshInterfaces_(mesh.interfaces()),
    procAgglomeratorPtr_
    (
        (
            (UPstream::nProcs(mesh.comm()) > 1)
         && controlDict.found("processorAgglomerator")
        )
      ? LDUGAMGProcAgglomeration::New
        (
            controlDict.lookup("processorAgglomerator"),
            *this,
            controlDict
        )
      : autoPtr<LDUGAMGProcAgglomeration>(nullptr)
    ),

    nCells_(maxLevels_),
    restrictAddressing_(maxLevels_),
    nFaces_(maxLevels_),
    faceRestrictAddressing_(maxLevels_),
    faceFlipMap_(maxLevels_),
    nPatchFaces_(maxLevels_),
    patchFaceRestrictAddressing_(maxLevels_),

    meshLevels_(maxLevels_)
{
    procCommunicator_.setSize(maxLevels_ + 1, -1);
    if (processorAgglomerate())
    {
        procAgglomMap_.setSize(maxLevels_);
        agglomProcIDs_.setSize(maxLevels_);
        procCellOffsets_.setSize(maxLevels_);
        procFaceMap_.setSize(maxLevels_);
        procBoundaryMap_.setSize(maxLevels_);
        procBoundaryFaceMap_.setSize(maxLevels_);
    }
}


const Foam::LDUGAMGAgglomeration& Foam::LDUGAMGAgglomeration::New
(
    const lduMesh& mesh,
    const dictionary& controlDict
)
{
    if
    (
        !mesh.thisDb().foundObject<LDUGAMGAgglomeration>
        (
            LDUGAMGAgglomeration::typeName
        )
    )
    {
        const word agglomeratorType
        (
            controlDict.lookupOrDefault<word>("agglomerator", "faceAreaPair")
        );

        const_cast<Time&>(mesh.thisDb().time()).libs().open
        (
            controlDict,
            "geometricLDUGAMGAgglomerationLibs",
            lduMeshConstructorTablePtr_
        );

        lduMeshConstructorTable::iterator cstrIter =
            lduMeshConstructorTablePtr_->find(agglomeratorType);

        if (cstrIter == lduMeshConstructorTablePtr_->end())
        {
            FatalErrorInFunction
                << "Unknown LDUGAMGAgglomeration type "
                << agglomeratorType << ".\n"
                << "Valid matrix LDUGAMGAgglomeration types are :"
                << lduMatrixConstructorTablePtr_->sortedToc() << endl
                << "Valid geometric LDUGAMGAgglomeration types are :"
                << lduMeshConstructorTablePtr_->sortedToc()
                << exit(FatalError);
        }

        return store(cstrIter()(mesh, controlDict).ptr());
    }
    else
    {
        return mesh.thisDb().lookupObject<LDUGAMGAgglomeration>
        (
            LDUGAMGAgglomeration::typeName
        );
    }
}


const Foam::LDUGAMGAgglomeration& Foam::LDUGAMGAgglomeration::New
(
    const lduMatrix& matrix,
    const dictionary& controlDict
)
{
    const lduMesh& mesh = matrix.mesh();

    if
    (
        !mesh.thisDb().foundObject<LDUGAMGAgglomeration>
        (
            LDUGAMGAgglomeration::typeName
        )
    )
    {
        const word agglomeratorType
        (
            controlDict.lookupOrDefault<word>("agglomerator", "faceAreaPair")
        );

        const_cast<Time&>(mesh.thisDb().time()).libs().open
        (
            controlDict,
            "algebraicLDUGAMGAgglomerationLibs",
            lduMatrixConstructorTablePtr_
        );

        if
        (
            !lduMatrixConstructorTablePtr_
         || !lduMatrixConstructorTablePtr_->found(agglomeratorType)
        )
        {
            return New(mesh, controlDict);
        }
        else
        {
            lduMatrixConstructorTable::iterator cstrIter =
                lduMatrixConstructorTablePtr_->find(agglomeratorType);

            return store(cstrIter()(matrix, controlDict).ptr());
        }
    }
    else
    {
        return mesh.thisDb().lookupObject<LDUGAMGAgglomeration>
        (
            LDUGAMGAgglomeration::typeName
        );
    }
}


Foam::autoPtr<Foam::LDUGAMGAgglomeration> Foam::LDUGAMGAgglomeration::New
(
    const lduMesh& mesh,
    const scalarField& cellVolumes,
    const vectorField& faceAreas,
    const dictionary& controlDict
)
{
    const word agglomeratorType
    (
        controlDict.lookupOrDefault<word>("agglomerator", "faceAreaPair")
    );

    const_cast<Time&>(mesh.thisDb().time()).libs().open
    (
        controlDict,
        "geometricLDUGAMGAgglomerationLibs",
        geometryConstructorTablePtr_
    );

    geometryConstructorTable::iterator cstrIter =
        geometryConstructorTablePtr_->find(agglomeratorType);

    if (cstrIter == geometryConstructorTablePtr_->end())
    {
        FatalErrorInFunction
            << "Unknown LDUGAMGAgglomeration type "
            << agglomeratorType << ".\n"
            << "Valid geometric LDUGAMGAgglomeration types are :"
            << geometryConstructorTablePtr_->sortedToc()
            << exit(FatalError);
    }

    return autoPtr<LDUGAMGAgglomeration>
    (
        cstrIter()
        (
            mesh,
            cellVolumes,
            faceAreas,
            controlDict
        )
    );
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::LDUGAMGAgglomeration::~LDUGAMGAgglomeration()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

const Foam::lduMesh& Foam::LDUGAMGAgglomeration::meshLevel
(
    const label i
) const
{
    if (i == 0)
    {
        return mesh_;
    }
    else
    {
        return meshLevels_[i - 1];
    }
}


bool Foam::LDUGAMGAgglomeration::hasMeshLevel(const label i) const
{
    if (i == 0)
    {
        return true;
    }
    else
    {
        return meshLevels_.set(i - 1);
    }
}


const Foam::lduInterfacePtrsList& Foam::LDUGAMGAgglomeration::interfaceLevel
(
    const label i
) const
{
    if (i == 0)
    {
        return meshInterfaces_;
    }
    else
    {
        return meshLevels_[i - 1].rawInterfaces();
    }
}


void Foam::LDUGAMGAgglomeration::clearLevel(const label i)
{
    if (hasMeshLevel(i))
    {
        meshLevels_.set(i - 1, nullptr);

        if (i < nCells_.size())
        {
            nCells_[i] = -555;
            restrictAddressing_.set(i, nullptr);
            nFaces_[i] = -666;
            faceRestrictAddressing_.set(i, nullptr);
            faceFlipMap_.set(i, nullptr);
            nPatchFaces_.set(i, nullptr);
            patchFaceRestrictAddressing_.set(i, nullptr);
        }
    }
}


const Foam::labelList& Foam::LDUGAMGAgglomeration::procAgglomMap
(
    const label leveli
) const
{
    return procAgglomMap_[leveli];
}


const Foam::labelList& Foam::LDUGAMGAgglomeration::agglomProcIDs
(
    const label leveli
) const
{
    return agglomProcIDs_[leveli];
}


bool Foam::LDUGAMGAgglomeration::hasProcMesh(const label leveli) const
{
    return procCommunicator_[leveli] != -1;
}


Foam::label Foam::LDUGAMGAgglomeration::procCommunicator(const label leveli) const
{
    return procCommunicator_[leveli];
}


const Foam::labelList& Foam::LDUGAMGAgglomeration::cellOffsets
(
    const label leveli
) const
{
    return procCellOffsets_[leveli];
}


const Foam::labelListList& Foam::LDUGAMGAgglomeration::faceMap
(
    const label leveli
) const
{
    return procFaceMap_[leveli];
}


const Foam::labelListList& Foam::LDUGAMGAgglomeration::boundaryMap
(
    const label leveli
) const
{
    return procBoundaryMap_[leveli];
}


const Foam::labelListListList& Foam::LDUGAMGAgglomeration::boundaryFaceMap
(
    const label leveli
) const
{
    return procBoundaryFaceMap_[leveli];
}


bool Foam::LDUGAMGAgglomeration::checkRestriction
(
    labelList& newRestrict,
    label& nNewCoarse,
    const lduAddressing& fineAddressing,
    const labelUList& restrict,
    const label nCoarse
)
{
    if (fineAddressing.size() != restrict.size())
    {
        FatalErrorInFunction
            << "nCells:" << fineAddressing.size()
            << " agglom:" << restrict.size()
            << abort(FatalError);
    }

    // Seed (master) for every region
    labelList master(identity(fineAddressing.size()));

    // Now loop and transport master through region
    const labelUList& lower = fineAddressing.lowerAddr();
    const labelUList& upper = fineAddressing.upperAddr();

    while (true)
    {
        label nChanged = 0;

        forAll(lower, facei)
        {
            label own = lower[facei];
            label nei = upper[facei];

            if (restrict[own] == restrict[nei])
            {
                // coarse-mesh-internal face

                if (master[own] < master[nei])
                {
                    master[nei] = master[own];
                    nChanged++;
                }
                else if (master[own] > master[nei])
                {
                    master[own] = master[nei];
                    nChanged++;
                }
            }
        }

        reduce(nChanged, sumOp<label>());

        if (nChanged == 0)
        {
            break;
        }
    }


    // Count number of regions/masters per coarse cell
    labelListList coarseToMasters(nCoarse);
    nNewCoarse = 0;
    forAll(restrict, celli)
    {
        labelList& masters = coarseToMasters[restrict[celli]];

        if (findIndex(masters, master[celli]) == -1)
        {
            masters.append(master[celli]);
            nNewCoarse++;
        }
    }

    if (nNewCoarse > nCoarse)
    {
        // WarningInFunction
        //    << "Have " << nCoarse
        //    << " agglomerated cells but " << nNewCoarse
        //    << " disconnected regions" << endl;

        // Keep coarseToMasters[0] the original coarse, allocate new ones
        // for the others
        labelListList coarseToNewCoarse(coarseToMasters.size());

        nNewCoarse = nCoarse;

        forAll(coarseToMasters, coarseI)
        {
            const labelList& masters = coarseToMasters[coarseI];

            labelList& newCoarse = coarseToNewCoarse[coarseI];
            newCoarse.setSize(masters.size());
            newCoarse[0] = coarseI;
            for (label i=1; i<newCoarse.size(); i++)
            {
                newCoarse[i] = nNewCoarse++;
            }
        }

        newRestrict.setSize(fineAddressing.size());
        forAll(restrict, celli)
        {
            label coarseI = restrict[celli];

            label index = findIndex(coarseToMasters[coarseI], master[celli]);
            newRestrict[celli] = coarseToNewCoarse[coarseI][index];
        }

        return false;
    }
    else
    {
        return true;
    }
}


// ************************************************************************* //
