/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2019 OpenFOAM Foundation
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

Application
    renumberMesh

Description
    Renumbers the cell list in order to reduce the bandwidth, reading and
    renumbering all renumberFields from all the time directories.

    By default uses bandCompression (CuthillMcKee) but will
    read system/renumberMeshDict if -dict option is present

\*---------------------------------------------------------------------------*/
// {

argList::addNote
(
    "Renumber mesh to minimize bandwidth"
);
#include "addRegionOption.H"
#include "addOverwriteOption.H"
#include "addTimeOptions.H"
#include "addDictOption.H"
argList::addBoolOption
(
    "frontWidth",
    "calculate the rms of the frontwidth"
);
argList::addBoolOption
(
    "noFields",
    "do not update renumberFields"
);

// #include "setRootCase.H"
// #include "createTime.H"
// runTime.functionObjects().off();

// Force linker to include zoltan symbols. This section is only needed since
// Zoltan is a static library
#ifdef FOAM_USE_ZOLTAN
    Info<< "renumberMesh built with zoltan support." << nl << endl;
    (void)zoltanRenumber::typeName;
#endif

// // Get times list
// instantList Times = runTime.times();

// // Set startTime and endTime depending on -time and -latestTime options
// #include "checkTimeOptions.H"

// runTime.setTime(Times[startTime], startTime);

// #include "createNamedMesh.H"
const word oldInstance = mesh.pointsInstance();

const bool readDict = true;
const bool doFrontWidth = args.optionFound("frontWidth");
const bool overwrite = false;
const bool renumberFields = !args.optionFound("noFields");

label band;
scalar profile;
scalar sumSqrIntersect;
getBand
(
    doFrontWidth,
    mesh.nCells(),
    mesh.faceOwner(),
    mesh.faceNeighbour(),
    band,
    profile,
    sumSqrIntersect
);

reduce(band, maxOp<label>());
reduce(profile, sumOp<scalar>());
scalar rmsFrontwidth = Foam::sqrt
(
    returnReduce
    (
        sumSqrIntersect,
        sumOp<scalar>()
    )/mesh.globalData().nTotalCells()
);

Info<< "Mesh size: " << mesh.globalData().nTotalCells() << nl
    << "Before renumbering :" << nl
    << "    band           : " << band << nl
    << "    profile        : " << profile << nl;

if (doFrontWidth)
{
    Info<< "    rms frontwidth : " << rmsFrontwidth << nl;
}

Info<< endl;

bool sortCoupledFaceCells = false;
bool writeMaps = false;
bool orderPoints = false;
label nBlocks = 1;
label blockSize = 0;
bool renumberSets = true;
labelList regionPtr;

// Construct renumberMethod
autoPtr<IOdictionary> renumberDictPtr;
autoPtr<renumberMethod> renumberPtr;



if (readDict)
{
    const word dictName("renumberMeshDict");
    #include "setSystemMeshDictionaryIO.H"

    renumberDictPtr.reset(new IOdictionary(dictIO));
    const IOdictionary& renumberDict = renumberDictPtr();

    renumberPtr = renumberMethod::New(renumberDict);

    sortCoupledFaceCells = renumberDict.lookupOrDefault
    (
        "sortCoupledFaceCells",
        false
    );
    if (sortCoupledFaceCells)
    {
        Info<< "Sorting cells on coupled boundaries to be last." << nl
            << endl;
    }

    blockSize = renumberDict.lookupOrDefault("blockSize", 0);
    nBlocks = renumberDict.lookupOrDefault("nBlocks", 1); 
    // if (blockSize > 0)
    // {
    //     Info<< "Ordering cells into regions of size " << blockSize
    //         << " (using decomposition);"
    //         << " ordering faces into region-internal and region-external."
    //         << nl << endl;

    //     if (blockSize < 0 || blockSize >= mesh.nCells())
    //     {
    //         FatalErrorInFunction
    //             << "Block size " << blockSize
    //             << " should be positive integer"
    //             << " and less than the number of cells in the mesh."
    //             << exit(FatalError);
    //     }
    // }

    orderPoints = renumberDict.lookupOrDefault("orderPoints", false);
    if (orderPoints)
    {
        Info<< "Ordering points into internal and boundary points." << nl
            << endl;
    }

    writeMaps = renumberDict.lookupOrDefault("writeMaps", false);
    if (writeMaps)
    {
        Info<< "Writing renumber maps (new to old) to polyMesh." << nl
            << endl;
    }

    renumberSets = renumberDict.lookupOrDefault("renumberSets", true);
}
else
{
    Info<< "Using default renumberMethod." << nl << endl;
    dictionary renumberDict;
    renumberPtr.reset(new CuthillMcKeeRenumber(renumberDict));
}

Info<< "Selecting renumberMethod : " << renumberPtr().type() << nl << endl;


// Read parallel reconstruct maps
labelIOList cellProcAddressing
(
    IOobject
    (
        "cellProcAddressing",
        mesh.facesInstance(),
        polyMesh::meshSubDir,
        mesh,
        IOobject::READ_IF_PRESENT
    ),
    labelList(0)
);

labelIOList faceProcAddressing
(
    IOobject
    (
        "faceProcAddressing",
        mesh.facesInstance(),
        polyMesh::meshSubDir,
        mesh,
        IOobject::READ_IF_PRESENT
    ),
    labelList(0)
);
labelIOList pointProcAddressing
(
    IOobject
    (
        "pointProcAddressing",
        mesh.pointsInstance(),
        polyMesh::meshSubDir,
        mesh,
        IOobject::READ_IF_PRESENT
    ),
    labelList(0)
);
labelIOList boundaryProcAddressing
(
    IOobject
    (
        "boundaryProcAddressing",
        mesh.pointsInstance(),
        polyMesh::meshSubDir,
        mesh,
        IOobject::READ_IF_PRESENT
    ),
    labelList(0)
);


// Read objects in time directory
// IOobjectList objects(mesh, runTime.timeName());
// forAll(objects.sortedNames(), i)
// {
//     Info << "object" <<objects.sortedNames()[i] << endl;
// }
// Info << runTime.timeName() << endl;

// if (renumberFields) Info<< "Reading geometric renumberFields" << nl << endl;

// #include "readVolFields.H"
// #include "readSurfaceFields.H"
// #include "readPointFields.H"

// Info<< endl;

// From renumbering:
// - from new cell/face back to original cell/face
labelList cellOrder;
labelList faceOrder;

if (blockSize > 0 || nBlocks > 1)
{
    // Renumbering in two phases. Should be done in one so mapping of
    // renumberFields is done correctly!

    // 如果设置了nBlocks，用设置的，如果没有就用blockSize算。
    if(nBlocks == 1){
        nBlocks = mesh.nCells() / blockSize;
    }
    // label nBlocks = mesh.nCells() / blockSize;
    Info<< "nBlocks = " << nBlocks << endl;

    // Read decompositionMethod dictionary
    dictionary decomposeDict(renumberDictPtr().subDict("blockCoeffs"));
    decomposeDict.set("numberOfSubdomains", nBlocks);

    bool oldParRun = UPstream::parRun();
    UPstream::parRun() = false;

    autoPtr<decompositionMethod> decomposePtr = decompositionMethod::New
    (
        decomposeDict
    );

    labelList cellToRegion
    (
        decomposePtr().decompose
        (
            mesh,
            mesh.cellCentres()
        )
    );

    // Restore state
    UPstream::parRun() = oldParRun;

    // For debugging: write out region
    // createScalarField
    // (
    //     mesh,
    //     "cellDist",
    //     cellToRegion
    // )().write();

    // Info<< nl << "Written decomposition as volScalarField to "
    //     << "cellDist for use in postprocessing."
    //     << nl << endl;
    

    // labelList newCellToRegion;

    cellOrder = regionRenumber(renumberPtr(), mesh, cellToRegion, regionPtr);

    // Determine new to old face order with new cell numbering
    faceOrder = getRegionFaceOrder
    (
        mesh,
        cellOrder,
        cellToRegion
    );

    // label nRegions = newCellToRegion.last() + 1;

    // labelListList regionToNewCells(invertOneToMany(nRegions, newCellToRegion));

    // assert(regionToNewCells.size() == nRegions);
    
    // Info << "RegionID : regionStart, regionSize" << endl;
    // label regionStart = 0;
    // for(label regionI = 0; regionI < nRegions; ++regionI){
    //     Info << regionI << " : " << regionStart << ", " << regionToNewCells[regionI].size() << endl;
    //     regionStart += regionToNewCells[regionI].size();
    // }

    Info << "RegionID : regionStart, regionSize" << endl;
    for(label regionI = 0; regionI < regionPtr.size() - 1; ++regionI){
        Info << regionI << " : " << regionPtr[regionI] << ", " << regionPtr[regionI+1] - regionPtr[regionI] << endl;
    }


}
else
{
    // Determines sorted back to original cell ordering
    cellOrder = renumberPtr().renumber
    (
        mesh,
        mesh.cellCentres()
    );

    if (sortCoupledFaceCells)
    {
        // Change order so all coupled patch faceCells are at the end.
        const polyBoundaryMesh& pbm = mesh.boundaryMesh();

        // Collect all boundary cells on coupled patches
        label nBndCells = 0;
        forAll(pbm, patchi)
        {
            if (pbm[patchi].coupled())
            {
                nBndCells += pbm[patchi].size();
            }
        }

        labelList reverseCellOrder = invert(mesh.nCells(), cellOrder);

        labelList bndCells(nBndCells);
        labelList bndCellMap(nBndCells);
        nBndCells = 0;
        forAll(pbm, patchi)
        {
            if (pbm[patchi].coupled())
            {
                const labelUList& faceCells = pbm[patchi].faceCells();
                forAll(faceCells, i)
                {
                    label celli = faceCells[i];

                    if (reverseCellOrder[celli] != -1)
                    {
                        bndCells[nBndCells] = celli;
                        bndCellMap[nBndCells++] = reverseCellOrder[celli];
                        reverseCellOrder[celli] = -1;
                    }
                }
            }
        }
        bndCells.setSize(nBndCells);
        bndCellMap.setSize(nBndCells);

        // Sort
        labelList order;
        sortedOrder(bndCellMap, order);

        // Redo newReverseCellOrder
        labelList newReverseCellOrder(mesh.nCells(), -1);

        label sortedI = mesh.nCells();
        forAllReverse(order, i)
        {
            label origCelli = bndCells[order[i]];
            newReverseCellOrder[origCelli] = --sortedI;
        }

        Info<< "Ordered all " << nBndCells << " cells with a coupled face"
            << " to the end of the cell list, starting at " << sortedI
            << endl;

        // Compact
        sortedI = 0;
        forAll(cellOrder, newCelli)
        {
            label origCelli = cellOrder[newCelli];
            if (newReverseCellOrder[origCelli] == -1)
            {
                newReverseCellOrder[origCelli] = sortedI++;
            }
        }

        // Update sorted back to original (unsorted) map
        cellOrder = invert(mesh.nCells(), newReverseCellOrder);
    }


    // Determine new to old face order with new cell numbering
    faceOrder = getFaceOrder
    (
        mesh,
        cellOrder      // New to old cell
    );
}


// if (!overwrite)
// {
//     runTime++;
// }
// runTime++;


// Change the mesh.
mesh.clearGeomNotOldVol();
autoPtr<mapPolyMesh> map = reorderMesh(mesh, cellOrder, faceOrder);
// Info << "cellOrder" << cellOrder << endl;
// Info << "faceOrder" << faceOrder << endl;

if (orderPoints)
{
    polyTopoChange meshMod(mesh);
    autoPtr<mapPolyMesh> pointOrderMap = meshMod.changeMesh
    (
        mesh,
        false,      // inflate
        true,       // syncParallel
        false,      // orderCells
        orderPoints // orderPoints
    );

    // Combine point reordering into map.
    const_cast<labelList&>(map().pointMap()) = UIndirectList<label>
    (
        map().pointMap(),
        pointOrderMap().pointMap()
    )();

    inplaceRenumber
    (
        pointOrderMap().reversePointMap(),
        const_cast<labelList&>(map().reversePointMap())
    );
}

// Info << "map oldCell = " << map->cellMap() << endl;
// Info << "hasOldCellVolumes = " << map->hasOldCellVolumes() << endl;
// Update renumberFields
mesh.updateMesh(map);

// Update proc maps
if
(
    cellProcAddressing.headerOk()
    && cellProcAddressing.size() == mesh.nCells()
)
{
    Info<< "Renumbering processor cell decomposition map "
        << cellProcAddressing.name() << endl;

    cellProcAddressing = labelList
    (
        UIndirectList<label>(cellProcAddressing, map().cellMap())
    );
}
if
(
    faceProcAddressing.headerOk()
    && faceProcAddressing.size() == mesh.nFaces()
)
{
    Info<< "Renumbering processor face decomposition map "
        << faceProcAddressing.name() << endl;

    faceProcAddressing = labelList
    (
        UIndirectList<label>(faceProcAddressing, map().faceMap())
    );

    // Detect any flips.
    const labelHashSet& fff = map().flipFaceFlux();
    forAllConstIter(labelHashSet, fff, iter)
    {
        label facei = iter.key();
        label masterFacei = faceProcAddressing[facei];

        faceProcAddressing[facei] = -masterFacei;

        if (masterFacei == 0)
        {
            FatalErrorInFunction
                << " masterFacei:" << masterFacei << exit(FatalError);
        }
    }
}
if
(
    pointProcAddressing.headerOk()
    && pointProcAddressing.size() == mesh.nPoints()
)
{
    Info<< "Renumbering processor point decomposition map "
        << pointProcAddressing.name() << endl;

    pointProcAddressing = labelList
    (
        UIndirectList<label>(pointProcAddressing, map().pointMap())
    );
}


// Move mesh (since morphing might not do this)
if (map().hasMotionPoints())
{
    mesh.movePoints(map().preMotionPoints());
}


{
    label band;
    scalar profile;
    scalar sumSqrIntersect;
    getBand
    (
        doFrontWidth,
        mesh.nCells(),
        mesh.faceOwner(),
        mesh.faceNeighbour(),
        band,
        profile,
        sumSqrIntersect
    );
    reduce(band, maxOp<label>());
    reduce(profile, sumOp<scalar>());
    scalar rmsFrontwidth = Foam::sqrt
    (
        returnReduce
        (
            sumSqrIntersect,
            sumOp<scalar>()
        )/mesh.globalData().nTotalCells()
    );

    Info<< "After renumbering :" << nl
        << "    band           : " << band << nl
        << "    profile        : " << profile << nl;

    if (doFrontWidth)
    {

        Info<< "    rms frontwidth : " << rmsFrontwidth << nl;
    }

    Info<< endl;
}

if (orderPoints)
{
    // Force edge calculation (since only reason that points would need to
    // be sorted)
    (void)mesh.edges();

    label nTotPoints = returnReduce
    (
        mesh.nPoints(),
        sumOp<label>()
    );
    label nTotIntPoints = returnReduce
    (
        mesh.nInternalPoints(),
        sumOp<label>()
    );

    label nTotEdges = returnReduce
    (
        mesh.nEdges(),
        sumOp<label>()
    );
    label nTotIntEdges = returnReduce
    (
        mesh.nInternalEdges(),
        sumOp<label>()
    );
    label nTotInt0Edges = returnReduce
    (
        mesh.nInternal0Edges(),
        sumOp<label>()
    );
    label nTotInt1Edges = returnReduce
    (
        mesh.nInternal1Edges(),
        sumOp<label>()
    );

    Info<< "Points:" << nl
        << "    total   : " << nTotPoints << nl
        << "    internal: " << nTotIntPoints << nl
        << "    boundary: " << nTotPoints-nTotIntPoints << nl
        << "Edges:" << nl
        << "    total   : " << nTotEdges << nl
        << "    internal: " << nTotIntEdges << nl
        << "        internal using 0 boundary points: "
        << nTotInt0Edges << nl
        << "        internal using 1 boundary points: "
        << nTotInt1Edges-nTotInt0Edges << nl
        << "        internal using 2 boundary points: "
        << nTotIntEdges-nTotInt1Edges << nl
        << "    boundary: " << nTotEdges-nTotIntEdges << nl
        << endl;
}
    // if (overwrite)
    // {
    //     mesh.setInstance(oldInstance);
    // }

    // Info<< "Writing mesh to " << mesh.facesInstance() << endl;

    // mesh.write();

    // if (cellProcAddressing.headerOk())
    // {
    //     cellProcAddressing.instance() = mesh.facesInstance();
    //     if (cellProcAddressing.size() == mesh.nCells())
    //     {
    //         cellProcAddressing.write();
    //     }
    //     else
    //     {
    //         // procAddressing file no longer valid. Delete it.
    //         const fileName fName(cellProcAddressing.filePath());
    //         if (fName.size())
    //         {
    //             Info<< "Deleting inconsistent processor cell decomposition"
    //                 << " map " << fName << endl;
    //             rm(fName);
    //         }
    //     }
    // }

    // if (faceProcAddressing.headerOk())
    // {
    //     faceProcAddressing.instance() = mesh.facesInstance();
    //     if (faceProcAddressing.size() == mesh.nFaces())
    //     {
    //         faceProcAddressing.write();
    //     }
    //     else
    //     {
    //         const fileName fName(faceProcAddressing.filePath());
    //         if (fName.size())
    //         {
    //             Info<< "Deleting inconsistent processor face decomposition"
    //                 << " map " << fName << endl;
    //             rm(fName);
    //         }
    //     }
    // }

    // if (pointProcAddressing.headerOk())
    // {
    //     pointProcAddressing.instance() = mesh.facesInstance();
    //     if (pointProcAddressing.size() == mesh.nPoints())
    //     {
    //         pointProcAddressing.write();
    //     }
    //     else
    //     {
    //         const fileName fName(pointProcAddressing.filePath());
    //         if (fName.size())
    //         {
    //             Info<< "Deleting inconsistent processor point decomposition"
    //                 << " map " << fName << endl;
    //             rm(fName);
    //         }
    //     }
    // }

    // if (boundaryProcAddressing.headerOk())
    // {
    //     boundaryProcAddressing.instance() = mesh.facesInstance();
    //     if (boundaryProcAddressing.size() == mesh.boundaryMesh().size())
    //     {
    //         boundaryProcAddressing.write();
    //     }
    //     else
    //     {
    //         const fileName fName(boundaryProcAddressing.filePath());
    //         if (fName.size())
    //         {
    //             Info<< "Deleting inconsistent processor patch decomposition"
    //                 << " map " << fName << endl;
    //             rm(fName);
    //         }
    //     }
    // }

    // if (writeMaps)
    // {
    //     // For debugging: write out region
    //     createScalarField
    //     (
    //         mesh,
    //         "origCellID",
    //         map().cellMap()
    //     )().write();

    //     createScalarField
    //     (
    //         mesh,
    //         "cellID",
    //         identity(mesh.nCells())
    //     )().write();

    //     Info<< nl << "Written current cellID and origCellID as volScalarField"
    //         << " for use in postprocessing."
    //         << nl << endl;

    //     labelIOList
    //     (
    //         IOobject
    //         (
    //             "cellMap",
    //             mesh.facesInstance(),
    //             polyMesh::meshSubDir,
    //             mesh,
    //             IOobject::NO_READ,
    //             IOobject::NO_WRITE,
    //             false
    //         ),
    //         map().cellMap()
    //     ).write();

    //     labelIOList
    //     (
    //         IOobject
    //         (
    //             "faceMap",
    //             mesh.facesInstance(),
    //             polyMesh::meshSubDir,
    //             mesh,
    //             IOobject::NO_READ,
    //             IOobject::NO_WRITE,
    //             false
    //         ),
    //         map().faceMap()
    //     ).write();

    //     labelIOList
    //     (
    //         IOobject
    //         (
    //             "pointMap",
    //             mesh.facesInstance(),
    //             polyMesh::meshSubDir,
    //             mesh,
    //             IOobject::NO_READ,
    //             IOobject::NO_WRITE,
    //             false
    //         ),
    //         map().pointMap()
    //     ).write();
    // }


    // Renumber sets if required
    if (renumberSets)
    {
        Info<< endl;

        // Read sets
        IOobjectList objects(mesh, mesh.facesInstance(), "polyMesh/sets");

        {
            IOobjectList cSets(objects.lookupClass(cellSet::typeName));
            if (cSets.size())
            {
                Info<< "Renumbering cellSets:" << endl;
                forAllConstIter(IOobjectList, cSets, iter)
                {
                    cellSet cs(*iter());
                    Info<< "    " << cs.name() << endl;
                    cs.updateMesh(map());
                    cs.instance() = mesh.facesInstance();
                    // cs.write();
                }
            }
        }

        {
            IOobjectList fSets(objects.lookupClass(faceSet::typeName));
            if (fSets.size())
            {
                Info<< "Renumbering faceSets:" << endl;
                forAllConstIter(IOobjectList, fSets, iter)
                {
                    faceSet fs(*iter());
                    Info<< "    " << fs.name() << endl;
                    fs.updateMesh(map());
                    fs.instance() = mesh.facesInstance();
                    // fs.write();
                }
            }
        }

        {
            IOobjectList pSets(objects.lookupClass(pointSet::typeName));
            if (pSets.size())
            {
                Info<< "Renumbering pointSets:" << endl;
                forAllConstIter(IOobjectList, pSets, iter)
                {
                    pointSet ps(*iter());
                    Info<< "    " << ps.name() << endl;
                    ps.updateMesh(map());
                    ps.instance() = mesh.facesInstance();
                    // ps.write();
                }
            }
        }
    }

    // Info<< "End\n" << endl;

    // return 0;
// }


// ************************************************************************* //

// }
