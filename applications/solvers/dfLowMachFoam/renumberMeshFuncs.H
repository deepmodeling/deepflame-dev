// Create named field from labelList for postprocessing
tmp<volScalarField> createScalarField
(
    const fvMesh& mesh,
    const word& name,
    const labelList& elems
)
{
    tmp<volScalarField> tfld
    (
        volScalarField::New
        (
            name,
            mesh,
            dimensionedScalar(dimless, 0),
            zeroGradientFvPatchScalarField::typeName
        )
    );
    volScalarField& fld = tfld.ref();

    forAll(fld, celli)
    {
       fld[celli] = elems[celli];
    }

    return tfld;
}


// Calculate band of matrix
label getBand(const labelList& owner, const labelList& neighbour)
{
    label band = 0;

    forAll(neighbour, facei)
    {
        label diff = neighbour[facei] - owner[facei];

        if (diff > band)
        {
            band = diff;
        }
    }
    return band;
}


// Calculate band of matrix
void getBand
(
    const bool calculateIntersect,
    const label nCells,
    const labelList& owner,
    const labelList& neighbour,
    label& bandwidth,
    scalar& profile,            // scalar to avoid overflow
    scalar& sumSqrIntersect     // scalar to avoid overflow
)
{
    labelList cellBandwidth(nCells, 0);
    scalarField nIntersect(nCells, 0.0);

    forAll(neighbour, facei)
    {
        label own = owner[facei];
        label nei = neighbour[facei];

        // Note: mag not necessary for correct (upper-triangular) ordering.
        label diff = nei-own;
        cellBandwidth[nei] = max(cellBandwidth[nei], diff);
    }

    bandwidth = max(cellBandwidth);

    // Do not use field algebra because of conversion label to scalar
    profile = 0.0;
    forAll(cellBandwidth, celli)
    {
        profile += 1.0*cellBandwidth[celli];
    }

    sumSqrIntersect = 0.0;
    if (calculateIntersect)
    {
        forAll(nIntersect, celli)
        {
            for (label colI = celli-cellBandwidth[celli]; colI <= celli; colI++)
            {
                nIntersect[colI] += 1.0;
            }
        }

        sumSqrIntersect = sum(Foam::sqr(nIntersect));
    }
}


// Determine upper-triangular face order
labelList getFaceOrder
(
    const primitiveMesh& mesh,
    const labelList& cellOrder      // New to old cell
)
{
    labelList reverseCellOrder(invert(cellOrder.size(), cellOrder));

    labelList oldToNewFace(mesh.nFaces(), -1);

    label newFacei = 0;

    labelList nbr;
    labelList order;

    forAll(cellOrder, newCelli)
    {
        label oldCelli = cellOrder[newCelli];

        const cell& cFaces = mesh.cells()[oldCelli];

        // Neighbouring cells
        nbr.setSize(cFaces.size());

        forAll(cFaces, i)
        {
            label facei = cFaces[i];

            if (mesh.isInternalFace(facei))
            {
                // Internal face. Get cell on other side.
                label nbrCelli = reverseCellOrder[mesh.faceNeighbour()[facei]];
                if (nbrCelli == newCelli)
                {
                    nbrCelli = reverseCellOrder[mesh.faceOwner()[facei]];
                }

                if (newCelli < nbrCelli)
                {
                    // Celli is master
                    nbr[i] = nbrCelli;
                }
                else
                {
                    // nbrCell is master. Let it handle this face.
                    nbr[i] = -1;
                }
            }
            else
            {
                // External face. Do later.
                nbr[i] = -1;
            }
        }

        order.setSize(nbr.size());
        sortedOrder(nbr, order);

        forAll(order, i)
        {
            label index = order[i];
            if (nbr[index] != -1)
            {
                oldToNewFace[cFaces[index]] = newFacei++;
            }
        }
    }

    // Leave patch faces intact.
    for (label facei = newFacei; facei < mesh.nFaces(); facei++)
    {
        oldToNewFace[facei] = facei;
    }


    // Check done all faces.
    forAll(oldToNewFace, facei)
    {
        if (oldToNewFace[facei] == -1)
        {
            FatalErrorInFunction
                << "Did not determine new position" << " for face " << facei
                << abort(FatalError);
        }
    }

    return invert(mesh.nFaces(), oldToNewFace);
}


// Determine face order such that inside region faces are sorted
// upper-triangular but in between region faces are handled like boundary faces.
labelList getRegionFaceOrder
(
    const primitiveMesh& mesh,
    const labelList& cellOrder,     // New to old cell
    const labelList& cellToRegion   // Old cell to region
)
{
    labelList reverseCellOrder(invert(cellOrder.size(), cellOrder));

    labelList oldToNewFace(mesh.nFaces(), -1);

    label newFacei = 0;

    label prevRegion = -1;

    forAll(cellOrder, newCelli)
    {
        label oldCelli = cellOrder[newCelli];

        if (cellToRegion[oldCelli] != prevRegion)
        {
            prevRegion = cellToRegion[oldCelli];
        }

        const cell& cFaces = mesh.cells()[oldCelli];

        SortableList<label> nbr(cFaces.size());

        forAll(cFaces, i)
        {
            label facei = cFaces[i];

            if (mesh.isInternalFace(facei))
            {
                // Internal face. Get cell on other side.
                label nbrCelli = reverseCellOrder[mesh.faceNeighbour()[facei]];
                if (nbrCelli == newCelli)
                {
                    nbrCelli = reverseCellOrder[mesh.faceOwner()[facei]];
                }

                if (cellToRegion[oldCelli] != cellToRegion[cellOrder[nbrCelli]])
                {
                    // Treat like external face. Do later.
                    nbr[i] = -1;
                }
                else if (newCelli < nbrCelli)
                {
                    // Celli is master
                    nbr[i] = nbrCelli;
                }
                else
                {
                    // nbrCell is master. Let it handle this face.
                    nbr[i] = -1;
                }
            }
            else
            {
                // External face. Do later.
                nbr[i] = -1;
            }
        }

        nbr.sort();

        forAll(nbr, i)
        {
            if (nbr[i] != -1)
            {
                oldToNewFace[cFaces[nbr.indices()[i]]] = newFacei++;
            }
        }
    }

    // Do region interfaces
    label nRegions = max(cellToRegion)+1;
    {
        // Sort in increasing region
        SortableList<label> sortKey(mesh.nFaces(), labelMax);

        for (label facei = 0; facei < mesh.nInternalFaces(); facei++)
        {
            label ownRegion = cellToRegion[mesh.faceOwner()[facei]];
            label neiRegion = cellToRegion[mesh.faceNeighbour()[facei]];

            if (ownRegion != neiRegion)
            {
                sortKey[facei] =
                    min(ownRegion, neiRegion)*nRegions
                   +max(ownRegion, neiRegion);
            }
        }
        sortKey.sort();

        // Extract.
        label prevKey = -1;
        forAll(sortKey, i)
        {
            label key = sortKey[i];

            if (key == labelMax)
            {
                break;
            }

            if (prevKey != key)
            {
                prevKey = key;
            }

            oldToNewFace[sortKey.indices()[i]] = newFacei++;
        }
    }

    // Leave patch faces intact.
    for (label facei = newFacei; facei < mesh.nFaces(); facei++)
    {
        oldToNewFace[facei] = facei;
    }


    // Check done all faces.
    forAll(oldToNewFace, facei)
    {
        if (oldToNewFace[facei] == -1)
        {
            FatalErrorInFunction
                << "Did not determine new position"
                << " for face " << facei
                << abort(FatalError);
        }
    }

    return invert(mesh.nFaces(), oldToNewFace);
}


// cellOrder: old cell for every new cell
// faceOrder: old face for every new face. Ordering of boundary faces not
//     changed.
autoPtr<mapPolyMesh> reorderMesh
(
    polyMesh& mesh,
    const labelList& cellOrder,
    const labelList& faceOrder
)
{
    labelList reverseCellOrder(invert(cellOrder.size(), cellOrder));
    labelList reverseFaceOrder(invert(faceOrder.size(), faceOrder));

    faceList newFaces(reorder(reverseFaceOrder, mesh.faces()));
    labelList newOwner
    (
        renumber
        (
            reverseCellOrder,
            reorder(reverseFaceOrder, mesh.faceOwner())
        )
    );
    labelList newNeighbour
    (
        renumber
        (
            reverseCellOrder,
            reorder(reverseFaceOrder, mesh.faceNeighbour())
        )
    );

    // Check if any faces need swapping.
    labelHashSet flipFaceFlux(newOwner.size());
    forAll(newNeighbour, facei)
    {
        label own = newOwner[facei];
        label nei = newNeighbour[facei];

        if (nei < own)
        {
            newFaces[facei].flip();
            Swap(newOwner[facei], newNeighbour[facei]);
            flipFaceFlux.insert(facei);
        }
    }

    const polyBoundaryMesh& patches = mesh.boundaryMesh();
    labelList patchSizes(patches.size());
    labelList patchStarts(patches.size());
    labelList oldPatchNMeshPoints(patches.size());
    labelListList patchPointMap(patches.size());

    forAll(patches, patchi)
    {
        patchSizes[patchi] = patches[patchi].size();
        patchStarts[patchi] = patches[patchi].start();
        oldPatchNMeshPoints[patchi] = patches[patchi].nPoints();
        patchPointMap[patchi] = identity(patches[patchi].nPoints());
    }

    mesh.resetPrimitives
    (
        NullObjectMove<pointField>(),
        move(newFaces),
        move(newOwner),
        move(newNeighbour),
        patchSizes,
        patchStarts,
        true
    );


    // Re-do the faceZones
    {
        faceZoneMesh& faceZones = mesh.faceZones();
        faceZones.clearAddressing();
        forAll(faceZones, zoneI)
        {
            faceZone& fZone = faceZones[zoneI];
            labelList newAddressing(fZone.size());
            boolList newFlipMap(fZone.size());
            forAll(fZone, i)
            {
                label oldFacei = fZone[i];
                newAddressing[i] = reverseFaceOrder[oldFacei];
                if (flipFaceFlux.found(newAddressing[i]))
                {
                    newFlipMap[i] = !fZone.flipMap()[i];
                }
                else
                {
                    newFlipMap[i] = fZone.flipMap()[i];
                }
            }
            labelList newToOld;
            sortedOrder(newAddressing, newToOld);
            fZone.resetAddressing
            (
                UIndirectList<label>(newAddressing, newToOld)(),
                UIndirectList<bool>(newFlipMap, newToOld)()
            );
        }
    }
    // Re-do the cellZones
    {
        cellZoneMesh& cellZones = mesh.cellZones();
        cellZones.clearAddressing();
        forAll(cellZones, zoneI)
        {
            cellZones[zoneI] = UIndirectList<label>
            (
                reverseCellOrder,
                cellZones[zoneI]
            )();
            Foam::sort(cellZones[zoneI]);
        }
    }


    return autoPtr<mapPolyMesh>
    (
        new mapPolyMesh
        (
            mesh,                       // const polyMesh& mesh,
            mesh.nPoints(),             // nOldPoints,
            mesh.nFaces(),              // nOldFaces,
            mesh.nCells(),              // nOldCells,
            identity(mesh.nPoints()),   // pointMap,
            List<objectMap>(0),         // pointsFromPoints,
            faceOrder,                  // faceMap,
            List<objectMap>(0),         // facesFromPoints,
            List<objectMap>(0),         // facesFromEdges,
            List<objectMap>(0),         // facesFromFaces,
            cellOrder,                  // cellMap,
            List<objectMap>(0),         // cellsFromPoints,
            List<objectMap>(0),         // cellsFromEdges,
            List<objectMap>(0),         // cellsFromFaces,
            List<objectMap>(0),         // cellsFromCells,
            identity(mesh.nPoints()),   // reversePointMap,
            reverseFaceOrder,           // reverseFaceMap,
            reverseCellOrder,           // reverseCellMap,
            flipFaceFlux,               // flipFaceFlux,
            patchPointMap,              // patchPointMap,
            labelListList(0),           // pointZoneMap,
            labelListList(0),           // faceZonePointMap,
            labelListList(0),           // faceZoneFaceMap,
            labelListList(0),           // cellZoneMap,
            pointField(0),              // preMotionPoints,
            patchStarts,                // oldPatchStarts,
            oldPatchNMeshPoints,        // oldPatchNMeshPoints
            autoPtr<scalarField>()      // oldCellVolumes
        )
    );
}


// Return new to old cell numbering
labelList regionRenumber
(
    const renumberMethod& method,
    const fvMesh& mesh,
    const labelList& cellToRegion,
    labelList& regionPtr
)
{
    Info<< "Determining cell order:" << endl;

    labelList cellOrder(cellToRegion.size());

    label nRegions = max(cellToRegion)+1;

    labelListList regionToCells(invertOneToMany(nRegions, cellToRegion));

    // newCellToRegion.resize(cellToRegion.size());

    label celli = 0;

    forAll(regionToCells, regionI)
    {
        Info<< "    region " << regionI << " starts at " << celli << endl;
        regionPtr.append(celli);

        // Make sure no parallel comms
        bool oldParRun = UPstream::parRun();
        UPstream::parRun() = false;

        // Per region do a reordering.
        fvMeshSubset subsetter(mesh);
        subsetter.setLargeCellSubset(cellToRegion, regionI);

        const fvMesh& subMesh = subsetter.subMesh();

        labelList subCellOrder = method.renumber
        (
            subMesh,
            subMesh.cellCentres()
        );

        // Restore state
        UPstream::parRun() = oldParRun;

        const labelList& cellMap = subsetter.cellMap();

        forAll(subCellOrder, i)
        {
            cellOrder[celli] = cellMap[subCellOrder[i]];
            // newCellToRegion[celli] = regionI;
            celli += 1;
        }
    }
    Info<< endl;
    
    regionPtr.append(cellToRegion.size());

    return cellOrder;
}