/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2015 OpenFOAM Foundation
    Copyright (C) 2017-2022 OpenCFD Ltd.
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

#include "faceShading.H"
#include "fvMesh.H"
#include "boundaryRadiationProperties.H"
#include "cyclicAMIPolyPatch.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "distributedTriSurfaceMesh.H"
#include "OBJstream.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(faceShading, 0);
}

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

void Foam::faceShading::calculate()
{
    const auto& pbm = mesh_.boundaryMesh();

    const PackedBoolList isOpaqueFace
    (
        selectOpaqueFaces
        (
            radiation::boundaryRadiationProperties::New(mesh_),
            patchIDs_,
            zoneIDs_
        )
    );

    // Find faces potentially hit by solar rays
    //  - correct normal
    //  - transmissivity 0
    labelList hitFacesIds;
    PackedBoolList hitFacesFlips;
    selectFaces
    (
        true,   // use normal to do first filtering
        isOpaqueFace,
        patchIDs_,
        zoneIDs_,

        hitFacesIds,
        hitFacesFlips
    );

    Info<< "Number of 'potential' direct hits : "
        << returnReduce(hitFacesIds.size(), sumOp<label>()) << endl;


    // * * * * * * * * * * * * * * *
    // Create distributedTriSurfaceMesh
    Random rndGen(653213);

    // Find potential obstructions. Include all faces that might potentially
    // block (so ignore normal)
    labelList blockingFacesIds;
    PackedBoolList blockingFacesFlips;
    selectFaces
    (
        false,   // use normal to do first filtering
        isOpaqueFace,
        patchIDs_,
        zoneIDs_,

        blockingFacesIds,
        blockingFacesFlips
    );

    const triSurface localSurface = triangulate
    (
        blockingFacesIds,
        blockingFacesFlips
    );

    // Determine mesh bounding boxes:
    List<treeBoundBox> meshBb
    (
        1,
        treeBoundBox(mesh_.points()).extend(1e-3)
    );

    // Dummy bounds dictionary
    dictionary dict;
    dict.add("bounds", meshBb);
    dict.add
    (
        "distributionType",
        distributedTriSurfaceMesh::distributionTypeNames_
        [
            distributedTriSurfaceMesh::FROZEN
        ]
    );
    dict.add("mergeDistance", SMALL);

    distributedTriSurfaceMesh surfacesMesh
    (
        IOobject
        (
            "opaqueSurface.stl",
            mesh_.time().constant(),    // directory
            "triSurface",               // instance
            mesh_.time(),               // registry
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        localSurface,
        dict
    );

    if (debug)
    {
        surfacesMesh.searchableSurface::write();
    }

    const scalar maxBounding =
        returnReduce(5.0*mesh_.bounds().mag(), maxOp<scalar>());

    // Calculate index of faces which have a direct hit (local)

    // Shoot Rays
    // * * * * * * * * * * * * * * * *
    {

        DynamicField<point> start(hitFacesIds.size());
        DynamicField<point> end(start.size());
        DynamicList<label> startIndex(start.size());

        const pointField& faceCentres = mesh_.faceCentres();

        const vector d(direction_*maxBounding);

        forAll(hitFacesIds, i)
        {
            const label facei = hitFacesIds[i];
            const point& fc = faceCentres[facei];

            start.append(fc - 0.001*d);
            startIndex.append(facei);
            end.append(fc - d);
        }

        List<pointIndexHit> hitInfo(startIndex.size());
        surfacesMesh.findLine(start, end, hitInfo);

        // Collect the rays which has 'only one not wall' obstacle between
        // start and end.
        // If the ray hit itself get stored in dRayIs

        label nVisible = 0;
        forAll(hitInfo, rayI)
        {
            if (!hitInfo[rayI].hit())
            {
                nVisible++;
            }
        }
        rayStartFaces_.setSize(nVisible);
        nVisible = 0;

        forAll(hitInfo, rayI)
        {
            if (!hitInfo[rayI].hit())
            {
                rayStartFaces_[nVisible++] = startIndex[rayI];
            }
        }

        // Plot all rays between visible faces.
        if (debug)
        {
            writeRays
            (
                mesh_.time().path()/"allVisibleFaces.obj",
                end,
                start
            );
        }

        start.clear();
        startIndex.clear();
        end.clear();
    }

    if (debug)
    {
        auto thitFaces = surfaceScalarField::New
        (
            "hitFaces",
            // IOobject::NO_REGISTER,
            mesh_,
            dimensionedScalar(dimless, Zero)
        );
        auto& hitFaces = thitFaces.ref();

        surfaceScalarField::Boundary& hitFacesBf = hitFaces.boundaryFieldRef();

        hitFacesBf = 0.0;
        for (const label facei : rayStartFaces_)
        {
            const label patchID = pbm.whichPatch(facei);
            if (patchID == -1)
            {
                hitFaces[facei] = 1.0;
            }
            else
            {
                const polyPatch& pp = pbm[patchID];
                hitFacesBf[patchID][facei - pp.start()] = 1.0;
            }
        }
        hitFaces.write();
    }

    Info<< "Total number of hit faces : "
        << returnReduce(rayStartFaces_.size(), sumOp<label>()) << endl;
}


Foam::triSurface Foam::faceShading::triangulate
(
    const labelUList& faceIDs,
    const PackedBoolList& flipMap
) const
{
    if (faceIDs.size() != flipMap.size())
    {
        FatalErrorInFunction << "Size problem :"
            << "faceIDs:" << faceIDs.size()
            << "flipMap:" << flipMap.size()
            << exit(FatalError);
    }

    const auto& points = mesh_.points();
    const auto& faces = mesh_.faces();
    const auto& bMesh = mesh_.boundaryMesh();
    const auto& fzs = mesh_.faceZones();

    // Patching of surface:
    // - non-processor patches
    // - faceZones
    // Note: check for faceZones on boundary? Who has priority?
    label nNonProcessor = 0;
    for (const polyPatch& p : bMesh)
    {
        if (isA<processorPolyPatch>(p))
        {
            break;
        }

        ++nNonProcessor;
    }
    geometricSurfacePatchList surfPatches(nNonProcessor+fzs.size());
    labelList patchID(mesh_.nFaces(), -1);
    {
        label newPatchi = 0;

        for (label patchi = 0; patchi < nNonProcessor; ++patchi)
        {
            const auto& pp = bMesh[patchi];

            surfPatches[newPatchi] = geometricSurfacePatch
            (
                pp.type(),
                pp.name(),
                newPatchi
            );
            SubList<label>
            (
                patchID,
                pp.size(),
                pp.start()
            ) = newPatchi;

            newPatchi++;
        }
        for (const auto& fz : fzs)
        {
            surfPatches[newPatchi] = geometricSurfacePatch
            (
                fz.type(),
                fz.name(),
                newPatchi
            );
            UIndirectList<label>(patchID, fz) = newPatchi;

            newPatchi++;
        }
    }


    // Storage for surfaceMesh. Size estimate.
    DynamicList<labelledTri> triangles(2*faceIDs.size());

    // Work array
    faceList triFaces;

    forAll(faceIDs, i)
    {
        const label facei = faceIDs[i];
        const bool flip = flipMap[i];
        const label patchi = patchID[facei];
        const face& f = faces[facei];

        // Triangulate face
        triFaces.setSize(f.nTriangles(points));
        label nTri = 0;
        f.triangles(points, nTri, triFaces);

        for (const face& f : triFaces)
        {
            if (!flip)
            {
                triangles.append(labelledTri(f[0], f[1], f[2], patchi));
            }
            else
            {
                triangles.append(labelledTri(f[0], f[2], f[1], patchi));
            }
        }
    }

    triangles.shrink();

    // Create globally numbered tri surface
    triSurface rawSurface(triangles, mesh_.points());

    // Create locally numbered tri surface
    triSurface surface
    (
        rawSurface.localFaces(),
        rawSurface.localPoints()
    );

    // Add patch names to surface
    surface.patches().transfer(surfPatches);

    return surface;
}


Foam::PackedBoolList Foam::faceShading::selectOpaqueFaces
(
    const radiation::boundaryRadiationProperties& boundaryRadiation,
    const labelUList& patchIDs,
    const labelUList& zoneIDs
) const
{
    const auto& pbm = mesh_.boundaryMesh();

    PackedBoolList isOpaqueFace(mesh_.nFaces(), false);

    // Check selected patches
    for (const label patchi : patchIDs)
    {
        const auto& pp = pbm[patchi];
        tmp<scalarField> tt = boundaryRadiation.transmissivity(patchi);
        const scalarField& t = tt();

        forAll(t, i)
        {
            isOpaqueFace[i + pp.start()] = (t[i] == 0.0);
        }
    }

    // Check selected faceZones
    const auto& fzs = mesh_.faceZones();

    for (const label zonei : zoneIDs)
    {
        const auto& fz = fzs[zonei];

        //- Note: slice mesh face centres preferentially
        tmp<scalarField> tt = boundaryRadiation.zoneTransmissivity
        (
            zonei,
            fz
        );
        const scalarField& t = tt();

        forAll(t, i)
        {
            isOpaqueFace[fz[i]] = (t[i] == 0.0);
        }
    }

    return isOpaqueFace;
}


void Foam::faceShading::selectFaces
(
    const bool useNormal,
    const PackedBoolList& isCandidateFace,
    const labelUList& patchIDs,
    const labelUList& zoneIDs,

    labelList& faceIDs,
    PackedBoolList& flipMap
) const
{
    const auto& pbm = mesh_.boundaryMesh();

    PackedBoolList isSelected(mesh_.nFaces());
    DynamicList<label> dynFaces(mesh_.nFaces() - mesh_.nInternalFaces());
    PackedBoolList isFaceFlipped(mesh_.nFaces());

    // Add patches
    for (const label patchi : patchIDs)
    {
        const auto& pp = pbm[patchi];
        const vectorField& n = pp.faceNormals();

        forAll(n, i)
        {
            const label meshFacei = i + pp.start();
            if
            (
                isCandidateFace[meshFacei]
             && (
                    !useNormal
                 || ((direction_ & n[i]) > 0)
                )
            )
            {
                isSelected.set(meshFacei);
                isFaceFlipped[meshFacei] = false;
                dynFaces.append(meshFacei);
            }
        }
    }


    // Add faceZones
    const auto& fzs = mesh_.faceZones();

    for (const label zonei : zoneIDs)
    {
        const auto& fz = fzs[zonei];
        const primitiveFacePatch& pp = fz();
        const vectorField& n = pp.faceNormals();

        forAll(n, i)
        {
            const label meshFacei = fz[i];

            if
            (
                !isSelected[meshFacei]
             && isCandidateFace[meshFacei]
             && (
                    !useNormal
                 || ((direction_ & n[i]) > 0)
                )
            )
            {
                isSelected.set(meshFacei);
                dynFaces.append(meshFacei);
                isFaceFlipped[meshFacei] = fz.flipMap()[i];
            }
        }
    }
    faceIDs = std::move(dynFaces);
    //flipMap = PackedBoolList(isFaceFlipped, faceIDs);
    const label len = faceIDs.size();
    flipMap = PackedBoolList(len);
    for (label i = 0; i < len; ++i)
    {
        flipMap.set(i, isFaceFlipped.get(faceIDs[i]));
    }
}


void Foam::faceShading::writeRays
(
    const fileName& fName,
    const DynamicField<point>& endCf,
    const pointField& myFc
)
{
    OBJstream os(fName);

    Pout<< "Dumping rays to " << os.name() << endl;

    forAll(myFc, facei)
    {
        os.write(linePointRef(myFc[facei], endCf[facei]));
    }
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::faceShading::faceShading
(
    const fvMesh& mesh,
    const vector& dir
)
:
    mesh_(mesh),
    patchIDs_(nonCoupledPatches(mesh)),
    zoneIDs_(0),
    direction_(dir),
    rayStartFaces_(0)
{
    calculate();
}


Foam::faceShading::faceShading
(
    const fvMesh& mesh,
    const labelList& patchIDs,
    const labelList& zoneIDs,
    const vector& dir
)
:
    mesh_(mesh),
    patchIDs_(patchIDs),
    zoneIDs_(zoneIDs),
    direction_(dir),
    rayStartFaces_(0)
{
    calculate();
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::labelList Foam::faceShading::nonCoupledPatches(const polyMesh& mesh)
{
    const auto& pbm = mesh.boundaryMesh();

    DynamicList<label> ncPatches;
    forAll(pbm, patchi)
    {
        const polyPatch& pp = pbm[patchi];
        if (!pp.coupled() && !isA<cyclicAMIPolyPatch>(pp))
        {
            ncPatches.append(patchi);
        }
    }
    return ncPatches;
}


void Foam::faceShading::correct()
{
    calculate();
}


// ************************************************************************* //
