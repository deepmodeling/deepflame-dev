/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2018-2022 OpenCFD Ltd.
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

#include "faceReflecting.H"
#include "boundaryRadiationProperties.H"
#include "cyclicAMIPolyPatch.H"
#include "volFields.H"


using namespace Foam::constant::mathematical;

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(faceReflecting, 0);
}

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

void Foam::faceReflecting::initialise(const dictionary& coeffs)
{

    forAll(qreflective_, bandI)
    {
        qreflective_.set
        (
            bandI,
            new volScalarField
            (
                IOobject
                (
                    "qreflective_" + Foam::name(bandI) ,
                    mesh_.time().timeName(),
                    mesh_,
                    IOobject::NO_READ,
                    IOobject::AUTO_WRITE
                ),
                mesh_,
                dimensionedScalar(dimMass/pow3(dimTime), Zero)
            )
        );
    }

    label rayI = 0;
    if (mesh_.nSolutionD() == 3)
    {
        nRay_ = 4*nPhi_*nTheta_;
        refDiscAngles_.resize(nRay_);
        const scalar deltaPhi = pi/(2.0*nPhi_);
        const scalar deltaTheta = pi/nTheta_;

        for (label n = 1; n <= nTheta_; n++)
        {
            for (label m = 1; m <= 4*nPhi_; m++)
            {
                const scalar thetai = (2*n - 1)*deltaTheta/2.0;
                const scalar phii = (2*m - 1)*deltaPhi/2.0;

                scalar sinTheta = Foam::sin(thetai);
                scalar cosTheta = Foam::cos(thetai);
                scalar sinPhi = Foam::sin(phii);
                scalar cosPhi = Foam::cos(phii);
                refDiscAngles_[rayI++] =
                    vector(sinTheta*sinPhi, sinTheta*cosPhi, cosTheta);

            }
        }

    }
    else if (mesh_.nSolutionD() == 2)
    {
        nRay_ = 4*nPhi_;
        refDiscAngles_.resize(nRay_);
        const scalar thetai = piByTwo;
        //const scalar deltaTheta = pi;
        const scalar deltaPhi = pi/(2.0*nPhi_);
        for (label m = 1; m <= 4*nPhi_; m++)
        {
            const scalar phii = (2*m - 1)*deltaPhi/2.0;

            scalar sinTheta = Foam::sin(thetai);
            scalar cosTheta = Foam::cos(thetai);
            scalar sinPhi = Foam::sin(phii);
            scalar cosPhi = Foam::cos(phii);

            refDiscAngles_[rayI++] =
                vector(sinTheta*sinPhi, sinTheta*cosPhi, cosTheta);
        }
    }
    else
    {
        FatalErrorInFunction
            << "The reflected rays are available in 2D or 3D "
            << abort(FatalError);
    }

    const polyBoundaryMesh& patches = mesh_.boundaryMesh();

    const radiation::boundaryRadiationProperties& boundaryRadiation =
        radiation::boundaryRadiationProperties::New(mesh_);

    // global face index
    globalIndex globalNumbering(mesh_.nFaces());


    // Collect faces with t = 0, r = 0 and a > 0 to shoot rays
    // and patches to construct the triSurface
    DynamicList<point> dynCf;
    DynamicList<vector> dynNf;
    DynamicList<label> dynFacesI;
    forAll(patches, patchI)
    {
        const polyPatch& pp = patches[patchI];
        const vectorField::subField cf = pp.faceCentres();

        if (!pp.coupled() && !isA<cyclicAMIPolyPatch>(pp))
        {
            const tmp<scalarField> tt =
                boundaryRadiation.transmissivity(patchI);

            const tmp<scalarField> tr =
                boundaryRadiation.specReflectivity(patchI);

            const tmp<scalarField> ta =
                boundaryRadiation.absorptivity(patchI);

            const scalarField& t = tt();
            const scalarField& r = tr();
            const scalarField& a = ta();

            const vectorField& n = pp.faceNormals();

            forAll(pp, faceI)
            {
                //const vector nf(n[faceI]);
                // Opaque, non-reflective, absortived faces to shoot
                if
                (
                    t[faceI] == 0
                 && r[faceI] == 0
                 && a[faceI] > 0
                )
                {
                    dynFacesI.append(faceI + pp.start());
                    dynCf.append(cf[faceI]);
                    dynNf.append(n[faceI]);
                }

                // relfective opaque patches to build reflective surface
                // plus opaque non-reflective
                if
                (
                    (r[faceI] > 0 && t[faceI] == 0) ||
                    (t[faceI] == 0 && a[faceI] > 0 && r[faceI] == 0)
                )
                {
                    includePatches_.insert(patchI);
                }
            }
        }
    }

    shootFacesIds_.reset(new labelList(dynFacesI));
    Cfs_.reset(new pointField(dynCf));
    Nfs_.reset(new vectorField(dynNf));

    // * * * * * * * * * * * * * * *
    // Create distributedTriSurfaceMesh
    Random rndGen(653213);

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


    triSurface localSurface = triSurfaceTools::triangulate
    (
        mesh_.boundaryMesh(),
        includePatches_,
        mapTriToGlobal_
    );

    surfacesMesh_.reset
    (
        new distributedTriSurfaceMesh
        (
            IOobject
            (
                "reflectiveSurface.stl",
                mesh_.time().constant(),
                "triSurface",
                mesh_.time(),
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            localSurface,
            dict
        )
    );

    if (debug)
    {
        surfacesMesh_->searchableSurface::write();
    }
}


void Foam::faceReflecting::calculate()
{
    const radiation::boundaryRadiationProperties& boundaryRadiation =
        radiation::boundaryRadiationProperties::New(mesh_);

    label nFaces = 0;

    const polyBoundaryMesh& patches = mesh_.boundaryMesh();

    const fvBoundaryMesh& fvPatches = mesh_.boundary();

    label nBands = spectralDistribution_.size();

    // Collect reflected directions from reflecting surfaces on direct hit
    // faces
    const vector sunDir = directHitFaces_.direction();
    const labelList& directHits = directHitFaces_.rayStartFaces();

    globalIndex globalNumbering(mesh_.nFaces());

    Map<label> refFacesDirIndex;
    labelList refDisDirsIndex(nRay_, -1);

    forAll(patches, patchI)
    {
        const polyPatch& pp = patches[patchI];

        if (!pp.coupled() && !isA<cyclicAMIPolyPatch>(pp))
        {
            const tmp<scalarField> tr =
                boundaryRadiation.specReflectivity(patchI);

            const scalarField& r = tr();
            const vectorField n(fvPatches[patchI].nf());

            forAll(pp, faceI)
            {
                label globalID = faceI + pp.start();

                if (r[faceI] > 0.0 && directHits.found(globalID))
                {
                    vector refDir =
                        sunDir + 2.0*(-sunDir & n[faceI]) * n[faceI];

                    // Look for the discrete direction
                    scalar dev(-GREAT);
                    label rayIndex = -1;
                    forAll(refDiscAngles_, iDisc)
                    {
                        scalar dotProd = refDir & refDiscAngles_[iDisc];
                        if (dev < dotProd)
                        {
                            dev = dotProd;
                            rayIndex = iDisc;
                        }
                    }

                    if (rayIndex >= 0)
                    {
                        if (refDisDirsIndex[rayIndex] == -1)
                        {
                            refDisDirsIndex[rayIndex] = 1;
                        }
                    }

                    refFacesDirIndex.insert
                    (
                        globalNumbering.toGlobal(globalID),
                        rayIndex
                    );

                    nFaces++;
                }
            }
        }
    }

    // Distribute ray indexes to all proc's
    // Make sure all the processors have the same information

    Pstream::listCombineReduce(refDisDirsIndex, maxEqOp<label>());
    Pstream::mapCombineReduce(refFacesDirIndex, minEqOp<label>());

    const scalar maxBounding =
        returnReduce(5.0*mesh_.bounds().mag(), maxOp<scalar>());

    // Shoot Rays
    // From faces t = 0, r = 0 and a > 0 to all 'used' discrete reflected
    // directions

    DynamicField<point> start(nFaces);
    DynamicField<point> end(start.size());
    DynamicList<label> startIndex(start.size());
    DynamicField<label> dirStartIndex(start.size());

    label i = 0;
    do
    {
        for (; i < Cfs_->size(); i++)
        {
            const point& fc = Cfs_()[i];

            const vector nf = Nfs_()[i];

            const label myFaceId = shootFacesIds_()[i];

            forAll(refDisDirsIndex, dirIndex)
            {
                if (refDisDirsIndex[dirIndex] > -1)
                {
                    if ((nf & refDiscAngles_[dirIndex]) > 0)
                    {
                        const vector direction = -refDiscAngles_[dirIndex];

                        start.append(fc + 0.001*direction);

                        startIndex.append(myFaceId);
                        dirStartIndex.append(dirIndex);

                        end.append(fc + maxBounding*direction);
                    }
                }
            }
        }

    } while (returnReduceOr(i < Cfs_->size()));

    List<pointIndexHit> hitInfo(startIndex.size());

    surfacesMesh_->findLine(start, end, hitInfo);

    // Query the local trigId on hit faces
    labelList triangleIndex;
    autoPtr<mapDistribute> mapPtr
    (
        surfacesMesh_->localQueries
        (
            hitInfo,
            triangleIndex
        )
    );
    const mapDistribute& map = mapPtr();

    PtrList<List<scalarField>> patchr(patches.size());
    PtrList<List<scalarField>> patcha(patches.size());
    forAll(patchr, patchi)
    {
        patchr.emplace_set(patchi, nBands);
        patcha.emplace_set(patchi, nBands);
    }

    // Fill patchr
    forAll(patchr, patchi)
    {
        const polyPatch& pp = patches[patchi];

        if (!pp.coupled() && !isA<cyclicAMIPolyPatch>(pp))
        {
            for (label bandI = 0; bandI < nBands; bandI++)
            {
                patchr[patchi][bandI] =
                    boundaryRadiation.specReflectivity
                    (
                        patchi,
                        bandI,
                        new vectorField(patches[patchi].size(), sunDir)
                    );

                patcha[patchi][bandI] =
                    boundaryRadiation.absorptivity
                    (
                        patchi,
                        bandI,
                        new vectorField(patches[patchi].size(), sunDir)
                    );
            }
        }
    }

    List<scalarField> r(nBands);
    for (label bandI = 0; bandI < nBands; bandI++)
    {
        r[bandI].setSize(triangleIndex.size());
    }
    labelList refDirIndex(triangleIndex.size(), -1);
    labelList refIndex(triangleIndex.size(), -1);
    // triangleIndex includes hits on non-reflecting and reflecting faces
    forAll(triangleIndex, i)
    {
        label trii = triangleIndex[i];
        label facei = mapTriToGlobal_[trii];
        label patchI = patches.whichPatch(facei);
        const polyPatch& pp = patches[patchI];
        label localFaceI = pp.whichFace(facei);

        label globalFace = globalNumbering.toGlobal(Pstream::myProcNo(), facei);
        if (refFacesDirIndex.found(globalFace))
        {
            refDirIndex[i] = refFacesDirIndex.find(globalFace)();
            refIndex[i] = globalFace;
        }
        for (label bandI = 0; bandI < nBands; bandI++)
        {
            r[bandI][i] = patchr[patchI][bandI][localFaceI];
        }
    }
    map.reverseDistribute(hitInfo.size(), refDirIndex);
    map.reverseDistribute(hitInfo.size(), refIndex);
    for (label bandI = 0; bandI < nBands; bandI++)
    {
        map.reverseDistribute(hitInfo.size(), r[bandI]);
    }

    for (label bandI = 0; bandI < nBands; bandI++)
    {
        volScalarField::Boundary& qrefBf =
            qreflective_[bandI].boundaryFieldRef();
        qrefBf = 0.0;
    }

    const vector qPrim(solarCalc_.directSolarRad()*solarCalc_.direction());

    // Collect rays with a hit (hitting reflecting surfaces)
    // and whose reflected direction are equal to the shot ray
    forAll(hitInfo, rayI)
    {
        if (hitInfo[rayI].hit())
        {
            if
            (
                dirStartIndex[rayI] == refDirIndex[rayI]
             && refFacesDirIndex.found(refIndex[rayI])
            )
            {
                for (label bandI = 0; bandI < nBands; bandI++)
                {
                    volScalarField::Boundary& qrefBf =
                        qreflective_[bandI].boundaryFieldRef();

                    label startFaceId = startIndex[rayI];
                    label startPatchI = patches.whichPatch(startFaceId);

                    const polyPatch& ppStart = patches[startPatchI];
                    label localStartFaceI = ppStart.whichFace(startFaceId);

                    scalar a = patcha[startPatchI][bandI][localStartFaceI];

                    const vectorField& nStart = ppStart.faceNormals();

                    vector rayIn = refDiscAngles_[dirStartIndex[rayI]];

                    rayIn /= mag(rayIn);

                    qrefBf[startPatchI][localStartFaceI] +=
                    (
                        (
                            mag(qPrim)
                           *r[bandI][rayI]
                           *spectralDistribution_[bandI]
                           *a
                           *rayIn
                        )
                        & nStart[localStartFaceI]
                    );
                }
            }
        }
    }

    start.clear();
    startIndex.clear();
    end.clear();
    dirStartIndex.clear();
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::faceReflecting::faceReflecting
(
    const fvMesh& mesh,
    const faceShading& directHiyFaces,
    const solarCalculator& solar,
    const scalarList& spectralDistribution,
    const dictionary& dict
)
:
    mesh_(mesh),
    nTheta_(dict.subDict("reflecting").getOrDefault<label>("nTheta", 10)),
    nPhi_(dict.subDict("reflecting").getOrDefault<label>("nPhi", 10)),
    nRay_(0),
    refDiscAngles_(0),
    spectralDistribution_(spectralDistribution),
    qreflective_(spectralDistribution_.size()),
    directHitFaces_(directHiyFaces),
    surfacesMesh_(),
    shootFacesIds_(),
    Cfs_(),
    Nfs_(),
    solarCalc_(solar),
    includePatches_(),
    mapTriToGlobal_()
{
    initialise(dict);
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::faceReflecting::correct()
{
    calculate();
}


// ************************************************************************* //
