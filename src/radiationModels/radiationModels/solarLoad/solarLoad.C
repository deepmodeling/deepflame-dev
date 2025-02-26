/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2015 OpenFOAM Foundation
    Copyright (C) 2016-2022 OpenCFD Ltd.
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

#include "solarLoad.H"
#include "surfaceFields.H"
#include "vectorList.H"
#include "addToRunTimeSelectionTable.H"
#include "boundaryRadiationProperties.H"
//#include "gravityMeshObject.H"
#include "cyclicAMIPolyPatch.H"
#include "mappedPatchBase.H"
#include "wallPolyPatch.H"
#include "constants.H"

using namespace Foam::constant;

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    namespace radiation
    {
        defineTypeNameAndDebug(solarLoad, 0);
        addToRadiationRunTimeSelectionTables(solarLoad);
    }
}

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

void Foam::radiation::solarLoad::updateReflectedRays
(
    const labelHashSet& includePatches
)
{
    if (reflectedFaces_.empty() && hitFaces_.valid())
    {
        reflectedFaces_.reset
        (
            new faceReflecting
            (
                mesh_,
                hitFaces_(),
                solarCalc_,
                spectralDistribution_,
                dict_
            )
        );
    }

    reflectedFaces_->correct();

    volScalarField::Boundary& qrBf = qr_.boundaryFieldRef();
    const scalarField& V = mesh_.V();
    const polyBoundaryMesh& patches = mesh_.boundaryMesh();

    forAll(qrBf, patchID)
    {
        if (includePatches[patchID])
        {
            for (label bandi = 0; bandi < nBands_; ++bandi)
            {
                qrBf[patchID] +=
                reflectedFaces_->qreflective(bandi).boundaryField()[patchID];
            }
        }
        else
        {
            const scalarField& sf = mesh_.magSf().boundaryField()[patchID];
            const labelUList& cellis = patches[patchID].faceCells();

            for (label bandi = 0; bandi < nBands_; ++bandi)
            {
                forAll(cellis, i)
                {
                    const label celli = cellis[i];

                    Ru_[celli] +=
                        (
                            reflectedFaces_->qreflective(bandi).
                                boundaryField()[patchID][i] * sf[i]
                        )/V[celli];
                }
            }
        }
    }
}


bool Foam::radiation::solarLoad::updateHitFaces()
{
    if (hitFaces_.empty())
    {
        const boundaryRadiationProperties& boundaryRadiation =
            boundaryRadiationProperties::New(mesh_);

        const labelList activeFaceZoneIds(boundaryRadiation.faceZoneIds());

        if (!activeFaceZoneIds.size())
        {
            hitFaces_.reset(new faceShading(mesh_, solarCalc_.direction()));
        }
        else
        {
            hitFaces_.reset
            (
                new faceShading
                (
                    mesh_,
                    faceShading::nonCoupledPatches(mesh_),
                    activeFaceZoneIds,
                    solarCalc_.direction()
                )
            );
        }

        return true;
    }
    else
    {
        switch (solarCalc_.sunDirectionModel())
        {
            case solarCalculator::mSunDirConstant:
            {
                return false;
                break;
            }
            case solarCalculator::mSunDirTracking:
            {
                const label updateIndex = label
                (
                    mesh_.time().value()/solarCalc_.sunTrackingUpdateInterval()
                );

                if (updateIndex > updateTimeIndex_)
                {
                    Info << "Updating Sun position..." << endl;
                    updateTimeIndex_ = updateIndex;
                    solarCalc_.correctSunDirection();
                    hitFaces_->direction() = solarCalc_.direction();
                    hitFaces_->correct();
                    return true;
                }
                break;
            }
        }
    }

    return false;
}


void Foam::radiation::solarLoad::updateAbsorptivity
(
    const labelHashSet& includePatches
)
{
    const boundaryRadiationProperties& boundaryRadiation =
        boundaryRadiationProperties::New(mesh_);

    //for (const label patchID : includePatches)
    //{
    forAllConstIter(labelHashSet, includePatches, iter)
    {
        const label patchID = iter.key();
        absorptivity_[patchID].setSize(nBands_);
        for (label bandi = 0; bandi < nBands_; ++bandi)
        {
            absorptivity_[patchID][bandi] =
                boundaryRadiation.absorptivity(patchID, bandi);
        }
    }
}


void Foam::radiation::solarLoad::updateDirectHitRadiation
(
    const labelList& hitFacesId,
    const labelHashSet& includeMappedPatchBasePatches
)
{
    const polyBoundaryMesh& patches = mesh_.boundaryMesh();
    const scalarField& V = mesh_.V();
    volScalarField::Boundary& qrBf = qr_.boundaryFieldRef();

    // Reset qr and qrPrimary
    qrBf = 0.0;

    for (label bandi = 0; bandi < nBands_; ++bandi)
    {
        volScalarField::Boundary& qprimaryBf =
            qprimaryRad_[bandi].boundaryFieldRef();

        qprimaryBf = 0.0;

        forAll(hitFacesId, i)
        {
            const label facei = hitFacesId[i];
            const label patchID = patches.whichPatch(facei);

            if (patchID == -1)
            {
                continue;
            }

            const polyPatch& pp = patches[patchID];
            const label localFaceI = facei - pp.start();
            const vector qPrim
            (
                solarCalc_.directSolarRad()*solarCalc_.direction()
            );

            const vectorField& n = pp.faceNormals();

            {
                qprimaryBf[patchID][localFaceI] +=
                    (qPrim & n[localFaceI])
                    * spectralDistribution_[bandi]
                    * absorptivity_[patchID][bandi]()[localFaceI];
            }

            if (includeMappedPatchBasePatches[patchID])
            {
                qrBf[patchID][localFaceI] += qprimaryBf[patchID][localFaceI];
            }
            else
            {
                const vectorField& sf = mesh_.Sf().boundaryField()[patchID];
                const label celli = pp.faceCells()[localFaceI];

                Ru_[celli] +=
                    (qPrim & sf[localFaceI])
                  * spectralDistribution_[bandi]
                  * absorptivity_[patchID][bandi]()[localFaceI]
                  / V[celli];
            }
        }
    }
}


void Foam::radiation::solarLoad::updateSkyDiffusiveRadiation
(
    const labelHashSet& includePatches,
    const labelHashSet& includeMappedPatchBasePatches
)
{
    const polyBoundaryMesh& patches = mesh_.boundaryMesh();
    const scalarField& V = mesh_.V();
    volScalarField::Boundary& qrBf = qr_.boundaryFieldRef();

    switch(solarCalc_.sunLoadModel())
    {
        case solarCalculator::mSunLoadFairWeatherConditions:
        case solarCalculator::mSunLoadTheoreticalMaximum:
        {
            //for (const label patchID : includePatches)
            //{
            forAllConstIter(labelHashSet, includePatches, iter)
            {
                const label patchID = iter.key();
                const polyPatch& pp = patches[patchID];
                const scalarField& sf = mesh_.magSf().boundaryField()[patchID];

                const vectorField n = pp.faceNormals();
                const labelList& cellids = pp.faceCells();

                // Calculate diffusive radiance
                // contribution from sky and ground
                tmp<scalarField> tdiffuseSolarRad =
                    solarCalc_.diffuseSolarRad(n);
                const scalarField& diffuseSolarRad = tdiffuseSolarRad();

                forAll(n, facei)
                {
                    const label celli = cellids[facei];
                    if (includeMappedPatchBasePatches[patchID])
                    {
                        for (label bandi = 0; bandi < nBands_; ++bandi)
                        {
                            qrBf[patchID][facei] +=
                                diffuseSolarRad[facei]
                              * spectralDistribution_[bandi]
                              * absorptivity_[patchID][bandi]()[facei];
                        }
                    }
                    else
                    {
                        for (label bandi = 0; bandi < nBands_; ++bandi)
                        {
                            Ru_[celli] +=
                                diffuseSolarRad[facei]
                              * spectralDistribution_[bandi]
                              * absorptivity_[patchID][bandi]()[facei]
                              * sf[facei]/V[celli];
                        }
                    }
                }
            }
        }
        break;

        case solarCalculator::mSunLoadConstant:
        case solarCalculator::mSunLoadTimeDependent:
        {
            //for (const label patchID : includePatches)
            //{
            forAllConstIter(labelHashSet, includePatches, iter)
            {
                const label patchID = iter.key();
                const polyPatch& pp = patches[patchID];
                const scalarField& sf = mesh_.magSf().boundaryField()[patchID];

                const labelList& cellids = pp.faceCells();
                forAll(pp, facei)
                {
                    const label celli = cellids[facei];
                    if (includeMappedPatchBasePatches[patchID])
                    {
                        for (label bandi = 0; bandi < nBands_; ++bandi)
                        {
                            qrBf[patchID][facei] +=
                                solarCalc_.diffuseSolarRad()
                              * spectralDistribution_[bandi]
                              * absorptivity_[patchID][bandi]()[facei];
                        }
                    }
                    else
                    {
                        for (label bandi = 0; bandi < nBands_; ++bandi)
                        {
                            Ru_[celli] +=
                                (
                                    spectralDistribution_[bandi]
                                  * absorptivity_[patchID][bandi]()[facei]
                                  * solarCalc_.diffuseSolarRad()
                                )*sf[facei]/V[celli];
                        }
                    }
                }
            }
            break;
        }
    }
}


void Foam::radiation::solarLoad::initialise(const dictionary& coeffs)
{
    spectralDistributions_.reset
    (
        Function1<scalarList>::New
        (
            "spectralDistribution",
            coeffs
            //&mesh_
        ).ptr()
    );

    spectralDistribution_ =
        spectralDistributions_->value(mesh_.time().timeOutputValue());

    nBands_ = spectralDistribution_.size();

    spectralDistribution_ =
        spectralDistribution_/sum(spectralDistribution_);

    qprimaryRad_.setSize(nBands_);

    forAll(qprimaryRad_, bandi)
    {
        qprimaryRad_.set
        (
            bandi,
            new volScalarField
            (
                IOobject
                (
                    "qprimaryRad_" + Foam::name(bandi) ,
                    mesh_.time().timeName(),
                    mesh_,
                    IOobject::NO_READ,
                    IOobject::AUTO_WRITE
                ),
                mesh_,
                dimensionedScalar(dimMass/pow3(dimTime), 0.0)
            )
        );
    }

    coeffs.readIfPresent("solidCoupled", solidCoupled_);
    coeffs.readIfPresent("wallCoupled", wallCoupled_);
    coeffs.readIfPresent("updateAbsorptivity", updateAbsorptivity_);
    //coeffs.readEntry("useReflectedRays", useReflectedRays_);
    coeffs.lookup("useReflectedRays") >> useReflectedRays_;

    (void) updateHitFaces();
}

/*
void Foam::radiation::solarLoad::calculateQdiff
(
    const labelHashSet& includePatches,
    const labelHashSet& includeMappedPatchBasePatches
)
{
    scalarListIOList FmyProc
    (
        IOobject
        (
            "F",
            mesh_.facesInstance(),
            mesh_,
            IOobject::MUST_READ,
            IOobject::NO_WRITE,
            IOobject::NO_REGISTER
        )
    );


    if (!coarseMesh_ && !finalAgglom_.empty())
    {
        coarseMesh_.reset
        (
            new singleCellFvMesh
            (
                IOobject
                (
                    "coarse:" + mesh_.name(),
                    mesh_.polyMesh::instance(),
                    mesh_.time(),
                    IOobject::NO_READ,
                    IOobject::NO_WRITE
                ),
                mesh_,
                finalAgglom_
            )
        );
    }

    label nLocalVFCoarseFaces = 0;
    forAll(includePatches_, i)
    {
        const label patchI = includePatches_[i];
        nLocalVFCoarseFaces += coarseMesh_->boundaryMesh()[patchI].size();
    }

    const label totalFVNCoarseFaces =
        returnReduce(nLocalVFCoarseFaces, sumOp<label>());

    // Calculate weighted absorptivity on coarse patches
    List<scalar> localCoarseEave(nLocalVFCoarseFaces);
    List<scalar> localTave(nLocalVFCoarseFaces);
    List<scalar> localCoarsePartialArea(nLocalVFCoarseFaces);
    List<vector> localCoarseNorm(nLocalVFCoarseFaces);

    scalarField compactCoarseEave(map_->constructSize(), 0.0);
    scalarField compactCoarseTave(map_->constructSize(), 0.0);

    scalarField compactCoarsePartialArea(map_->constructSize(), 0.0);

    vectorList compactCoarseNorm(map_->constructSize(), Zero);

    const boundaryRadiationProperties& boundaryRadiation =
        boundaryRadiationProperties::New(mesh_);

    coarseToFine_.setSize(includePatches_.size());

    const labelList& hitFacesId = hitFaces_->rayStartFaces();

    label startI = 0;
    label compactI = 0;
    forAll(includePatches_, i)
    {
        const label patchID = includePatches_[i];
        const polyPatch& pp = mesh_.boundaryMesh()[patchID];

        const polyPatch& cpp = coarseMesh_->boundaryMesh()[patchID];

        const labelList& agglom = finalAgglom_[patchID];

        if (agglom.size() > 0)
        {
            label nAgglom = max(agglom) + 1;
            coarseToFine_[i] = invertOneToMany(nAgglom, agglom);
        }

        // Weight emissivity by spectral distribution
        scalarField e(pp.size(), 0.0);

        for (label bandi = 0; bandi < nBands_; bandi++)
        {
            const tmp<scalarField> te =
                spectralDistribution_[bandi]
               *boundaryRadiation.diffReflectivity(patchID, bandi);
            e += te();
        }

        scalarList Eave(cpp.size(), 0.0);
        scalarList Tave(cpp.size(), 0.0);

        const scalarField& sf = mesh_.magSf().boundaryField()[patchID];
        const scalarField& Tf = T_.boundaryField()[patchID];

        const labelList& coarsePatchFace=coarseMesh_->patchFaceMap()[patchID];

        forAll(cpp, coarseI)
        {
            const label coarseFaceID = coarsePatchFace[coarseI];
            const labelList& fineFaces = coarseToFine_[i][coarseFaceID];

            UIndirectList<scalar> fineSf(sf, fineFaces);
            scalar fineArea = sum(fineSf());

            scalar fullArea = 0.0;
            forAll(fineFaces, j)
            {
                label facei = fineFaces[j];
                label globalFaceI = facei + pp.start();

                if (hitFacesId.found(globalFaceI))
                {
                    fullArea += sf[facei];
                }
                Eave[coarseI] += (e[facei]*sf[facei])/fineArea;
                Tave[coarseI] += (pow4(Tf[facei])*sf[facei])/fineArea;
            }
            localCoarsePartialArea[compactI++] = fullArea/fineArea;
        }

        SubList<scalar>
        (
            localCoarseEave,
            Eave.size(),
            startI
        ) = Eave;

        SubList<scalar>
        (
            localTave,
            Tave.size(),
            startI
        ) = Tave;

        const vectorList coarseNSf = cpp.faceNormals();
        SubList<vector>
        (
            localCoarseNorm,
            cpp.size(),
            startI
        ) = coarseNSf;

        startI += cpp.size();
    }


    SubList<scalar>(compactCoarsePartialArea, nLocalVFCoarseFaces) =
        localCoarsePartialArea;

    SubList<scalar>(compactCoarseEave, nLocalVFCoarseFaces) =
        localCoarseEave;

    SubList<scalar>(compactCoarseTave, nLocalVFCoarseFaces) =
        localTave;

    SubList<vector>(compactCoarseNorm, nLocalVFCoarseFaces) =
        localCoarseNorm;

    map_->distribute(compactCoarsePartialArea);
    map_->distribute(compactCoarseEave);
    map_->distribute(compactCoarseTave);
    map_->distribute(compactCoarseNorm);


    // Calculate coarse hitFaces and Sun direct hit heat fluxes
    scalarList localqDiffusive(nLocalVFCoarseFaces, Zero);

    label locaFaceI = 0;
    forAll(includePatches_, i)
    {
        const label patchID = includePatches_[i];
        const polyPatch& pp = coarseMesh_->boundaryMesh()[patchID];
        const polyPatch& ppf = mesh_.boundaryMesh()[patchID];

        const labelList& coarsePatchFace = coarseMesh_->patchFaceMap()[patchID];
        const scalarField& sf = mesh_.magSf().boundaryField()[patchID];


        scalarField a(ppf.size(), Zero);

        for (label bandi = 0; bandi < nBands_; bandi++)
        {
            const tmp<scalarField> ta =
                spectralDistribution_[bandi]
              * absorptivity_[patchID][bandi]();
            a += ta();
        }

        forAll(pp, coarseI)
        {
            const label coarseFaceID = coarsePatchFace[coarseI];
            const labelList& fineFaces = coarseToFine_[i][coarseFaceID];
            UIndirectList<scalar> fineSf(sf, fineFaces);
            scalar fineArea = sum(fineSf());

            // // Weighting absorptivity per area on secondary diffussive flux
            scalar aAve = 0.0;
            forAll(fineFaces, j)
            {
                label facei = fineFaces[j];
                aAve += (a[facei]*sf[facei])/fineArea;
            }

            const scalarList& vf = FmyProc[locaFaceI];
            const labelList& compactFaces = visibleFaceFaces_[locaFaceI];

            forAll(compactFaces, j)
            {
                label compactI = compactFaces[j];

                scalar qin =
                    (
                        solarCalc_.directSolarRad()*solarCalc_.direction()
                      & compactCoarseNorm[compactI]
                    )*compactCoarsePartialArea[compactI];

                // q emission
                scalar qem =
                    compactCoarseEave[compactI]
                   *physicoChemical::sigma.value()
                   *compactCoarseTave[compactI];

                // compactCoarseEave is the diffussive reflected coeff
                scalar qDiff = (compactCoarseEave[compactI])*qin;

                localqDiffusive[locaFaceI] += (qDiff)*aAve*vf[j];
            }
            locaFaceI++;
        }
    }

    volScalarField::Boundary& qsBf = qsecondRad_.boundaryFieldRef();

    // Fill qsecondRad_
    label compactId = 0;
    forAll(includePatches_, i)
    {
        const label patchID = includePatches_[i];
        const polyPatch& pp = coarseMesh_->boundaryMesh()[patchID];

        if (pp.size() > 0)
        {
            scalarField& qrp = qsBf[patchID];

            const labelList& coarsePatchFace =
                coarseMesh_->patchFaceMap()[patchID];

            forAll(pp, coarseI)
            {
                const label coarseFaceID = coarsePatchFace[coarseI];
                const labelList& fineFaces = coarseToFine_[i][coarseFaceID];
                forAll(fineFaces, k)
                {
                    label facei = fineFaces[k];
                    qrp[facei] = localqDiffusive[compactId];
                }
                compactId ++;
            }
        }
    }

    const scalarField& V = mesh_.V();
    const polyBoundaryMesh& patches = mesh_.boundaryMesh();
    volScalarField::Boundary& qrBf = qr_.boundaryFieldRef();

    for (const label patchID : includePatches)
    {
        const scalarField& qSecond = qsecondRad_.boundaryField()[patchID];
        if (includeMappedPatchBasePatches[patchID])
        {
            qrBf[patchID] += qSecond;
        }
        else
        {
            const polyPatch& pp = patches[patchID];
            const labelList& cellids = pp.faceCells();
            const scalarField& sf = mesh_.magSf().boundaryField()[patchID];
            forAll(pp, facei)
            {
                const label celli = cellids[facei];
                Ru_[celli] += qSecond[facei]*sf[facei]/V[celli];
            }
        }
    }
}
*/

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::radiation::solarLoad::solarLoad(const volScalarField& T)
:
    radiationModel(typeName, T),
    solarLoadBase(mesh_),
    solarCalc_(coeffs_, mesh_),
    dict_(coeffs_),
    qr_
    (
        IOobject
        (
            "qr",
            mesh_.time().timeName(),
            mesh_,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        mesh_,
        dimensionedScalar(dimMass/pow3(dimTime), 0.0)
    ),
    hitFaces_(),
    reflectedFaces_(),
    Ru_
    (
        IOobject
        (
            "Ru",
            mesh_.time().timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh_,
        dimensionedScalar(dimMass/dimLength/pow3(dimTime), 0.0)
    ),
    absorptivity_(mesh_.boundaryMesh().size()),
    spectralDistribution_(),
    spectralDistributions_(),
    qprimaryRad_(0),
    nBands_(0),
    updateTimeIndex_(0),
    solidCoupled_(true),
    wallCoupled_(false),
    updateAbsorptivity_(false),
    useReflectedRays_(false),
    firstIter_(true)
{
    initialise(coeffs_);
}


Foam::radiation::solarLoad::solarLoad
(
    const dictionary& dict,
    const volScalarField& T
)
:
    radiationModel(typeName, dict, T),
    solarLoadBase(mesh_),
    solarCalc_(dict, mesh_),
    dict_(dict),
    qr_
    (
        IOobject
        (
            "qr",
            mesh_.time().timeName(),
            mesh_,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        mesh_,
        dimensionedScalar(dimMass/pow3(dimTime), 0.0)
    ),
    hitFaces_(),
    reflectedFaces_(),
    Ru_
    (
        IOobject
        (
            "Ru",
            mesh_.time().timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh_,
        dimensionedScalar(dimMass/dimLength/pow3(dimTime), 0.0)
    ),
    absorptivity_(mesh_.boundaryMesh().size()),
    spectralDistribution_(),
    spectralDistributions_(),
    qprimaryRad_(0),
    nBands_(0),
    updateTimeIndex_(0),
    solidCoupled_(true),
    wallCoupled_(false),
    updateAbsorptivity_(false),
    useReflectedRays_(false),
    firstIter_(true)
{
    initialise(dict);
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

bool Foam::radiation::solarLoad::read()
{
    if (radiationModel::read())
    {
        return true;
    }

    return false;
}


void Foam::radiation::solarLoad::calculate()
{
    const polyBoundaryMesh& patches = mesh_.boundaryMesh();
    labelHashSet includePatches;
    forAll(patches, patchI)
    {
        const polyPatch& pp = patches[patchI];
        if (!pp.coupled() && !isA<cyclicAMIPolyPatch>(pp))
        {
            includePatches.insert(patchI);
        }
    }

    labelHashSet includeMappedPatchBasePatches;
    forAll(patches, patchI)
    {
        const polyPatch& pp = patches[patchI];
        if
        (
            (isA<mappedPatchBase>(pp) && solidCoupled_)
         || (isA<wallPolyPatch>(pp) && wallCoupled_)
        )
        {
            includeMappedPatchBasePatches.insert(patchI);
        }
    }

    if (updateAbsorptivity_ || firstIter_)
    {
        updateAbsorptivity(includePatches);
    }

    const bool facesChanged = updateHitFaces();

    const bool timeDependentLoad =
        solarCalc_.sunLoadModel() == solarCalculator::mSunLoadTimeDependent;

    if (firstIter_ || facesChanged || timeDependentLoad)
    {
        // Reset Ru
        Ru_ = dimensionedScalar("Ru", dimMass/dimLength/pow3(dimTime), 0.0);

        solarCalc_.correctDirectSolarRad();
        solarCalc_.correctDiffuseSolarRad();

        spectralDistribution_ =
            spectralDistributions_->value(mesh_.time().value());

        spectralDistribution_ =
            spectralDistribution_/sum(spectralDistribution_);

        // Add direct hit radiation
        const labelList& hitFacesId = hitFaces_->rayStartFaces();
        updateDirectHitRadiation(hitFacesId, includeMappedPatchBasePatches);

        // Add sky diffusive radiation
        updateSkyDiffusiveRadiation
        (
            includePatches,
            includeMappedPatchBasePatches
        );

        // Add specular reflected radiation
        if (useReflectedRays_)
        {
            updateReflectedRays(includeMappedPatchBasePatches);
        }

        firstIter_ = false;
    }

    if (debug)
    {
        if (mesh_.time().writeTime())
        {
            Ru_.write();
        }
    }
}


Foam::tmp<Foam::volScalarField> Foam::radiation::solarLoad::Rp() const
{
    return volScalarField::New
    (
        "Rp",
        //IOobject::NO_REGISTER,
        mesh_,
        dimensionedScalar
        (
            dimMass/pow3(dimTime)/dimLength/pow4(dimTemperature),
            0.0
        )
    );
}


Foam::tmp<Foam::DimensionedField<Foam::scalar, Foam::volMesh>>
Foam::radiation::solarLoad::Ru() const
{
    return Ru_;
}


const Foam::solarCalculator&
Foam::radiation::solarLoad::solarCalculatorRef() const
{
    return solarCalc_;
}


const Foam::faceShading&
Foam::radiation::solarLoad::faceShadingRef() const
{
    return hitFaces_();
}


// ************************************************************************* //
