#include "GenFvMatrix.H"
#include "multivariateGaussConvectionScheme.H"
#include "gaussConvectionScheme.H"
#include "snGradScheme.H"
#include "linear.H"
#include "orthogonalSnGrad.H"
#include <typeinfo>
#include "ITstream.H"
namespace Foam{

tmp<fvScalarMatrix>
GenMatrix_E(
    const volScalarField& rho,
    volScalarField& he,
    const surfaceScalarField& phi,
    const volScalarField& K,
    const volScalarField& dpdt,
    const volScalarField& alphaEff,
    const volScalarField& diffAlphaD,
    const volVectorField& hDiffCorrFlux,
    const surfaceScalarField& linear_weights
){
    tmp<fvScalarMatrix> tfvm
    (
        new fvScalarMatrix
        (
            he,
            rho.dimensions()*he.dimensions()*dimVol/dimTime
        )
    );
    he.correctBoundaryConditions();

    fvScalarMatrix& fvm = tfvm.ref();

    typedef scalar* scalarPtr;
    typedef const scalar* constScalarPtr;
    typedef const label* constLabelPtr;

    const fvMesh& mesh = he.mesh();
    assert(mesh.moving() == false);

    const MeshSchedule& meshSchedule = getSchedule();

    const label nCells = meshSchedule.nCells();
    const label nFaces = meshSchedule.nFaces();
    const label nPatches = meshSchedule.nPatches();
    const List<MeshSchedule::PatchType> patchTypes = meshSchedule.patchTypes();
    const labelList& patchSizes = meshSchedule.patchSizes();

    const label *l = &fvm.lduAddr().lowerAddr()[0];
    const label *u = &fvm.lduAddr().upperAddr()[0];

    scalar rDeltaT = 1.0/mesh.time().deltaTValue();

    const scalar* const __restrict__ rhoPtr = &rho.primitiveField()[0];
    const scalar* const __restrict__ meshVscPtr = &mesh.Vsc()()[0];
    const scalar* const __restrict__ meshVPtr = &mesh.V()[0];
    const scalar* const __restrict__ rhoOldTimePtr = &rho.oldTime().primitiveField()[0];
    const scalar* const __restrict__ heOldTimePtr = &he.oldTime().primitiveField()[0];
    
    tmp<fv::convectionScheme<scalar>> cs_he = fv::convectionScheme<scalar>::New(mesh, phi, mesh.divScheme("div("+phi.name()+','+he.name()+')'));
    fv::gaussConvectionScheme<scalar>& gcs_he = dynamic_cast<fv::gaussConvectionScheme<scalar>&>(cs_he.ref());
    tmp<fv::convectionScheme<scalar>> cs_K = fv::convectionScheme<scalar>::New(mesh, phi, mesh.divScheme("div("+phi.name()+','+K.name()+')'));
    fv::gaussConvectionScheme<scalar>& gcs_K = dynamic_cast<fv::gaussConvectionScheme<scalar>&>(cs_K.ref());

    tmp<surfaceInterpolationScheme<vector>> surfaceInterpolationScheme_vector_linear(new linear<vector>(mesh));
    tmp<surfaceInterpolationScheme<scalar>> surfaceInterpolationScheme_linear(new linear<scalar>(mesh));
    tmp<fv::snGradScheme<scalar>> snGradScheme_orthogonal(new fv::orthogonalSnGrad<scalar>(mesh));

    tmp<surfaceScalarField> tweights_he = gcs_he.interpScheme().weights(he);
    const surfaceScalarField& weights_he = tweights_he();    


    tmp<surfaceScalarField> tweights_K = gcs_K.interpScheme().weights(K);
    const surfaceScalarField& weights_K = tweights_K();

    tmp<surfaceScalarField> tweightshDiffCorrFlux= surfaceInterpolationScheme_vector_linear().weights(hDiffCorrFlux);
    tmp<surfaceScalarField> tlaplacianWeight = surfaceInterpolationScheme_linear().weights(alphaEff);
    tmp<surfaceScalarField> tdeltaCoeffs = snGradScheme_orthogonal().deltaCoeffs(he);

    const surfaceScalarField& weightshDiffCorrFlux = tweightshDiffCorrFlux();
    const surfaceScalarField laplacianWeight = tlaplacianWeight();
    const surfaceScalarField& deltaCoeffs = tdeltaCoeffs();

    surfaceScalarField Kf(
        IOobject
        (
            "interpolate("+K.name()+')',
            K.instance(),
            K.db()
        ),
        mesh,
        mesh.Sf().dimensions()*K.dimensions()
    );

    surfaceScalarField hDiffCorrFluxf(
        IOobject
        (
            "interpolate("+hDiffCorrFlux.name()+')',
            hDiffCorrFlux.instance(),
            hDiffCorrFlux.db()
        ),
        mesh,
        mesh.Sf().dimensions()*hDiffCorrFlux.dimensions()
    );

    scalar* KfPtr = &Kf[0];
    const scalar* const __restrict__ weights_K_Ptr = &weights_K.primitiveField()[0]; // get weight
    const scalar* const __restrict__ KPtr = &K.primitiveField()[0];
    
    const scalar* const __restrict__ laplacianWeightsPtr = &laplacianWeight.primitiveField()[0];
    const scalar* const __restrict__ alphaPtr = &alphaEff.primitiveField()[0];

    scalar* hDiffCorrFluxfPtr = &hDiffCorrFluxf[0];
    const scalar* const __restrict__ weightshDiffCorrFluxPtr = &weightshDiffCorrFlux.primitiveField()[0]; // get weight

    const scalar* const __restrict__ meshSfPtr = &mesh.Sf()[0][0];
    const scalar* const __restrict__ hDiffCorrFluxPtr = &hDiffCorrFlux[0][0];

#ifdef OPT_FACE2CELL_COLORING_SCHEDULE

    {
        const XYBlock1DColoringStructuredMeshSchedule& coloringMeshSchedule = XYBlock1DColoringStructuredMeshSchedule::getXYBlock1DColoringStructuredMeshSchedule();
        const labelList& face_scheduling = coloringMeshSchedule.face_scheduling();

        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for(label face_scheduling_i = 0; face_scheduling_i < face_scheduling.size()-1; face_scheduling_i += 2){
            label face_start = face_scheduling[face_scheduling_i]; 
            label face_end = face_scheduling[face_scheduling_i+1];
            for (label f = face_start; f < face_end; ++f) {
                KfPtr[f] = weights_K_Ptr[f] * (KPtr[l[f]] - KPtr[u[f]]) + KPtr[u[f]];
                scalar Sfx = meshSfPtr[3 * f];
                scalar Sfy = meshSfPtr[3 * f + 1];
                scalar Sfz = meshSfPtr[3 * f + 2];
                scalar w = weightshDiffCorrFluxPtr[f];
                hDiffCorrFluxfPtr[f] = Sfx * (w * (hDiffCorrFluxPtr[l[f] * 3 + 0] - hDiffCorrFluxPtr[u[f] * 3 + 0]) + hDiffCorrFluxPtr[u[f] * 3 + 0]) +
                    Sfy * (w * (hDiffCorrFluxPtr[l[f] * 3 + 1] - hDiffCorrFluxPtr[u[f] * 3 + 1]) + hDiffCorrFluxPtr[u[f] * 3 + 1]) +
                    Sfz * (w * (hDiffCorrFluxPtr[l[f] * 3 + 2] - hDiffCorrFluxPtr[u[f] * 3 + 2]) + hDiffCorrFluxPtr[u[f] * 3 + 2]);
            }
        }
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for(label face_scheduling_i = 1; face_scheduling_i < face_scheduling.size(); face_scheduling_i += 2){
            label face_start = face_scheduling[face_scheduling_i]; 
            label face_end = face_scheduling[face_scheduling_i+1];
            for (label f = face_start; f < face_end; ++f) {
                KfPtr[f] = weights_K_Ptr[f] * (KPtr[l[f]] - KPtr[u[f]]) + KPtr[u[f]];
                scalar Sfx = meshSfPtr[3 * f];
                scalar Sfy = meshSfPtr[3 * f + 1];
                scalar Sfz = meshSfPtr[3 * f + 2];
                scalar w = weightshDiffCorrFluxPtr[f];
                hDiffCorrFluxfPtr[f] = Sfx * (w * (hDiffCorrFluxPtr[l[f] * 3 + 0] - hDiffCorrFluxPtr[u[f] * 3 + 0]) + hDiffCorrFluxPtr[u[f] * 3 + 0]) +
                    Sfy * (w * (hDiffCorrFluxPtr[l[f] * 3 + 1] - hDiffCorrFluxPtr[u[f] * 3 + 1]) + hDiffCorrFluxPtr[u[f] * 3 + 1]) +
                    Sfz * (w * (hDiffCorrFluxPtr[l[f] * 3 + 2] - hDiffCorrFluxPtr[u[f] * 3 + 2]) + hDiffCorrFluxPtr[u[f] * 3 + 2]);
            }
        }
    }

#else

    for (label f = 0; f < nFaces; ++f) {
        KfPtr[f] = weights_K_Ptr[f] * (KPtr[l[f]] - KPtr[u[f]]) + KPtr[u[f]];
        scalar Sfx = meshSfPtr[3 * f];
        scalar Sfy = meshSfPtr[3 * f + 1];
        scalar Sfz = meshSfPtr[3 * f + 2];
        scalar w = weightshDiffCorrFluxPtr[f];
        hDiffCorrFluxfPtr[f] = Sfx * (w * (hDiffCorrFluxPtr[l[f] * 3 + 0] - hDiffCorrFluxPtr[u[f] * 3 + 0]) + hDiffCorrFluxPtr[u[f] * 3 + 0]) +
            Sfy * (w * (hDiffCorrFluxPtr[l[f] * 3 + 1] - hDiffCorrFluxPtr[u[f] * 3 + 1]) + hDiffCorrFluxPtr[u[f] * 3 + 1]) +
            Sfz * (w * (hDiffCorrFluxPtr[l[f] * 3 + 2] - hDiffCorrFluxPtr[u[f] * 3 + 2]) + hDiffCorrFluxPtr[u[f] * 3 + 2]);
    }

#endif

    /* --------------------------------------------------------------------------------------------------------------------------------------------- */

    constScalarPtr* boundaryWeightsK = new constScalarPtr[nPatches];
    scalarPtr* boundaryKf = new scalarPtr[nPatches];
    constScalarPtr* boundaryLinearWeights = new constScalarPtr[nPatches];
    constScalarPtr* boundaryWeightshDiff = new constScalarPtr[nPatches];
    constScalarPtr* boundarySf = new constScalarPtr[nPatches];
    scalarPtr* boundaryhDiff = new scalarPtr[nPatches];
    constScalarPtr* boundaryK = new constScalarPtr[nPatches];
    scalarPtr* boundaryK_internal = new scalarPtr[nPatches];
    scalarPtr* boundaryK_neighbour = new scalarPtr[nPatches];
    constScalarPtr* boundaryAlphaEff = new constScalarPtr[nPatches];
    scalarPtr* boundaryAlphaEff_internal = new scalarPtr[nPatches];
    scalarPtr* boundaryAlphaEff_neighbour = new scalarPtr[nPatches];
    constScalarPtr* boundaryHDiffCorrFlux = new constScalarPtr[nPatches];
    scalarPtr* boundaryHDiffCorrFlux_internal = new scalarPtr[nPatches];
    scalarPtr* boundaryHDiffCorrFlux_neighbour = new scalarPtr[nPatches];

    for (label patchi = 0; patchi < nPatches; ++patchi){
        label patchSize = patchSizes[patchi];

        // const fvsPatchScalarField& pLambda = weights_K.boundaryField()[patchi];
        boundaryWeightsK[patchi] = weights_K.boundaryField()[patchi].begin();
        // fvsPatchScalarField& psf = Kf.boundaryFieldRef()[patchi];
        boundaryKf[patchi] = Kf.boundaryFieldRef()[patchi].begin();

        // const fvsPatchScalarField& pLambda = linear_weights.boundaryField()[patchi];
        boundaryLinearWeights[patchi] = linear_weights.boundaryField()[patchi].begin();

        // const fvsPatchScalarField& pLambda = weightshDiffCorrFlux.boundaryField()[patchi];
        boundaryWeightshDiff[patchi] = weightshDiffCorrFlux.boundaryField()[patchi].begin();
        // const fvsPatchVectorField& pSf = mesh.Sf().boundaryField()[patchi];
        boundarySf[patchi] = (scalar*)mesh.Sf().boundaryField()[patchi].begin();
        // fvsPatchScalarField& psf = hDiffCorrFluxf.boundaryFieldRef()[patchi];
        boundaryhDiff[patchi] = hDiffCorrFluxf.boundaryFieldRef()[patchi].begin();

        const fvPatchScalarField& patchK = K.boundaryField()[patchi];
        boundaryK[patchi] = patchK.begin();
        const fvPatchScalarField& patchAlphaEff = alphaEff.boundaryField()[patchi];
        boundaryAlphaEff[patchi] = patchAlphaEff.begin();
        const fvPatchVectorField& patchHDiffCorrFlux = hDiffCorrFlux.boundaryField()[patchi];
        boundaryHDiffCorrFlux[patchi] = (scalar*) patchHDiffCorrFlux.begin();

        if (patchTypes[patchi] == MeshSchedule::PatchType::processor){

            tmp<scalarField> tPatchKInternal = 
                    dynamic_cast<const processorFvPatchField<scalar>&>(patchK).patchInternalField();
            const scalarField& patchKInternal = tPatchKInternal();
            boundaryK_internal[patchi] = new scalar[patchSize];
            memcpy(boundaryK_internal[patchi], &patchKInternal[0], patchSize * sizeof(scalar));

            tmp<scalarField> tPatchKNeighbour = 
                    dynamic_cast<const processorFvPatchField<scalar>&>(patchK).patchNeighbourField();
            const scalarField& patchKNeighbour = tPatchKNeighbour();
            boundaryK_neighbour[patchi] = new scalar[patchSize];
            memcpy(boundaryK_neighbour[patchi], &patchKNeighbour[0], patchSize * sizeof(scalar));

            tmp<scalarField> tPatchAlphaEffInternal = 
                    dynamic_cast<const processorFvPatchField<scalar>&>(patchAlphaEff).patchInternalField();
            const scalarField& patchAlphaEffInternal = tPatchAlphaEffInternal();
            boundaryAlphaEff_internal[patchi] = new scalar[patchSize];
            memcpy(boundaryAlphaEff_internal[patchi], &patchAlphaEffInternal[0], patchSize * sizeof(scalar));

            tmp<scalarField> tPatchAlphaEffNeighbour = 
                    dynamic_cast<const processorFvPatchField<scalar>&>(patchAlphaEff).patchNeighbourField();
            const scalarField& patchAlphaEffNeighbour = tPatchAlphaEffNeighbour();
            boundaryAlphaEff_neighbour[patchi] = new scalar[patchSize];
            memcpy(boundaryAlphaEff_neighbour[patchi], &patchAlphaEffNeighbour[0], patchSize * sizeof(scalar));

            tmp<vectorField> tPatchHDiffCorrFluxInternal = 
                    dynamic_cast<const processorFvPatchField<vector>&>(patchHDiffCorrFlux).patchInternalField();
            const vectorField& patchHDiffCorrFluxInternal = tPatchHDiffCorrFluxInternal();
            boundaryHDiffCorrFlux_internal[patchi] = new scalar[patchSize * 3];
            memcpy(boundaryHDiffCorrFlux_internal[patchi], &patchHDiffCorrFluxInternal[0][0], patchSize * 3 * sizeof(scalar));

            tmp<vectorField> tPatchHDiffCorrFluxNeighbour = 
                    dynamic_cast<const processorFvPatchField<vector>&>(patchHDiffCorrFlux).patchNeighbourField();
            const vectorField& patchHDiffCorrFluxNeighbour = tPatchHDiffCorrFluxNeighbour();
            boundaryHDiffCorrFlux_neighbour[patchi] = new scalar[patchSize * 3];
            memcpy(boundaryHDiffCorrFlux_neighbour[patchi], &patchHDiffCorrFluxNeighbour[0][0], patchSize * 3 * sizeof(scalar));

        }else if (patchTypes[patchi] == MeshSchedule::PatchType::wall){

            boundaryK_internal[patchi] = nullptr;

            boundaryK_neighbour[patchi] = nullptr;

            boundaryAlphaEff_internal[patchi] = nullptr;

            boundaryAlphaEff_neighbour[patchi] = nullptr;

            boundaryHDiffCorrFlux_internal[patchi] = nullptr;

            boundaryHDiffCorrFlux_neighbour[patchi] = nullptr;

        }else{
            Pout << "patch type not supported" << endl;
            std::exit(-1);
        }

    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    for(label patchi = 0; patchi < nPatches; ++patchi){
        if (patchTypes[patchi] == MeshSchedule::PatchType::processor){
#ifdef _OPENMP
#pragma omp for
#endif
            for(label s = 0 ; s < patchSizes[patchi] ; ++s){
                boundaryKf[patchi][s] = (boundaryWeightsK[patchi][s] * boundaryK_internal[patchi][s] +
                        (1.0 - boundaryWeightsK[patchi][s]) * boundaryK_neighbour[patchi][s]);
            }
        }else if(patchTypes[patchi] == MeshSchedule::PatchType::wall){
#ifdef _OPENMP
#pragma omp for
#endif
            for(label s = 0 ; s < patchSizes[patchi] ; ++s){
                boundaryKf[patchi][s] = boundaryK[patchi][s];
            }
        }else{
            Pout << "patch type not supported" << endl;
            std::exit(-1);        
        }
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    for(label patchi = 0; patchi < nPatches; ++patchi){
        if (patchTypes[patchi] == MeshSchedule::PatchType::processor){
#ifdef _OPENMP
#pragma omp for
#endif
            for(label s = 0 ; s < patchSizes[patchi] ; ++s){
                scalar Sfx = boundarySf[patchi][s * 3 + 0];
                scalar Sfy = boundarySf[patchi][s * 3 + 1];
                scalar Sfz = boundarySf[patchi][s * 3 + 2];
                scalar w = boundaryWeightshDiff[patchi][s];
                boundaryhDiff[patchi][s] = Sfx * (w * boundaryHDiffCorrFlux_internal[patchi][s * 3 + 0] + (1.0 - w) * boundaryHDiffCorrFlux_neighbour[patchi][s * 3 + 0]) +
                        Sfy * (w * boundaryHDiffCorrFlux_internal[patchi][s * 3 + 1] + (1.0 - w) * boundaryHDiffCorrFlux_neighbour[patchi][s * 3 + 1]) +
                        Sfz * (w * boundaryHDiffCorrFlux_internal[patchi][s * 3 + 2] + (1.0 - w) * boundaryHDiffCorrFlux_neighbour[patchi][s * 3 + 2]);
            }
        }else if(patchTypes[patchi] == MeshSchedule::PatchType::wall){
#ifdef _OPENMP
#pragma omp for
#endif
            for(label s = 0 ; s < patchSizes[patchi] ; ++s){
                scalar Sfx = boundarySf[patchi][s * 3 + 0];
                scalar Sfy = boundarySf[patchi][s * 3 + 1];
                scalar Sfz = boundarySf[patchi][s * 3 + 2];
                boundaryhDiff[patchi][s] = Sfx * boundaryHDiffCorrFlux[patchi][s * 3 + 0] +
                        Sfy * boundaryHDiffCorrFlux[patchi][s * 3 + 1] +
                        Sfz * boundaryHDiffCorrFlux[patchi][s * 3 + 2];
            }
        }else{
            Pout << "patch type not supported" << endl;
            std::exit(-1);        
        }
    }

    /* --------------------------------------------------------------------------------------------------------------------------------------------- */

    tmp<surfaceScalarField> tAlphafRef = surfaceInterpolationScheme_linear().interpolate(alphaEff);
    surfaceScalarField AlphafRef = tAlphafRef.ref();
    surfaceScalarField gammaMagSf(AlphafRef * mesh.magSf());

    scalar* __restrict__ diagPtr = &fvm.diag()[0];
    scalar* __restrict__ sourcePtr = &fvm.source()[0];
    scalar* __restrict__ lowerPtr = &fvm.lower()[0];
    scalar* __restrict__ upperPtr = &fvm.upper()[0];

    const scalar* const __restrict__ weights_he_Ptr = &weights_he.primitiveField()[0]; // get weight
    const scalar* const __restrict__ phiPtr = &phi.primitiveField()[0];

    const scalar* const __restrict__ deltaCoeffsPtr = &deltaCoeffs.primitiveField()[0];
    const scalar* const __restrict__ gammaMagSfPtr = &gammaMagSf.primitiveField()[0];

    const scalar* const __restrict__ KOldTimePtr = &K.oldTime().primitiveField()[0];

    scalar *fvcDivPtr = new scalar[nCells]{0.};
    scalar *fvcDiv2Ptr = new scalar[nCells]{0.};
    scalar *diagLaplacPtr = new scalar[nCells]{0.};

#ifdef OPT_FACE2CELL_COLORING_SCHEDULE

    {
        const XYBlock1DColoringStructuredMeshSchedule& coloringMeshSchedule = XYBlock1DColoringStructuredMeshSchedule::getXYBlock1DColoringStructuredMeshSchedule();
        const labelList& face_scheduling = coloringMeshSchedule.face_scheduling();

        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for(label face_scheduling_i = 0; face_scheduling_i < face_scheduling.size()-1; face_scheduling_i += 2){
            label face_start = face_scheduling[face_scheduling_i]; 
            label face_end = face_scheduling[face_scheduling_i+1];
            for (label f = face_start; f < face_end; ++f) {
                scalar var1 = - weights_he_Ptr[f] * phiPtr[f];
                scalar var2 = - weights_he_Ptr[f] * phiPtr[f] + phiPtr[f];
                lowerPtr[f] += var1;
                upperPtr[f] += var2;
                diagPtr[l[f]] -= var1;
                diagPtr[u[f]] -= var2;
                fvcDivPtr[l[f]] += phiPtr[f] * KfPtr[f];
                fvcDivPtr[u[f]] -= phiPtr[f] * KfPtr[f];
                fvcDiv2Ptr[l[f]] += hDiffCorrFluxfPtr[f];
                fvcDiv2Ptr[u[f]] -= hDiffCorrFluxfPtr[f];
            }
        }

        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for(label face_scheduling_i = 1; face_scheduling_i < face_scheduling.size(); face_scheduling_i += 2){
            label face_start = face_scheduling[face_scheduling_i]; 
            label face_end = face_scheduling[face_scheduling_i+1];
            for (label f = face_start; f < face_end; ++f) {
                scalar var1 = - weights_he_Ptr[f] * phiPtr[f];
                scalar var2 = - weights_he_Ptr[f] * phiPtr[f] + phiPtr[f];
                lowerPtr[f] += var1;
                upperPtr[f] += var2;
                diagPtr[l[f]] -= var1;
                diagPtr[u[f]] -= var2;
                fvcDivPtr[l[f]] += phiPtr[f] * KfPtr[f];
                fvcDivPtr[u[f]] -= phiPtr[f] * KfPtr[f];
                fvcDiv2Ptr[l[f]] += hDiffCorrFluxfPtr[f];
                fvcDiv2Ptr[u[f]] -= hDiffCorrFluxfPtr[f];
            }
        }
    }

#else

    for (label f = 0; f < nFaces; ++f) {
        scalar var1 = - weights_he_Ptr[f] * phiPtr[f];
        scalar var2 = - weights_he_Ptr[f] * phiPtr[f] + phiPtr[f];
        lowerPtr[f] += var1;
        upperPtr[f] += var2;
        diagPtr[l[f]] -= var1;
        diagPtr[u[f]] -= var2;
        fvcDivPtr[l[f]] += phiPtr[f] * KfPtr[f];
        fvcDivPtr[u[f]] -= phiPtr[f] * KfPtr[f];
        fvcDiv2Ptr[l[f]] += hDiffCorrFluxfPtr[f];
        fvcDiv2Ptr[u[f]] -= hDiffCorrFluxfPtr[f];
    }

#endif

#ifdef OPT_FACE2CELL_COLORING_SCHEDULE

    {
        const XYBlock1DColoringStructuredMeshSchedule& coloringMeshSchedule = XYBlock1DColoringStructuredMeshSchedule::getXYBlock1DColoringStructuredMeshSchedule();
        const labelList& face_scheduling = coloringMeshSchedule.face_scheduling();
        
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for(label face_scheduling_i = 0; face_scheduling_i < face_scheduling.size()-1; face_scheduling_i += 2){
            label face_start = face_scheduling[face_scheduling_i]; 
            label face_end = face_scheduling[face_scheduling_i+1];
            for (label f = face_start; f < face_end; ++f) {
                scalar var1 = deltaCoeffsPtr[f] * gammaMagSfPtr[f];
                lowerPtr[f] -= var1;
                upperPtr[f] -= var1;
                diagLaplacPtr[l[f]] += var1; //negSumDiag
                diagLaplacPtr[u[f]] += var1;
            }
        }


        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for(label face_scheduling_i = 1; face_scheduling_i < face_scheduling.size(); face_scheduling_i += 2){
            label face_start = face_scheduling[face_scheduling_i]; 
            label face_end = face_scheduling[face_scheduling_i+1];
            for (label f = face_start; f < face_end; ++f) {
                scalar var1 = deltaCoeffsPtr[f] * gammaMagSfPtr[f];
                lowerPtr[f] -= var1;
                upperPtr[f] -= var1;
                diagLaplacPtr[l[f]] += var1; //negSumDiag
                diagLaplacPtr[u[f]] += var1;
            }
        }
    }

#else

    for (label f = 0; f < nFaces; ++f) {
        scalar var1 = deltaCoeffsPtr[f] * gammaMagSfPtr[f];
        lowerPtr[f] -= var1;
        upperPtr[f] -= var1;
        diagLaplacPtr[l[f]] += var1; //negSumDiag
        diagLaplacPtr[u[f]] += var1;
    }

#endif

    /* --------------------------------------------------------------------------------------------------------------------------------------------- */
    forAll(he.boundaryField(), patchi)
    {
        const fvPatchField<scalar>& psf = he.boundaryField()[patchi];
        const fvsPatchScalarField& patchFlux = phi.boundaryField()[patchi];
        const fvsPatchScalarField& pw_fvm = weights_he.boundaryField()[patchi];
        const fvsPatchScalarField& pGamma = gammaMagSf.boundaryField()[patchi];
        const fvsPatchScalarField& phDiffCorrFluxf = hDiffCorrFluxf.boundaryField()[patchi];
        const fvsPatchScalarField& pDeltaCoeffs =
            deltaCoeffs.boundaryField()[patchi];

        const fvsPatchField<scalar>& pssf = Kf.boundaryField()[patchi];
        const labelUList& pFaceCells =
            mesh.boundary()[patchi].faceCells();

        if (psf.coupled())
        {
            fvm.internalCoeffs()[patchi] =
                -pGamma*psf.gradientInternalCoeffs(pDeltaCoeffs) + patchFlux*psf.valueInternalCoeffs(pw_fvm);
            fvm.boundaryCoeffs()[patchi] =
                pGamma*psf.gradientBoundaryCoeffs(pDeltaCoeffs) - patchFlux*psf.valueBoundaryCoeffs(pw_fvm);
        }
        else
        {
            fvm.internalCoeffs()[patchi] = - pGamma*psf.gradientInternalCoeffs() + patchFlux*psf.valueInternalCoeffs(pw_fvm);
            fvm.boundaryCoeffs()[patchi] = pGamma*psf.gradientBoundaryCoeffs() - patchFlux*psf.valueBoundaryCoeffs(pw_fvm);
        }
        forAll(mesh.boundary()[patchi], facei)
        {
            fvcDivPtr[pFaceCells[facei]] += pssf[facei] * patchFlux[facei];
            fvcDiv2Ptr[pFaceCells[facei]] += phDiffCorrFluxf[facei];
        }
    }

    /* --------------------------------------------------------------------------------------------------------------------------------------------- */

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(label c = 0; c < nCells; ++c){      // cell loop
        diagPtr[c] += rDeltaT * rhoPtr[c] * meshVscPtr[c];
        diagPtr[c] += diagLaplacPtr[c];
        sourcePtr[c] += rDeltaT * rhoOldTimePtr[c] * heOldTimePtr[c] * meshVscPtr[c];
        sourcePtr[c] -= rDeltaT * (KPtr[c] * rhoPtr[c] - KOldTimePtr[c] * rhoOldTimePtr[c]) * meshVscPtr[c];
        sourcePtr[c] -= fvcDivPtr[c];
        sourcePtr[c] += dpdt[c] * meshVPtr[c];
        sourcePtr[c] -= diffAlphaD[c] * meshVscPtr[c];
        sourcePtr[c] += fvcDiv2Ptr[c];
    }

    delete[] fvcDivPtr;
    delete[] fvcDiv2Ptr;
    delete[] diagLaplacPtr;

    for (label patchi = 0; patchi < nPatches; ++patchi){
        if(patchTypes[patchi] == MeshSchedule::PatchType::processor){
            delete[] boundaryK_internal[patchi];
            delete[] boundaryK_neighbour[patchi];
            delete[] boundaryAlphaEff_internal[patchi];
            delete[] boundaryAlphaEff_neighbour[patchi];
            delete[] boundaryHDiffCorrFlux_internal[patchi];
            delete[] boundaryHDiffCorrFlux_neighbour[patchi];
        }
    }
    delete[] boundaryK_internal;
    delete[] boundaryK_neighbour;
    delete[] boundaryAlphaEff_internal;
    delete[] boundaryAlphaEff_neighbour;
    delete[] boundaryHDiffCorrFlux_internal;
    delete[] boundaryHDiffCorrFlux_neighbour;

    delete[] boundaryWeightsK;
    delete[] boundaryKf;
    delete[] boundaryLinearWeights;
    delete[] boundaryWeightshDiff;
    delete[] boundarySf;
    delete[] boundaryhDiff;
    delete[] boundaryK;
    delete[] boundaryAlphaEff;
    delete[] boundaryHDiffCorrFlux;

    return tfvm;
}
}