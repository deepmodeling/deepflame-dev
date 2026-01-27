#include "GenFvMatrix.H"
#include "multivariateGaussConvectionScheme.H"
#include "gaussConvectionScheme.H"
#include "snGradScheme.H"
#include "linear.H"
#include "orthogonalSnGrad.H"
namespace Foam{
// #undef _OPENMP
tmp<fvVectorMatrix>
GenMatrix_U(
    const volScalarField& rho,
    volVectorField& U,
    const surfaceScalarField& phi,
    const volScalarField& p, 
    compressible::turbulenceModel& turbulence
){

    clockTime clock;
    typedef scalar* scalarPtr;
    typedef const scalar* constScalarPtr;
    typedef const label* constLabelPtr;

    const fvMesh& mesh = U.mesh();
    assert(mesh.moving() == false);

    const MeshSchedule& meshSchedule = getSchedule();

    const label nCells = meshSchedule.nCells();
    const label nFaces = meshSchedule.nFaces();
    const label nPatches = meshSchedule.nPatches();
    const List<MeshSchedule::PatchType> patchTypes = meshSchedule.patchTypes();
    const labelList& patchSizes = meshSchedule.patchSizes();

    // basic variable

    scalar rDeltaT = 1.0/mesh.time().deltaTValue();
    const surfaceScalarField& weights = mesh.surfaceInterpolation::weights();
    const surfaceScalarField& deltaCoeffs = mesh.nonOrthDeltaCoeffs();
    const scalar *weightsPtr = &weights[0];
    const scalar *deltaCoeffsPtr = &deltaCoeffs[0];
    const scalar* const __restrict__ phiPtr = &phi[0];
    const scalar* const __restrict__ UOldTimePtr = &U.oldTime()[0][0];
    const scalar* const __restrict__ meshSfPtr = &mesh.Sf()[0][0];
    const scalar* const __restrict__ meshVPtr = &mesh.V()[0];
    const scalar* const __restrict__ meshMagSfPtr = &mesh.magSf()[0];
    const scalar* const __restrict__ rhoPtr = &rho.primitiveField()[0];
    const scalar* const __restrict__ rhoOldTimePtr = &rho.oldTime().primitiveField()[0];
    const scalar* const __restrict__ pPtr = &p[0];
    scalar *gradUPtr = new scalar[nCells * 9]; // nCells * 9
    
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (label i = 0; i < nCells * 9; ++i){
        gradUPtr[i] = 0.;
    }
    
    // intermediate variable

    tmp<fvVectorMatrix> tfvm
    (
        new fvVectorMatrix
        (
            U,
            rho.dimensions()*U.dimensions()*dimVol/dimTime
        )
    );
    fvVectorMatrix& fvm = tfvm.ref();
    
    scalar* __restrict__ diagPtr = &fvm.diag()[0];
    scalar* __restrict__ sourcePtr = &fvm.source()[0][0];
    scalar* __restrict__ lowerPtr = &fvm.lower()[0];
    scalar* __restrict__ upperPtr = &fvm.upper()[0];


    volScalarField gamma = (turbulence.rho()*turbulence.nuEff()).ref();
    const scalar* const __restrict__ gammaPtr = &gamma[0];

    const label *l = &fvm.lduAddr().lowerAddr()[0];
    const label *u = &fvm.lduAddr().upperAddr()[0];

    tmp<volScalarField> tNuEff = turbulence.nuEff();
    volScalarField turbNuEff = tNuEff.ref();

    /* --------------------------------------------------------------------------------------------------------------------------------------------- */
    
    // allocate memory
    scalarPtr* internalCoeffsPtr = new scalarPtr[nPatches];
    scalarPtr* boundaryCoeffsPtr = new scalarPtr[nPatches];
    scalarPtr* valueInternalCoeffs = new scalarPtr[nPatches];
    scalarPtr* valueBoundaryCoeffs = new scalarPtr[nPatches];
    scalarPtr* gradientInternalCoeffs = new scalarPtr[nPatches];
    scalarPtr* gradientBoundaryCoeffs = new scalarPtr[nPatches];

    constScalarPtr* boundaryU = new constScalarPtr[nPatches];
    scalarPtr* boundaryU_internal = new scalarPtr[nPatches];
    constScalarPtr* boundaryWeight = new constScalarPtr[nPatches];
    constScalarPtr* boundaryDeltaCoeffs = new constScalarPtr[nPatches];
    constScalarPtr* boundaryPhi = new constScalarPtr[nPatches];
    constScalarPtr* boundarySf = new constScalarPtr[nPatches];
    constScalarPtr* boundaryMagSf = new constScalarPtr[nPatches];
    constLabelPtr* faceCells = new constLabelPtr[nPatches];
    constScalarPtr* bouTurbRho = new constScalarPtr[nPatches];
    scalarPtr* bouTurbRho_internal = new scalarPtr[nPatches];
    constScalarPtr* bouTurbNuEff = new constScalarPtr[nPatches];
    scalarPtr* bouTurbNuEff_internal = new scalarPtr[nPatches];
    constScalarPtr* boundaryP = new constScalarPtr[nPatches];
    scalarPtr* boundaryP_internal = new scalarPtr[nPatches];
    scalarPtr *boundaryGradU = new scalarPtr[nPatches];
    scalarPtr* boundaryGradU_internal = new scalarPtr[nPatches];

    for (label patchi = 0; patchi < nPatches; ++patchi){
        label patchSize = patchSizes[patchi];

        internalCoeffsPtr[patchi] = new scalar[patchSize * 3];
        boundaryCoeffsPtr[patchi] = new scalar[patchSize * 3];
        valueInternalCoeffs[patchi] = new scalar[patchSize * 3];
        valueBoundaryCoeffs[patchi] = new scalar[patchSize * 3];
        gradientInternalCoeffs[patchi] = new scalar[patchSize * 3];
        gradientBoundaryCoeffs[patchi] = new scalar[patchSize * 3];

        const fvPatchVectorField& patchU = U.boundaryField()[patchi];
        const fvPatchScalarField& pTurbRho = turbulence.rho().boundaryField()[patchi];
        const fvPatchScalarField& pTurbNuEff = turbNuEff.boundaryField()[patchi];
        const fvPatchScalarField& patchP = p.boundaryField()[patchi];
        
        boundaryU[patchi] = (scalar*) patchU.begin(); // patchSize * 3
        boundaryWeight[patchi] = mesh.surfaceInterpolation::weights().boundaryField()[patchi].begin(); // patchSize
        boundaryDeltaCoeffs[patchi] = mesh.deltaCoeffs().boundaryField()[patchi].begin(); // patchSize
        boundaryPhi[patchi] = phi.boundaryField()[patchi].begin(); // patchSize
        boundarySf[patchi] = (scalar*) mesh.Sf().boundaryField()[patchi].begin(); // patchSize * 3
        boundaryMagSf[patchi] = mesh.magSf().boundaryField()[patchi].begin(); // patchSize
        faceCells[patchi] = mesh.boundary()[patchi].faceCells().begin(); // patchSize
        bouTurbRho[patchi] = pTurbRho.begin(); // patchSize
        bouTurbNuEff[patchi] = pTurbNuEff.begin(); // patchSize
        boundaryP[patchi] = patchP.begin(); // patchSize
        boundaryGradU[patchi] = new scalar[patchSize * 9](); // patchSize * 9

        if (patchTypes[patchi] == MeshSchedule::PatchType::processor){

            vectorField patchUInternal = 
                    dynamic_cast<const processorFvPatchField<vector>&>(patchU).patchInternalField()();
            boundaryU_internal[patchi] = new scalar[patchSize * 3];
            memcpy(boundaryU_internal[patchi], &patchUInternal[0][0], patchSize * 3 * sizeof(scalar));

            scalarField patchPTurbRhoInternal = 
                    dynamic_cast<const processorFvPatchField<scalar>&>(pTurbRho).patchInternalField()();
            bouTurbRho_internal[patchi] = new scalar[patchSize];
            memcpy(bouTurbRho_internal[patchi], &patchPTurbRhoInternal[0], patchSize * sizeof(scalar));

            scalarField patchPTurbNuEffInternal = 
                    dynamic_cast<const processorFvPatchField<scalar>&>(pTurbNuEff).patchInternalField()();
            bouTurbNuEff_internal[patchi] = new scalar[patchSize];
            memcpy(bouTurbNuEff_internal[patchi], &patchPTurbNuEffInternal[0], patchSize * sizeof(scalar));

            scalarField patchPInternal = 
                    dynamic_cast<const processorFvPatchField<scalar>&>(patchP).patchInternalField()();
            boundaryP_internal[patchi] = new scalar[patchSize];
            memcpy(boundaryP_internal[patchi], &patchPInternal[0], patchSize * sizeof(scalar));

            boundaryGradU_internal[patchi] = new scalar[patchSize * 9]();


        }else if(patchTypes[patchi] == MeshSchedule::PatchType::wall){

            boundaryU_internal[patchi] = nullptr;

            bouTurbRho_internal[patchi] = nullptr;

            bouTurbNuEff_internal[patchi] = nullptr;

            boundaryP_internal[patchi] = nullptr;

            boundaryGradU_internal[patchi] = nullptr;

        }else{
            Info << "patch type not supported" << endl;
            std::exit(-1);
        }
    }
    Info << "GenMatrix_U init : " << clock.timeIncrement() << endl;

    /* --------------------------------------------------------------------------------------------------------------------------------------------- */

    /**
     * update_boundary_coeffs_vector
    */

#ifdef _OPENMP
#pragma omp parallel
#endif
    for(label patchi = 0 ; patchi < nPatches ; ++patchi){
        if (patchTypes[patchi] == MeshSchedule::PatchType::wall){
#ifdef _OPENMP
#pragma omp for
#endif
            for(label s = 0 ; s < patchSizes[patchi] ; ++s){
                valueInternalCoeffs[patchi][s * 3 + 0] = 1;
                valueInternalCoeffs[patchi][s * 3 + 1] = 1;
                valueInternalCoeffs[patchi][s * 3 + 2] = 1;
                valueBoundaryCoeffs[patchi][s * 3 + 0] = 0;
                valueBoundaryCoeffs[patchi][s * 3 + 1] = 0;
                valueBoundaryCoeffs[patchi][s * 3 + 2] = 0;
                gradientInternalCoeffs[patchi][s * 3 + 0] = 0;
                gradientInternalCoeffs[patchi][s * 3 + 1] = 0;
                gradientInternalCoeffs[patchi][s * 3 + 2] = 0;
                gradientBoundaryCoeffs[patchi][s * 3 + 0] = 0;
                gradientBoundaryCoeffs[patchi][s * 3 + 1] = 0;
                gradientBoundaryCoeffs[patchi][s * 3 + 2] = 0;
            }
        }
        else if (patchTypes[patchi] == MeshSchedule::PatchType::processor){
#ifdef _OPENMP
#pragma omp for
#endif
            for(label s = 0 ; s < patchSizes[patchi] ; ++s){
                valueInternalCoeffs[patchi][s * 3 + 0] = boundaryWeight[patchi][s];
                valueInternalCoeffs[patchi][s * 3 + 1] = boundaryWeight[patchi][s];
                valueInternalCoeffs[patchi][s * 3 + 2] = boundaryWeight[patchi][s];
                valueBoundaryCoeffs[patchi][s * 3 + 0] = 1 - boundaryWeight[patchi][s];
                valueBoundaryCoeffs[patchi][s * 3 + 1] = 1 - boundaryWeight[patchi][s];
                valueBoundaryCoeffs[patchi][s * 3 + 2] = 1 - boundaryWeight[patchi][s];
                gradientInternalCoeffs[patchi][s * 3 + 0] = -1 * boundaryDeltaCoeffs[patchi][s];
                gradientInternalCoeffs[patchi][s * 3 + 1] = -1 * boundaryDeltaCoeffs[patchi][s];
                gradientInternalCoeffs[patchi][s * 3 + 2] = -1 * boundaryDeltaCoeffs[patchi][s];
                gradientBoundaryCoeffs[patchi][s * 3 + 0] = boundaryDeltaCoeffs[patchi][s];
                gradientBoundaryCoeffs[patchi][s * 3 + 1] = boundaryDeltaCoeffs[patchi][s];
                gradientBoundaryCoeffs[patchi][s * 3 + 2] = boundaryDeltaCoeffs[patchi][s];
            }
        }
        else{
            Info << "patch type not supported" << endl;
            std::exit(-1);        
        }
    }
    Info << "GenMatrix_U update boundary coeffs : " << clock.timeIncrement() << endl;

    /* --------------------------------------------------------------------------------------------------------------------------------------------- */

    /**
     * fvm_ddt_vector
    */
#ifdef _OPENMP    
#pragma omp parallel for
#endif
    for(label c = 0; c < nCells; ++c){
        diagPtr[c] += rDeltaT * rhoPtr[c] * meshVPtr[c];
        sourcePtr[c * 3] += rDeltaT * rhoOldTimePtr[c] * UOldTimePtr[c * 3] * meshVPtr[c];
        sourcePtr[1 + c * 3] += rDeltaT * rhoOldTimePtr[c] * UOldTimePtr[1 + c * 3] * meshVPtr[c];
        sourcePtr[2 + c * 3] += rDeltaT * rhoOldTimePtr[c] * UOldTimePtr[2 + c * 3] * meshVPtr[c];
    }
    Info << "GenMatrix_U ddt : " << clock.timeIncrement() << endl;

    /* --------------------------------------------------------------------------------------------------------------------------------------------- */

    /**
     * fvm_div_vector
    */

#ifdef OPT_FACE2CELL_COLORING_SCHEDULE

    {
        const XYBlock1DColoringStructuredMeshSchedule& coloringSchedule = XYBlock1DColoringStructuredMeshSchedule::getXYBlock1DColoringStructuredMeshSchedule();
        const labelList& face_scheduling = coloringSchedule.face_scheduling();

   /* fvm_div_vector_internal */
#ifdef _OPENMP    
#pragma omp parallel for
#endif
        for(label face_scheduling_i = 0; face_scheduling_i < face_scheduling.size()-1; face_scheduling_i += 2){
            label face_start = face_scheduling[face_scheduling_i]; 
            label face_end = face_scheduling[face_scheduling_i+1];
            for (label f = face_start; f < face_end; ++f) {
                scalar lower_value = - weightsPtr[f] * phiPtr[f];
                scalar upper_value = (- weightsPtr[f] + 1.) * phiPtr[f];        
                lowerPtr[f] += lower_value;
                upperPtr[f] += upper_value;
                diagPtr[l[f]] -= lower_value;
                diagPtr[u[f]] -= upper_value;
            }
        }


#ifdef _OPENMP    
#pragma omp parallel for
#endif
        for(label face_scheduling_i = 1; face_scheduling_i < face_scheduling.size(); face_scheduling_i += 2){
            label face_start = face_scheduling[face_scheduling_i]; 
            label face_end = face_scheduling[face_scheduling_i+1];
            for (label f = face_start; f < face_end; ++f) {
                scalar lower_value = - weightsPtr[f] * phiPtr[f];
                scalar upper_value = (- weightsPtr[f] + 1.) * phiPtr[f];        
                lowerPtr[f] += lower_value;
                upperPtr[f] += upper_value;
                diagPtr[l[f]] -= lower_value;
                diagPtr[u[f]] -= upper_value;
            }
        }

    }
 

#else

    for (label f = 0; f < nFaces; ++f) {
        scalar lower_value = - weightsPtr[f] * phiPtr[f];
        scalar upper_value = (- weightsPtr[f] + 1.) * phiPtr[f];        
        lowerPtr[f] += lower_value;
        upperPtr[f] += upper_value;
        diagPtr[l[f]] -= lower_value;
        diagPtr[u[f]] -= upper_value;
    }

#endif

    /* fvm_div_vector_boundary */
    for(label patchi= 0; patchi < nPatches ; ++patchi){
#ifdef _OPENMP    
#pragma omp parallel for
#endif
        for(label s = 0; s < patchSizes[patchi] ; ++s){
            internalCoeffsPtr[patchi][s * 3 + 0] = boundaryPhi[patchi][s] * valueInternalCoeffs[patchi][s * 3 + 0];
            internalCoeffsPtr[patchi][s * 3 + 1] = boundaryPhi[patchi][s] * valueInternalCoeffs[patchi][s * 3 + 1];
            internalCoeffsPtr[patchi][s * 3 + 2] = boundaryPhi[patchi][s] * valueInternalCoeffs[patchi][s * 3 + 2];
            boundaryCoeffsPtr[patchi][s * 3 + 0] = -boundaryPhi[patchi][s] * valueBoundaryCoeffs[patchi][s * 3 + 0];
            boundaryCoeffsPtr[patchi][s * 3 + 1] = -boundaryPhi[patchi][s] * valueBoundaryCoeffs[patchi][s * 3 + 1];
            boundaryCoeffsPtr[patchi][s * 3 + 2] = -boundaryPhi[patchi][s] * valueBoundaryCoeffs[patchi][s * 3 + 2];
        }
    }

    Info << "GenMatrix_U div : " << clock.timeIncrement() << endl;

    /* --------------------------------------------------------------------------------------------------------------------------------------------- */

#ifdef OPT_FACE2CELL_COLORING_SCHEDULE
    {
        const XYBlock1DColoringStructuredMeshSchedule& coloringSchedule = XYBlock1DColoringStructuredMeshSchedule::getXYBlock1DColoringStructuredMeshSchedule();
        const labelList& face_scheduling = coloringSchedule.face_scheduling();

        /**
         * fvm_laplacian_vector 
        */
        /* fvm_laplacian_internal */
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(label face_scheduling_i = 0; face_scheduling_i < face_scheduling.size()-1; face_scheduling_i += 2){
            label face_start = face_scheduling[face_scheduling_i]; 
            label face_end = face_scheduling[face_scheduling_i+1];
            for (label f = face_start; f < face_end; ++f) {
                scalar w = weightsPtr[f];
                scalar face_gamma = w * gammaPtr[l[f]] + (1 - w) * gammaPtr[u[f]];
                scalar upper_value = face_gamma * meshMagSfPtr[f] * deltaCoeffsPtr[f];
                scalar lower_value = upper_value;
                lower_value = lower_value * (-1);
                upper_value = upper_value * (-1);
                lowerPtr[f] += lower_value;
                upperPtr[f] += upper_value;
                diagPtr[l[f]] -= lower_value;
                diagPtr[u[f]] -= upper_value;
            }
        }
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(label face_scheduling_i = 1; face_scheduling_i < face_scheduling.size(); face_scheduling_i += 2){
            label face_start = face_scheduling[face_scheduling_i]; 
            label face_end = face_scheduling[face_scheduling_i+1];
            for (label f = face_start; f < face_end; ++f) {
                scalar w = weightsPtr[f];
                scalar face_gamma = w * gammaPtr[l[f]] + (1 - w) * gammaPtr[u[f]];
                scalar upper_value = face_gamma * meshMagSfPtr[f] * deltaCoeffsPtr[f];
                scalar lower_value = upper_value;
                lower_value = lower_value * (-1);
                upper_value = upper_value * (-1);
                lowerPtr[f] += lower_value;
                upperPtr[f] += upper_value;
                diagPtr[l[f]] -= lower_value;
                diagPtr[u[f]] -= upper_value;
            }
        }
    }

#else

    for (label f = 0; f < nFaces; ++f) {
        scalar w = weightsPtr[f];
        scalar face_gamma = w * gammaPtr[l[f]] + (1 - w) * gammaPtr[u[f]];
        scalar upper_value = face_gamma * meshMagSfPtr[f] * deltaCoeffsPtr[f];
        scalar lower_value = upper_value;
        lower_value = lower_value * (-1);
        upper_value = upper_value * (-1);
        lowerPtr[f] += lower_value;
        upperPtr[f] += upper_value;
        diagPtr[l[f]] -= lower_value;
        diagPtr[u[f]] -= upper_value;
    }

#endif
    
    /*  fvm_laplacian_vector_boundary */
    for(label patchi= 0; patchi < nPatches ; ++patchi){
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(label s = 0; s < patchSizes[patchi] ; ++s){
            internalCoeffsPtr[patchi][s * 3 + 0] += (-1) * bouTurbRho[patchi][s] * bouTurbNuEff[patchi][s] * boundaryMagSf[patchi][s] * gradientInternalCoeffs[patchi][s * 3 + 0];
            internalCoeffsPtr[patchi][s * 3 + 1] += (-1) * bouTurbRho[patchi][s] * bouTurbNuEff[patchi][s] * boundaryMagSf[patchi][s] * gradientInternalCoeffs[patchi][s * 3 + 1];
            internalCoeffsPtr[patchi][s * 3 + 2] += (-1) * bouTurbRho[patchi][s] * bouTurbNuEff[patchi][s] * boundaryMagSf[patchi][s] * gradientInternalCoeffs[patchi][s * 3 + 2];
            boundaryCoeffsPtr[patchi][s * 3 + 0] -= (-1) * bouTurbRho[patchi][s] * bouTurbNuEff[patchi][s] * boundaryMagSf[patchi][s] * gradientBoundaryCoeffs[patchi][s * 3 + 0];
            boundaryCoeffsPtr[patchi][s * 3 + 1] -= (-1) * bouTurbRho[patchi][s] * bouTurbNuEff[patchi][s] * boundaryMagSf[patchi][s] * gradientBoundaryCoeffs[patchi][s * 3 + 1];
            boundaryCoeffsPtr[patchi][s * 3 + 2] -= (-1) * bouTurbRho[patchi][s] * bouTurbNuEff[patchi][s] * boundaryMagSf[patchi][s] * gradientBoundaryCoeffs[patchi][s * 3 + 2];
            
        }
    }
    Info << "GenMatrix_U laplacian : " << clock.timeIncrement() << endl;

    /* --------------------------------------------------------------------------------------------------------------------------------------------- */

    /**
     *  fvc_grad_vector
    */

#ifdef OPT_FACE2CELL_COLORING_SCHEDULE

    {

        const XYBlock1DColoringStructuredMeshSchedule& coloringSchedule = XYBlock1DColoringStructuredMeshSchedule::getXYBlock1DColoringStructuredMeshSchedule();
        const labelList& face_scheduling = coloringSchedule.face_scheduling();

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(label face_scheduling_i = 0; face_scheduling_i < face_scheduling.size()-1; face_scheduling_i += 2){
            label face_start = face_scheduling[face_scheduling_i]; 
            label face_end = face_scheduling[face_scheduling_i+1];
            for (label f = face_start; f < face_end; ++f) {
                scalar ssfx = weightsPtr[f] * (UOldTimePtr[3 * l[f]] - UOldTimePtr[3 * u[f]]) + UOldTimePtr[3 * u[f]];
                scalar ssfy = weightsPtr[f] * (UOldTimePtr[3 * l[f] + 1] - UOldTimePtr[3 * u[f] + 1]) + UOldTimePtr[3 * u[f] + 1];
                scalar ssfz = weightsPtr[f] * (UOldTimePtr[3 * l[f] + 2] - UOldTimePtr[3 * u[f] + 2]) + UOldTimePtr[3 * u[f] + 2];

                scalar Sfx = meshSfPtr[3 * f];
                scalar Sfy = meshSfPtr[3 * f + 1];
                scalar Sfz = meshSfPtr[3 * f + 2];

                gradUPtr[9 * l[f]] += Sfx * ssfx;
                gradUPtr[9 * l[f] + 1] += Sfx * ssfy;
                gradUPtr[9 * l[f] + 2] += Sfx * ssfz;
                gradUPtr[9 * l[f] + 3] += Sfy * ssfx;
                gradUPtr[9 * l[f] + 4] += Sfy * ssfy;
                gradUPtr[9 * l[f] + 5] += Sfy * ssfz;
                gradUPtr[9 * l[f] + 6] += Sfz * ssfx;
                gradUPtr[9 * l[f] + 7] += Sfz * ssfy;
                gradUPtr[9 * l[f] + 8] += Sfz * ssfz;
                gradUPtr[9 * u[f]] -= Sfx * ssfx;
                gradUPtr[9 * u[f] + 1] -= Sfx * ssfy;
                gradUPtr[9 * u[f] + 2] -= Sfx * ssfz;
                gradUPtr[9 * u[f] + 3] -= Sfy * ssfx;
                gradUPtr[9 * u[f] + 4] -= Sfy * ssfy;
                gradUPtr[9 * u[f] + 5] -= Sfy * ssfz;
                gradUPtr[9 * u[f] + 6] -= Sfz * ssfx;
                gradUPtr[9 * u[f] + 7] -= Sfz * ssfy;
                gradUPtr[9 * u[f] + 8] -= Sfz * ssfz;
            }
        }

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(label face_scheduling_i = 1; face_scheduling_i < face_scheduling.size(); face_scheduling_i += 2){
            label face_start = face_scheduling[face_scheduling_i]; 
            label face_end = face_scheduling[face_scheduling_i+1];
            for (label f = face_start; f < face_end; ++f) {
                scalar ssfx = weightsPtr[f] * (UOldTimePtr[3 * l[f]] - UOldTimePtr[3 * u[f]]) + UOldTimePtr[3 * u[f]];
                scalar ssfy = weightsPtr[f] * (UOldTimePtr[3 * l[f] + 1] - UOldTimePtr[3 * u[f] + 1]) + UOldTimePtr[3 * u[f] + 1];
                scalar ssfz = weightsPtr[f] * (UOldTimePtr[3 * l[f] + 2] - UOldTimePtr[3 * u[f] + 2]) + UOldTimePtr[3 * u[f] + 2];

                scalar Sfx = meshSfPtr[3 * f];
                scalar Sfy = meshSfPtr[3 * f + 1];
                scalar Sfz = meshSfPtr[3 * f + 2];

                gradUPtr[9 * l[f]] += Sfx * ssfx;
                gradUPtr[9 * l[f] + 1] += Sfx * ssfy;
                gradUPtr[9 * l[f] + 2] += Sfx * ssfz;
                gradUPtr[9 * l[f] + 3] += Sfy * ssfx;
                gradUPtr[9 * l[f] + 4] += Sfy * ssfy;
                gradUPtr[9 * l[f] + 5] += Sfy * ssfz;
                gradUPtr[9 * l[f] + 6] += Sfz * ssfx;
                gradUPtr[9 * l[f] + 7] += Sfz * ssfy;
                gradUPtr[9 * l[f] + 8] += Sfz * ssfz;
                gradUPtr[9 * u[f]] -= Sfx * ssfx;
                gradUPtr[9 * u[f] + 1] -= Sfx * ssfy;
                gradUPtr[9 * u[f] + 2] -= Sfx * ssfz;
                gradUPtr[9 * u[f] + 3] -= Sfy * ssfx;
                gradUPtr[9 * u[f] + 4] -= Sfy * ssfy;
                gradUPtr[9 * u[f] + 5] -= Sfy * ssfz;
                gradUPtr[9 * u[f] + 6] -= Sfz * ssfx;
                gradUPtr[9 * u[f] + 7] -= Sfz * ssfy;
                gradUPtr[9 * u[f] + 8] -= Sfz * ssfz;
            }
        }

    }

#else

    for (label f = 0; f < nFaces; ++f) {
        scalar ssfx = weightsPtr[f] * (UOldTimePtr[3 * l[f]] - UOldTimePtr[3 * u[f]]) + UOldTimePtr[3 * u[f]];
        scalar ssfy = weightsPtr[f] * (UOldTimePtr[3 * l[f] + 1] - UOldTimePtr[3 * u[f] + 1]) + UOldTimePtr[3 * u[f] + 1];
        scalar ssfz = weightsPtr[f] * (UOldTimePtr[3 * l[f] + 2] - UOldTimePtr[3 * u[f] + 2]) + UOldTimePtr[3 * u[f] + 2];
        scalar Sfx = meshSfPtr[3 * f];
        scalar Sfy = meshSfPtr[3 * f + 1];
        scalar Sfz = meshSfPtr[3 * f + 2];
        gradUPtr[9 * l[f]] += Sfx * ssfx;
        gradUPtr[9 * l[f] + 1] += Sfx * ssfy;
        gradUPtr[9 * l[f] + 2] += Sfx * ssfz;
        gradUPtr[9 * l[f] + 3] += Sfy * ssfx;
        gradUPtr[9 * l[f] + 4] += Sfy * ssfy;
        gradUPtr[9 * l[f] + 5] += Sfy * ssfz;
        gradUPtr[9 * l[f] + 6] += Sfz * ssfx;
        gradUPtr[9 * l[f] + 7] += Sfz * ssfy;
        gradUPtr[9 * l[f] + 8] += Sfz * ssfz;
        gradUPtr[9 * u[f]] -= Sfx * ssfx;
        gradUPtr[9 * u[f] + 1] -= Sfx * ssfy;
        gradUPtr[9 * u[f] + 2] -= Sfx * ssfz;
        gradUPtr[9 * u[f] + 3] -= Sfy * ssfx;
        gradUPtr[9 * u[f] + 4] -= Sfy * ssfy;
        gradUPtr[9 * u[f] + 5] -= Sfy * ssfz;
        gradUPtr[9 * u[f] + 6] -= Sfz * ssfx;
        gradUPtr[9 * u[f] + 7] -= Sfz * ssfy;
        gradUPtr[9 * u[f] + 8] -= Sfz * ssfz;
    }

#endif


    /* fvc_grad_vector_internal */


#ifdef _OPENMP
#pragma omp parallel
#endif
    for(label patchi = 0; patchi < nPatches ; ++patchi){
        if(patchTypes[patchi] == MeshSchedule::PatchType::wall){
#ifdef _OPENMP
#pragma omp for
#endif
            for(label s = 0; s < patchSizes[patchi] ; ++s){
                scalar bouSfx = boundarySf[patchi][s * 3 + 0];
                scalar bouSfy = boundarySf[patchi][s * 3 + 1];
                scalar bouSfz = boundarySf[patchi][s * 3 + 2];
                scalar boussfx = boundaryU[patchi][s * 3 + 0];
                scalar boussfy = boundaryU[patchi][s * 3 + 1];
                scalar boussfz = boundaryU[patchi][s * 3 + 2];
                scalar grad_xx = bouSfx * boussfx;
                scalar grad_xy = bouSfx * boussfy;
                scalar grad_xz = bouSfx * boussfz;
                scalar grad_yx = bouSfy * boussfx;
                scalar grad_yy = bouSfy * boussfy;
                scalar grad_yz = bouSfy * boussfz;
                scalar grad_zx = bouSfz * boussfx;
                scalar grad_zy = bouSfz * boussfy;
                scalar grad_zz = bouSfz * boussfz;
                gradUPtr[faceCells[patchi][s] * 9] += grad_xx;
                gradUPtr[faceCells[patchi][s] * 9 + 1] += grad_xy;
                gradUPtr[faceCells[patchi][s] * 9 + 2] += grad_xz;
                gradUPtr[faceCells[patchi][s] * 9 + 3] += grad_yx;
                gradUPtr[faceCells[patchi][s] * 9 + 4] += grad_yy;
                gradUPtr[faceCells[patchi][s] * 9 + 5] += grad_yz;
                gradUPtr[faceCells[patchi][s] * 9 + 6] += grad_zx;
                gradUPtr[faceCells[patchi][s] * 9 + 7] += grad_zy;
                gradUPtr[faceCells[patchi][s] * 9 + 8] += grad_zz;
            }

        }else if(patchTypes[patchi] == MeshSchedule::PatchType::processor){
#ifdef _OPENMP
#pragma omp for
#endif
            for(label s = 0; s < patchSizes[patchi] ; ++s){
                scalar bouWeight = boundaryWeight[patchi][s];
                scalar bouSfx = boundarySf[patchi][s * 3 + 0];
                scalar bouSfy = boundarySf[patchi][s * 3 + 1];
                scalar bouSfz = boundarySf[patchi][s * 3 + 2];
                scalar boussfx = (1 - bouWeight) * boundaryU[patchi][s * 3 + 0] + bouWeight * boundaryU_internal[patchi][s * 3 + 0];
                scalar boussfy = (1 - bouWeight) * boundaryU[patchi][s * 3 + 1] + bouWeight * boundaryU_internal[patchi][s * 3 + 1];
                scalar boussfz = (1 - bouWeight) * boundaryU[patchi][s * 3 + 2] + bouWeight * boundaryU_internal[patchi][s * 3 + 2];
                scalar grad_xx = bouSfx * boussfx;
                scalar grad_xy = bouSfx * boussfy;
                scalar grad_xz = bouSfx * boussfz;
                scalar grad_yx = bouSfy * boussfx;
                scalar grad_yy = bouSfy * boussfy;
                scalar grad_yz = bouSfy * boussfz;
                scalar grad_zx = bouSfz * boussfx;
                scalar grad_zy = bouSfz * boussfy;
                scalar grad_zz = bouSfz * boussfz;
                gradUPtr[faceCells[patchi][s] * 9] += grad_xx;
                gradUPtr[faceCells[patchi][s] * 9 + 1] += grad_xy;
                gradUPtr[faceCells[patchi][s] * 9 + 2] += grad_xz;
                gradUPtr[faceCells[patchi][s] * 9 + 3] += grad_yx;
                gradUPtr[faceCells[patchi][s] * 9 + 4] += grad_yy;
                gradUPtr[faceCells[patchi][s] * 9 + 5] += grad_yz;
                gradUPtr[faceCells[patchi][s] * 9 + 6] += grad_zx;
                gradUPtr[faceCells[patchi][s] * 9 + 7] += grad_zy;
                gradUPtr[faceCells[patchi][s] * 9 + 8] += grad_zz;
            }
        }else {
            Info << "patch type not supported" << endl;
            std::exit(-1);        
        }
    }
    Info << "GenMatrix_U gradU : " << clock.timeIncrement() << endl;

    /* divide_cell_volume_tsr */
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(label c = 0; c < nCells; ++c){
        scalar meshVRTmp = 1. / meshVPtr[c];
        gradUPtr[9 * c] = gradUPtr[9 * c] * meshVRTmp;
        gradUPtr[9 * c + 1] = gradUPtr[9 * c + 1] * meshVRTmp;
        gradUPtr[9 * c + 2] = gradUPtr[9 * c + 2] * meshVRTmp;
        gradUPtr[9 * c + 3] = gradUPtr[9 * c + 3] * meshVRTmp;
        gradUPtr[9 * c + 4] = gradUPtr[9 * c + 4] * meshVRTmp;
        gradUPtr[9 * c + 5] = gradUPtr[9 * c + 5] * meshVRTmp;
        gradUPtr[9 * c + 6] = gradUPtr[9 * c + 6] * meshVRTmp;
        gradUPtr[9 * c + 7] = gradUPtr[9 * c + 7] * meshVRTmp;
        gradUPtr[9 * c + 8] = gradUPtr[9 * c + 8] * meshVRTmp;
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    for(label patchi = 0; patchi < nPatches; ++patchi){
        if(patchTypes[patchi] == MeshSchedule::PatchType::wall){
#ifdef _OPENMP
#pragma omp for
#endif
            for(label s = 0; s < patchSizes[patchi] ; ++s){
                scalar grad_xx = gradUPtr[faceCells[patchi][s] * 9];
                scalar grad_xy = gradUPtr[faceCells[patchi][s] * 9 + 1];
                scalar grad_xz = gradUPtr[faceCells[patchi][s] * 9 + 2];
                scalar grad_yx = gradUPtr[faceCells[patchi][s] * 9 + 3];
                scalar grad_yy = gradUPtr[faceCells[patchi][s] * 9 + 4];
                scalar grad_yz = gradUPtr[faceCells[patchi][s] * 9 + 5];
                scalar grad_zx = gradUPtr[faceCells[patchi][s] * 9 + 6];
                scalar grad_zy = gradUPtr[faceCells[patchi][s] * 9 + 7];
                scalar grad_zz = gradUPtr[faceCells[patchi][s] * 9 + 8];
                scalar n_x = boundarySf[patchi][s * 3 + 0] / boundaryMagSf[patchi][s];
                scalar n_y = boundarySf[patchi][s * 3 + 1] / boundaryMagSf[patchi][s];
                scalar n_z = boundarySf[patchi][s * 3 + 2] / boundaryMagSf[patchi][s];
                scalar grad_correction_x = - (n_x * grad_xx + n_y * grad_yx + n_z * grad_zx);
                scalar grad_correction_y = - (n_x * grad_xy + n_y * grad_yy + n_z * grad_zy);
                scalar grad_correction_z = - (n_x * grad_xz + n_y * grad_yz + n_z * grad_zz);
                boundaryGradU[patchi][s * 9 + 0] = grad_xx + n_x * grad_correction_x;
                boundaryGradU[patchi][s * 9 + 1] = grad_xy + n_x * grad_correction_y;
                boundaryGradU[patchi][s * 9 + 2] = grad_xz + n_x * grad_correction_z;
                boundaryGradU[patchi][s * 9 + 3] = grad_yx + n_y * grad_correction_x;
                boundaryGradU[patchi][s * 9 + 4] = grad_yy + n_y * grad_correction_y;
                boundaryGradU[patchi][s * 9 + 5] = grad_yz + n_y * grad_correction_z;
                boundaryGradU[patchi][s * 9 + 6] = grad_zx + n_z * grad_correction_x;
                boundaryGradU[patchi][s * 9 + 7] = grad_zy + n_z * grad_correction_y;
                boundaryGradU[patchi][s * 9 + 8] = grad_zz + n_z * grad_correction_z;
            }

        } else if (patchTypes[patchi] == MeshSchedule::PatchType::processor){
#ifdef _OPENMP
#pragma omp for
#endif
            for(label s = 0; s < patchSizes[patchi] ; ++s){
                boundaryGradU_internal[patchi][s * 9 + 0] = gradUPtr[faceCells[patchi][s] * 9];
                boundaryGradU_internal[patchi][s * 9 + 1] = gradUPtr[faceCells[patchi][s] * 9 + 1];
                boundaryGradU_internal[patchi][s * 9 + 2] = gradUPtr[faceCells[patchi][s] * 9 + 2];
                boundaryGradU_internal[patchi][s * 9 + 3] = gradUPtr[faceCells[patchi][s] * 9 + 3];
                boundaryGradU_internal[patchi][s * 9 + 4] = gradUPtr[faceCells[patchi][s] * 9 + 4];
                boundaryGradU_internal[patchi][s * 9 + 5] = gradUPtr[faceCells[patchi][s] * 9 + 5];
                boundaryGradU_internal[patchi][s * 9 + 6] = gradUPtr[faceCells[patchi][s] * 9 + 6];
                boundaryGradU_internal[patchi][s * 9 + 7] = gradUPtr[faceCells[patchi][s] * 9 + 7];
                boundaryGradU_internal[patchi][s * 9 + 8] = gradUPtr[faceCells[patchi][s] * 9 + 8];           
            }
        }
        else{
            Info << "patch type not supported" << endl;
            std::exit(-1);        
        }
    }
    for (label patchi = 0; patchi < nPatches; ++patchi) {
        if (patchTypes[patchi] == MeshSchedule::PatchType::processor){
            MPI_Sendrecv(
                boundaryGradU_internal[patchi], 9 * patchSizes[patchi], MPI_DOUBLE, neighbProcNo[patchi], 0,
                boundaryGradU[patchi], 9 * patchSizes[patchi], MPI_DOUBLE, neighbProcNo[patchi], 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );
        }
    }
    Info << "GenMatrix_U gradU boundary : " << clock.timeIncrement() << endl;

   /* --------------------------------------------------------------------------------------------------------------------------------------------- */

    /**
     *  scale_dec2T_tensor
    */
    
    /* scale_dev2t_tensor_kernel(internal) */
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (label c = 0; c < nCells; ++c)
    {
        scalar scale = gammaPtr[c]; 
        scalar val_xx = gradUPtr[9 * c];
        scalar val_xy = gradUPtr[9 * c + 1];
        scalar val_xz = gradUPtr[9 * c + 2];
        scalar val_yx = gradUPtr[9 * c + 3];
        scalar val_yy = gradUPtr[9 * c + 4];
        scalar val_yz = gradUPtr[9 * c + 5];
        scalar val_zx = gradUPtr[9 * c + 6];
        scalar val_zy = gradUPtr[9 * c + 7];
        scalar val_zz = gradUPtr[9 * c + 8];
        scalar trace_coeff = (2. / 3.) * (val_xx + val_yy + val_zz);
        gradUPtr[9 * c] = scale * (val_xx - trace_coeff);
        gradUPtr[9 * c + 1] = scale * (val_yx);
        gradUPtr[9 * c + 2] = scale * (val_zx);
        gradUPtr[9 * c + 3] = scale * (val_xy);
        gradUPtr[9 * c + 4] = scale * (val_yy - trace_coeff);
        gradUPtr[9 * c + 5] = scale * (val_zy);
        gradUPtr[9 * c + 6] = scale * (val_xz);
        gradUPtr[9 * c + 7] = scale * (val_yz);
        gradUPtr[9 * c + 8] = scale * (val_zz - trace_coeff);
    }

    /* scale_dev2t_tensor_kernel(boundary) */
    for (label patchi = 0; patchi < nPatches; ++patchi){
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (label s = 0; s < patchSizes[patchi]; ++s){
            scalar scale = bouTurbRho[patchi][s] * bouTurbNuEff[patchi][s];
            scalar val_xx = boundaryGradU[patchi][s * 9 + 0];
            scalar val_xy = boundaryGradU[patchi][s * 9 + 1];
            scalar val_xz = boundaryGradU[patchi][s * 9 + 2];
            scalar val_yx = boundaryGradU[patchi][s * 9 + 3];
            scalar val_yy = boundaryGradU[patchi][s * 9 + 4];
            scalar val_yz = boundaryGradU[patchi][s * 9 + 5];
            scalar val_zx = boundaryGradU[patchi][s * 9 + 6];
            scalar val_zy = boundaryGradU[patchi][s * 9 + 7];
            scalar val_zz = boundaryGradU[patchi][s * 9 + 8];
            scalar trace_coeff = (2. / 3.) * (val_xx + val_yy + val_zz);
            boundaryGradU[patchi][s * 9 + 0] = scale * (val_xx - trace_coeff);
            boundaryGradU[patchi][s * 9 + 1] = scale * (val_yx);
            boundaryGradU[patchi][s * 9 + 2] = scale * (val_zx);
            boundaryGradU[patchi][s * 9 + 3] = scale * (val_xy);
            boundaryGradU[patchi][s * 9 + 4] = scale * (val_yy - trace_coeff);
            boundaryGradU[patchi][s * 9 + 5] = scale * (val_zy);
            boundaryGradU[patchi][s * 9 + 6] = scale * (val_xz);
            boundaryGradU[patchi][s * 9 + 7] = scale * (val_yz);
            boundaryGradU[patchi][s * 9 + 8] = scale * (val_zz - trace_coeff);
            if (patchTypes[patchi] == MeshSchedule::PatchType::processor){
                scale = bouTurbRho_internal[patchi][s] * bouTurbNuEff_internal[patchi][s];
                val_xx = boundaryGradU_internal[patchi][s * 9 + 0];
                val_xy = boundaryGradU_internal[patchi][s * 9 + 1];
                val_xz = boundaryGradU_internal[patchi][s * 9 + 2];
                val_yx = boundaryGradU_internal[patchi][s * 9 + 3];
                val_yy = boundaryGradU_internal[patchi][s * 9 + 4];
                val_yz = boundaryGradU_internal[patchi][s * 9 + 5];
                val_zx = boundaryGradU_internal[patchi][s * 9 + 6];
                val_zy = boundaryGradU_internal[patchi][s * 9 + 7];
                val_zz = boundaryGradU_internal[patchi][s * 9 + 8];
                trace_coeff = (2. / 3.) * (val_xx + val_yy + val_zz);
                boundaryGradU_internal[patchi][s * 9 + 0] = scale * (val_xx - trace_coeff);
                boundaryGradU_internal[patchi][s * 9 + 1] = scale * (val_yx);
                boundaryGradU_internal[patchi][s * 9 + 2] = scale * (val_zx);
                boundaryGradU_internal[patchi][s * 9 + 3] = scale * (val_xy);
                boundaryGradU_internal[patchi][s * 9 + 4] = scale * (val_yy - trace_coeff);
                boundaryGradU_internal[patchi][s * 9 + 5] = scale * (val_zy);
                boundaryGradU_internal[patchi][s * 9 + 6] = scale * (val_xz);
                boundaryGradU_internal[patchi][s * 9 + 7] = scale * (val_yz);
                boundaryGradU_internal[patchi][s * 9 + 8] = scale * (val_zz - trace_coeff);
            }
        }
    }
    Info << "GenMatrix_U tensor opt : " << clock.timeIncrement() << endl;

    /* --------------------------------------------------------------------------------------------------------------------------------------------- */
    
    /**
     *  fvc_div_cell_tensor
    */

    /* fvc_div_cell_tensor_internal */
#ifdef OPT_FACE2CELL_COLORING_SCHEDULE

    {

        const XYBlock1DColoringStructuredMeshSchedule& coloringSchedule = XYBlock1DColoringStructuredMeshSchedule::getXYBlock1DColoringStructuredMeshSchedule();
        const labelList& face_scheduling = coloringSchedule.face_scheduling();

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(label face_scheduling_i = 0; face_scheduling_i < face_scheduling.size()-1; face_scheduling_i += 2){
            label face_start = face_scheduling[face_scheduling_i]; 
            label face_end = face_scheduling[face_scheduling_i+1];
            for (label f = face_start; f < face_end; ++f) {
                scalar w = weightsPtr[f];
                label owner = l[f];
                label neighbor = u[f];
                scalar Sfx = meshSfPtr[3 * f];
                scalar Sfy = meshSfPtr[3 * f + 1];
                scalar Sfz = meshSfPtr[3 * f + 2];
                scalar ssf_xx = (w * (gradUPtr[owner * 9] - gradUPtr[neighbor * 9]) + gradUPtr[neighbor * 9]);
                scalar ssf_xy = (w * (gradUPtr[owner * 9 + 1] - gradUPtr[neighbor * 9 + 1]) + gradUPtr[neighbor * 9 + 1]);
                scalar ssf_xz = (w * (gradUPtr[owner * 9 + 2] - gradUPtr[neighbor * 9 + 2]) + gradUPtr[neighbor * 9 + 2]);
                scalar ssf_yx = (w * (gradUPtr[owner * 9 + 3] - gradUPtr[neighbor * 9 + 3]) + gradUPtr[neighbor * 9 + 3]);
                scalar ssf_yy = (w * (gradUPtr[owner * 9 + 4] - gradUPtr[neighbor * 9 + 4]) + gradUPtr[neighbor * 9 + 4]);
                scalar ssf_yz = (w * (gradUPtr[owner * 9 + 5] - gradUPtr[neighbor * 9 + 5]) + gradUPtr[neighbor * 9 + 5]);
                scalar ssf_zx = (w * (gradUPtr[owner * 9 + 6] - gradUPtr[neighbor * 9 + 6]) + gradUPtr[neighbor * 9 + 6]);
                scalar ssf_zy = (w * (gradUPtr[owner * 9 + 7] - gradUPtr[neighbor * 9 + 7]) + gradUPtr[neighbor * 9 + 7]);
                scalar ssf_zz = (w * (gradUPtr[owner * 9 + 8] - gradUPtr[neighbor * 9 + 8]) + gradUPtr[neighbor * 9 + 8]);
                scalar div_x = (Sfx * ssf_xx + Sfy * ssf_yx + Sfz * ssf_zx);
                scalar div_y = (Sfx * ssf_xy + Sfy * ssf_yy + Sfz * ssf_zy);
                scalar div_z = (Sfx * ssf_xz + Sfy * ssf_yz + Sfz * ssf_zz);
                sourcePtr[owner * 3] += div_x;
                sourcePtr[owner * 3 + 1] += div_y;
                sourcePtr[owner * 3 + 2] += div_z;
                sourcePtr[neighbor * 3] -= div_x;
                sourcePtr[neighbor * 3 + 1] -= div_y;
                sourcePtr[neighbor * 3 + 2] -= div_z;
            }
        }
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(label face_scheduling_i = 1; face_scheduling_i < face_scheduling.size(); face_scheduling_i += 2){
            label face_start = face_scheduling[face_scheduling_i]; 
            label face_end = face_scheduling[face_scheduling_i+1];
            for (label f = face_start; f < face_end; ++f) {
                scalar w = weightsPtr[f];
                label owner = l[f];
                label neighbor = u[f];
                scalar Sfx = meshSfPtr[3 * f];
                scalar Sfy = meshSfPtr[3 * f + 1];
                scalar Sfz = meshSfPtr[3 * f + 2];
                scalar ssf_xx = (w * (gradUPtr[owner * 9] - gradUPtr[neighbor * 9]) + gradUPtr[neighbor * 9]);
                scalar ssf_xy = (w * (gradUPtr[owner * 9 + 1] - gradUPtr[neighbor * 9 + 1]) + gradUPtr[neighbor * 9 + 1]);
                scalar ssf_xz = (w * (gradUPtr[owner * 9 + 2] - gradUPtr[neighbor * 9 + 2]) + gradUPtr[neighbor * 9 + 2]);
                scalar ssf_yx = (w * (gradUPtr[owner * 9 + 3] - gradUPtr[neighbor * 9 + 3]) + gradUPtr[neighbor * 9 + 3]);
                scalar ssf_yy = (w * (gradUPtr[owner * 9 + 4] - gradUPtr[neighbor * 9 + 4]) + gradUPtr[neighbor * 9 + 4]);
                scalar ssf_yz = (w * (gradUPtr[owner * 9 + 5] - gradUPtr[neighbor * 9 + 5]) + gradUPtr[neighbor * 9 + 5]);
                scalar ssf_zx = (w * (gradUPtr[owner * 9 + 6] - gradUPtr[neighbor * 9 + 6]) + gradUPtr[neighbor * 9 + 6]);
                scalar ssf_zy = (w * (gradUPtr[owner * 9 + 7] - gradUPtr[neighbor * 9 + 7]) + gradUPtr[neighbor * 9 + 7]);
                scalar ssf_zz = (w * (gradUPtr[owner * 9 + 8] - gradUPtr[neighbor * 9 + 8]) + gradUPtr[neighbor * 9 + 8]);
                scalar div_x = (Sfx * ssf_xx + Sfy * ssf_yx + Sfz * ssf_zx);
                scalar div_y = (Sfx * ssf_xy + Sfy * ssf_yy + Sfz * ssf_zy);
                scalar div_z = (Sfx * ssf_xz + Sfy * ssf_yz + Sfz * ssf_zz);
                sourcePtr[owner * 3] += div_x;
                sourcePtr[owner * 3 + 1] += div_y;
                sourcePtr[owner * 3 + 2] += div_z;
                sourcePtr[neighbor * 3] -= div_x;
                sourcePtr[neighbor * 3 + 1] -= div_y;
                sourcePtr[neighbor * 3 + 2] -= div_z;
            }
        }
    }

#else

    for (label f = 0; f < nFaces; ++f) {
        scalar w = weightsPtr[f];
        label owner = l[f];
        label neighbor = u[f];
        scalar Sfx = meshSfPtr[3 * f];
        scalar Sfy = meshSfPtr[3 * f + 1];
        scalar Sfz = meshSfPtr[3 * f + 2];
        scalar ssf_xx = (w * (gradUPtr[owner * 9] - gradUPtr[neighbor * 9]) + gradUPtr[neighbor * 9]);
        scalar ssf_xy = (w * (gradUPtr[owner * 9 + 1] - gradUPtr[neighbor * 9 + 1]) + gradUPtr[neighbor * 9 + 1]);
        scalar ssf_xz = (w * (gradUPtr[owner * 9 + 2] - gradUPtr[neighbor * 9 + 2]) + gradUPtr[neighbor * 9 + 2]);
        scalar ssf_yx = (w * (gradUPtr[owner * 9 + 3] - gradUPtr[neighbor * 9 + 3]) + gradUPtr[neighbor * 9 + 3]);
        scalar ssf_yy = (w * (gradUPtr[owner * 9 + 4] - gradUPtr[neighbor * 9 + 4]) + gradUPtr[neighbor * 9 + 4]);
        scalar ssf_yz = (w * (gradUPtr[owner * 9 + 5] - gradUPtr[neighbor * 9 + 5]) + gradUPtr[neighbor * 9 + 5]);
        scalar ssf_zx = (w * (gradUPtr[owner * 9 + 6] - gradUPtr[neighbor * 9 + 6]) + gradUPtr[neighbor * 9 + 6]);
        scalar ssf_zy = (w * (gradUPtr[owner * 9 + 7] - gradUPtr[neighbor * 9 + 7]) + gradUPtr[neighbor * 9 + 7]);
        scalar ssf_zz = (w * (gradUPtr[owner * 9 + 8] - gradUPtr[neighbor * 9 + 8]) + gradUPtr[neighbor * 9 + 8]);
        scalar div_x = (Sfx * ssf_xx + Sfy * ssf_yx + Sfz * ssf_zx);
        scalar div_y = (Sfx * ssf_xy + Sfy * ssf_yy + Sfz * ssf_zy);
        scalar div_z = (Sfx * ssf_xz + Sfy * ssf_yz + Sfz * ssf_zz);
        sourcePtr[owner * 3] += div_x;
        sourcePtr[owner * 3 + 1] += div_y;
        sourcePtr[owner * 3 + 2] += div_z;
        sourcePtr[neighbor * 3] -= div_x;
        sourcePtr[neighbor * 3 + 1] -= div_y;
        sourcePtr[neighbor * 3 + 2] -= div_z;
    }

#endif

    for(label patchi = 0; patchi < nPatches; ++patchi){
        if(patchTypes[patchi] == MeshSchedule::PatchType::wall){
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(label s = 0; s < patchSizes[patchi] ; ++s){
                scalar bouSfx = boundarySf[patchi][s * 3 + 0];
                scalar bouSfy = boundarySf[patchi][s * 3 + 1];
                scalar bouSfz = boundarySf[patchi][s * 3 + 2];
                scalar boussf_xx = boundaryGradU[patchi][s * 9 + 0];
                scalar boussf_xy = boundaryGradU[patchi][s * 9 + 1];
                scalar boussf_xz = boundaryGradU[patchi][s * 9 + 2];
                scalar boussf_yx = boundaryGradU[patchi][s * 9 + 3];
                scalar boussf_yy = boundaryGradU[patchi][s * 9 + 4];
                scalar boussf_yz = boundaryGradU[patchi][s * 9 + 5];
                scalar boussf_zx = boundaryGradU[patchi][s * 9 + 6];
                scalar boussf_zy = boundaryGradU[patchi][s * 9 + 7];
                scalar boussf_zz = boundaryGradU[patchi][s * 9 + 8];
                scalar bouDiv_x = (bouSfx * boussf_xx + bouSfy * boussf_yx + bouSfz * boussf_zx);
                scalar bouDiv_y = (bouSfx * boussf_xy + bouSfy * boussf_yy + bouSfz * boussf_zy);
                scalar bouDiv_z = (bouSfx * boussf_xz + bouSfy * boussf_yz + bouSfz * boussf_zz);
                sourcePtr[faceCells[patchi][s] * 3] += bouDiv_x;
                sourcePtr[faceCells[patchi][s] * 3 + 1] += bouDiv_y;
                sourcePtr[faceCells[patchi][s] * 3 + 2] += bouDiv_z;

            }
        }else if(patchTypes[patchi] == MeshSchedule::PatchType::processor){
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (label s = 0; s < patchSizes[patchi]; ++s){
                scalar bouWeight = boundaryWeight[patchi][s];
                scalar bouSfx = boundarySf[patchi][s * 3 + 0];
                scalar bouSfy = boundarySf[patchi][s * 3 + 1];
                scalar bouSfz = boundarySf[patchi][s * 3 + 2];
                scalar boussf_xx = (1 - bouWeight) * boundaryGradU[patchi][s * 9 + 0] + bouWeight * boundaryGradU_internal[patchi][s * 9 + 0];
                scalar boussf_xy = (1 - bouWeight) * boundaryGradU[patchi][s * 9 + 1] + bouWeight * boundaryGradU_internal[patchi][s * 9 + 1];
                scalar boussf_xz = (1 - bouWeight) * boundaryGradU[patchi][s * 9 + 2] + bouWeight * boundaryGradU_internal[patchi][s * 9 + 2];
                scalar boussf_yx = (1 - bouWeight) * boundaryGradU[patchi][s * 9 + 3] + bouWeight * boundaryGradU_internal[patchi][s * 9 + 3];
                scalar boussf_yy = (1 - bouWeight) * boundaryGradU[patchi][s * 9 + 4] + bouWeight * boundaryGradU_internal[patchi][s * 9 + 4];
                scalar boussf_yz = (1 - bouWeight) * boundaryGradU[patchi][s * 9 + 5] + bouWeight * boundaryGradU_internal[patchi][s * 9 + 5];
                scalar boussf_zx = (1 - bouWeight) * boundaryGradU[patchi][s * 9 + 6] + bouWeight * boundaryGradU_internal[patchi][s * 9 + 6];
                scalar boussf_zy = (1 - bouWeight) * boundaryGradU[patchi][s * 9 + 7] + bouWeight * boundaryGradU_internal[patchi][s * 9 + 7];
                scalar boussf_zz = (1 - bouWeight) * boundaryGradU[patchi][s * 9 + 8] + bouWeight * boundaryGradU_internal[patchi][s * 9 + 8];
                scalar bouDiv_x = (bouSfx * boussf_xx + bouSfy * boussf_yx + bouSfz * boussf_zx);
                scalar bouDiv_y = (bouSfx * boussf_xy + bouSfy * boussf_yy + bouSfz * boussf_zy);
                scalar bouDiv_z = (bouSfx * boussf_xz + bouSfy * boussf_yz + bouSfz * boussf_zz);
                label cellIndex = faceCells[patchi][s];
                sourcePtr[cellIndex * 3] += bouDiv_x;
                sourcePtr[cellIndex * 3 + 1] += bouDiv_y;
                sourcePtr[cellIndex * 3 + 2] += bouDiv_z;
            }
        }else{
            Info << "patch type not supported" << endl;
            std::exit(-1);        
        }
    }

    Info << "GenMatrix_U fvc div : " << clock.timeIncrement() << endl;

    /* --------------------------------------------------------------------------------------------------------------------------------------------- */

    /**
     *  fvc_grad_cell_scalar
    */

    /* fvc_grad_scalar_internal */

#ifdef OPT_FACE2CELL_COLORING_SCHEDULE

    {

        const XYBlock1DColoringStructuredMeshSchedule& coloringSchedule = XYBlock1DColoringStructuredMeshSchedule::getXYBlock1DColoringStructuredMeshSchedule();
        const labelList& face_scheduling = coloringSchedule.face_scheduling();

        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for(label face_scheduling_i = 0; face_scheduling_i < face_scheduling.size()-1; face_scheduling_i += 2){
            label face_start = face_scheduling[face_scheduling_i]; 
            label face_end = face_scheduling[face_scheduling_i+1];
            for (label f = face_start; f < face_end; ++f) {
                scalar w = weightsPtr[f];
                label owner = l[f];
                label neighbor = u[f];
                scalar Sfx = meshSfPtr[3 * f];
                scalar Sfy = meshSfPtr[3 * f + 1];
                scalar Sfz = meshSfPtr[3 * f + 2];
                scalar ssf = (w * (pPtr[owner] - pPtr[neighbor]) + pPtr[neighbor]);
                scalar grad_x = -Sfx * ssf;
                scalar grad_y = -Sfy * ssf;
                scalar grad_z = -Sfz * ssf;
                sourcePtr[owner * 3] += grad_x;
                sourcePtr[owner * 3 + 1] += grad_y;
                sourcePtr[owner * 3 + 2] += grad_z;
                sourcePtr[neighbor * 3] -= grad_x;
                sourcePtr[neighbor * 3 + 1] -= grad_y;
                sourcePtr[neighbor * 3 + 2] -= grad_z;
            }
        }

        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for(label face_scheduling_i = 1; face_scheduling_i < face_scheduling.size(); face_scheduling_i += 2){
            label face_start = face_scheduling[face_scheduling_i]; 
            label face_end = face_scheduling[face_scheduling_i+1];
            for (label f = face_start; f < face_end; ++f) {
                scalar w = weightsPtr[f];
                label owner = l[f];
                label neighbor = u[f];
                scalar Sfx = meshSfPtr[3 * f];
                scalar Sfy = meshSfPtr[3 * f + 1];
                scalar Sfz = meshSfPtr[3 * f + 2];
                scalar ssf = (w * (pPtr[owner] - pPtr[neighbor]) + pPtr[neighbor]);
                scalar grad_x = -Sfx * ssf;
                scalar grad_y = -Sfy * ssf;
                scalar grad_z = -Sfz * ssf;
                sourcePtr[owner * 3] += grad_x;
                sourcePtr[owner * 3 + 1] += grad_y;
                sourcePtr[owner * 3 + 2] += grad_z;
                sourcePtr[neighbor * 3] -= grad_x;
                sourcePtr[neighbor * 3 + 1] -= grad_y;
                sourcePtr[neighbor * 3 + 2] -= grad_z;
            }
        }
    }

#else

    for (label f = 0; f < nFaces; ++f) {
        scalar w = weightsPtr[f];
        label owner = l[f];
        label neighbor = u[f];
        scalar Sfx = meshSfPtr[3 * f];
        scalar Sfy = meshSfPtr[3 * f + 1];
        scalar Sfz = meshSfPtr[3 * f + 2];
        scalar ssf = (w * (pPtr[owner] - pPtr[neighbor]) + pPtr[neighbor]);
        scalar grad_x = -Sfx * ssf;
        scalar grad_y = -Sfy * ssf;
        scalar grad_z = -Sfz * ssf;
        sourcePtr[owner * 3] += grad_x;
        sourcePtr[owner * 3 + 1] += grad_y;
        sourcePtr[owner * 3 + 2] += grad_z;
        sourcePtr[neighbor * 3] -= grad_x;
        sourcePtr[neighbor * 3 + 1] -= grad_y;
        sourcePtr[neighbor * 3 + 2] -= grad_z;
    }

#endif


    for(label patchi = 0; patchi < nPatches; ++patchi){
        if(patchTypes[patchi] == MeshSchedule::PatchType::wall){
            #ifdef _OPENMP
            #pragma omp parallel for
            #endif
            for (label s = 0; s < patchSizes[patchi]; ++s){
                scalar bouvf = boundaryP[patchi][s];
                scalar bouSfx = boundarySf[patchi][s * 3 + 0];
                scalar bouSfy = boundarySf[patchi][s * 3 + 1];
                scalar bouSfz = boundarySf[patchi][s * 3 + 2];
                scalar grad_x = bouSfx * bouvf;
                scalar grad_y = bouSfy * bouvf;
                scalar grad_z = bouSfz * bouvf;
                sourcePtr[faceCells[patchi][s] * 3] -= grad_x;
                sourcePtr[faceCells[patchi][s] * 3 + 1] -= grad_y;
                sourcePtr[faceCells[patchi][s] * 3 + 2] -= grad_z;
            }
        }else if(patchTypes[patchi] == MeshSchedule::PatchType::processor){
            #ifdef _OPENMP
            #pragma omp parallel for
            #endif
            for (label s = 0; s < patchSizes[patchi]; ++s){
                scalar bouWeight = boundaryWeight[patchi][s];
                scalar bouSfx = boundarySf[patchi][s * 3 + 0];
                scalar bouSfy = boundarySf[patchi][s * 3 + 1];
                scalar bouSfz = boundarySf[patchi][s * 3 + 2];
                scalar bouvf = (1 - bouWeight) * boundaryP[patchi][s] + bouWeight * boundaryP_internal[patchi][s];
                scalar grad_x = bouSfx * bouvf;
                scalar grad_y = bouSfy * bouvf;
                scalar grad_z = bouSfz * bouvf;
                sourcePtr[faceCells[patchi][s] * 3] -= grad_x;
                sourcePtr[faceCells[patchi][s] * 3 + 1] -= grad_y;
                sourcePtr[faceCells[patchi][s] * 3 + 2] -= grad_z;

            }
        }else{
            Info << "patch type not supported" << endl;
            std::exit(-1);        
        }
    }



    Info << "GenMatrix_U grad : " << clock.timeIncrement() << endl;

    /* --------------------------------------------------------------------------------------------------------------------------------------------- */
    
    for (label patchi = 0; patchi < nPatches; ++patchi){
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (label s = 0; s < patchSizes[patchi]; ++s){
            fvm.internalCoeffs()[patchi][s][0] = internalCoeffsPtr[patchi][s * 3 + 0];
            fvm.internalCoeffs()[patchi][s][1] = internalCoeffsPtr[patchi][s * 3 + 1];
            fvm.internalCoeffs()[patchi][s][2] = internalCoeffsPtr[patchi][s * 3 + 2];
            fvm.boundaryCoeffs()[patchi][s][0] = boundaryCoeffsPtr[patchi][s * 3 + 0];
            fvm.boundaryCoeffs()[patchi][s][1] = boundaryCoeffsPtr[patchi][s * 3 + 1];
            fvm.boundaryCoeffs()[patchi][s][2] = boundaryCoeffsPtr[patchi][s * 3 + 2];
        }
    }

    Info << "GenMatrix_U fvm coeffs : " << clock.timeIncrement() << endl;

    // free ptrs
    delete[] gradUPtr;

    for(label i = 0; i < nPatches; ++i){
        if (boundaryU_internal[i] != nullptr){
            delete[] boundaryU_internal[i];
            delete[] bouTurbRho_internal[i];
            delete[] bouTurbNuEff_internal[i];
            delete[] boundaryGradU_internal[i];
            delete[] boundaryP_internal[i];
        }
        delete[] internalCoeffsPtr[i];
        delete[] boundaryCoeffsPtr[i];
        delete[] valueInternalCoeffs[i];
        delete[] valueBoundaryCoeffs[i];
        delete[] gradientInternalCoeffs[i];
        delete[] gradientBoundaryCoeffs[i];
        delete[] boundaryGradU[i];
    }
    delete[] boundaryU_internal;
    delete[] bouTurbRho_internal;
    delete[] bouTurbNuEff_internal;
    delete[] boundaryGradU_internal;
    delete[] boundaryP_internal;
    delete[] internalCoeffsPtr;
    delete[] boundaryCoeffsPtr;
    delete[] valueInternalCoeffs;
    delete[] valueBoundaryCoeffs;
    delete[] gradientInternalCoeffs;
    delete[] gradientBoundaryCoeffs;
    delete[] boundaryGradU;
    delete[] boundaryWeight;
    delete[] boundarySf;
    delete[] boundaryMagSf;
    delete[] boundaryP;
    delete[] bouTurbRho;
    delete[] bouTurbNuEff;
    delete[] faceCells;
    delete[] boundaryU;

    Info << "GenMatrix_U free : " << clock.timeIncrement() << endl;
    Info << "GenMatrix_U total : " << clock.elapsedTime() << endl;

    return tfvm;
}

void getrAUandHbyA(volScalarField& rAUout, volVectorField& HbyAout, fvVectorMatrix& UEqn, volVectorField& U){
    
    clockTime clock;
    scalar* __restrict__ lowerPtr = &UEqn.lower()[0];
    scalar* __restrict__ upperPtr = &UEqn.upper()[0];
    scalar* __restrict__ diagPtr = &UEqn.diag()[0];
    scalar* __restrict__ sourcePtr = &UEqn.source()[0][0];

    const label *l = &UEqn.lduAddr().lowerAddr()[0];
    const label *u = &UEqn.lduAddr().upperAddr()[0];

    const fvMesh& mesh = U.mesh();
    assert(mesh.moving() == false);

    typedef scalar* scalarPtr;
    typedef const scalar* constScalarPtr;
    typedef const label* constLabelPtr;

    const MeshSchedule& meshSchedule = getSchedule();

    const label nCells = meshSchedule.nCells();
    const label nFaces = meshSchedule.nFaces();
    const label nPatches = meshSchedule.nPatches();
    const List<MeshSchedule::PatchType> patchTypes = meshSchedule.patchTypes();
    const labelList& patchSizes = meshSchedule.patchSizes();

    const scalar* const __restrict__ meshVPtr = &mesh.V()[0];
    const scalar* const __restrict__ UPtr = &U[0][0];

    scalarPtr* internalCoeffsPtr = new scalarPtr[nPatches];
    scalarPtr* boundaryCoeffsPtr = new scalarPtr[nPatches];

    constScalarPtr* boundaryU = new constScalarPtr[nPatches];
    constLabelPtr* faceCells = new constLabelPtr[nPatches];
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (label patchi = 0; patchi < nPatches; ++patchi){

        const fvPatchVectorField& patchU = U.boundaryField()[patchi];

        label patchSize = patchSizes[patchi];
        internalCoeffsPtr[patchi] = &UEqn.internalCoeffs()[patchi][0][0];
        boundaryCoeffsPtr[patchi] = &UEqn.boundaryCoeffs()[patchi][0][0];
        boundaryU[patchi] = (scalar*) patchU.begin(); // patchSize * 3
        faceCells[patchi] = mesh.boundary()[patchi].faceCells().begin(); // patchSize
    }

    scalarPtr rAU = rAUout.begin();
    scalarPtr* boundaryrAU = new scalarPtr[nPatches];
    scalarPtr* boundaryrAU_internal = new scalarPtr[nPatches];

    scalarPtr HbyA = (scalar*)HbyAout.begin();
    scalarPtr* boundaryHbyA = new scalarPtr[nPatches];
    scalarPtr* boundaryHbyA_internal = new scalarPtr[nPatches];

    // memcpy(rAU, &UEqn.diag()[0], sizeof(scalar) * nCells);
    // memcpy(HbyA, &UEqn.source()[0][0], sizeof(scalar) * nCells * 3);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (label c = 0; c < nCells; ++c) {
        rAU[c] = diagPtr[c];
        HbyA[3 * c] = sourcePtr[3 * c];
        HbyA[3 * c + 1] = sourcePtr[3 * c + 1];
        HbyA[3 * c + 2] = sourcePtr[3 * c + 2];
    }
    for (label patchi = 0; patchi < nPatches; ++patchi){
        label patchSize = patchSizes[patchi];
        boundaryrAU[patchi] = (scalar*)rAUout.boundaryField()[patchi].begin();
        boundaryHbyA[patchi] = (scalar*)HbyAout.boundaryField()[patchi].begin();
        if (patchTypes[patchi] == MeshSchedule::PatchType::processor){
            boundaryrAU_internal[patchi] = new scalar[patchSize]();
            boundaryHbyA_internal[patchi] = new scalar[patchSize * 3]();
        }else if(patchTypes[patchi] == MeshSchedule::PatchType::wall){
            boundaryrAU_internal[patchi] = nullptr;
            boundaryHbyA_internal[patchi] = nullptr;
        }else{
            Info << "patch type not supported" << endl;
            std::exit(-1);
        }

    }
    Info << "prepreprocess_p init : " << clock.timeIncrement() << endl;

    /* addAveInternaltoDiagUeqn */
    for (label patchi = 0; patchi < nPatches; ++patchi){
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (label s = 0; s < patchSizes[patchi]; ++s){
            label cellIndex = faceCells[patchi][s];
            scalar internal_x = internalCoeffsPtr[patchi][s * 3 + 0];
            scalar internal_y = internalCoeffsPtr[patchi][s * 3 + 1];
            scalar internal_z = internalCoeffsPtr[patchi][s * 3 + 2];
            scalar ave_internal = (internal_x + internal_y + internal_z)/3;
            rAU[cellIndex] += ave_internal;
        }
    }
    Info << "prepreprocess_p addAveInternaltoDiagUeqn : " << clock.timeIncrement() << endl;

    /* divide_cell_volume_scalar_reverse */
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (label c = 0; c < nCells; ++c){
        scalar vol = meshVPtr[c];
        rAU[c] = 1/ rAU[c] * vol;
    }

    Info << "prepreprocess_p divide_cell_volume_scalar_reverse : " << clock.timeIncrement() << endl;

    /* correct_boundary_conditions_scalar */
    for(label patchi = 0; patchi < nPatches; ++patchi){
        if(patchTypes[patchi] == MeshSchedule::PatchType::wall){
            /* correct_boundary_conditions_zeroGradient_scalar */
            for (label s = 0; s < patchSizes[patchi]; ++s){
                label cellIndex = faceCells[patchi][s];
                boundaryrAU[patchi][s] = rAU[cellIndex];
            }

        }else if(patchTypes[patchi] == MeshSchedule::PatchType::processor){
            /* correct_boundary_conditions_processor_scalar */
            /* correct_internal_boundary_field_scalar */
            for (label s = 0; s < patchSizes[patchi]; ++s){
                label cellIndex = faceCells[patchi][s];
                boundaryrAU_internal[patchi][s] = rAU[cellIndex];
            }
        }
    }

    for (label patchi = 0; patchi < nPatches; ++patchi) {
        if (patchTypes[patchi] == MeshSchedule::PatchType::processor){
            MPI_Sendrecv(
                &boundaryrAU_internal[patchi][0], patchSizes[patchi], MPI_DOUBLE, neighbProcNo[patchi], 0, 
                &boundaryrAU[patchi][0], patchSizes[patchi], MPI_DOUBLE, neighbProcNo[patchi], 0, 
                MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );
        }
    }
    Info << "prepreprocess_p correct_boundary_conditions_scalar : " << clock.timeIncrement() << endl;

/* --------------------------------------------------------------------------------------------------------------------------------------------- */
/**
 *  getHbyA
*/
    /* ueqn_addBoundaryDiag */
    for (label patchi = 0; patchi < nPatches; ++patchi){
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (label s = 0; s < patchSizes[patchi]; ++s){
            scalar internal_x = internalCoeffsPtr[patchi][s * 3 + 0];
            scalar internal_y = internalCoeffsPtr[patchi][s * 3 + 1];
            scalar internal_z = internalCoeffsPtr[patchi][s * 3 + 2];
            scalar ave_internal = (internal_x + internal_y + internal_z)/3;
            label cellIndex = faceCells[patchi][s];
            HbyA[cellIndex * 3 + 0] += ( (-internal_x + ave_internal) * UPtr[cellIndex * 3 + 0]);
            HbyA[cellIndex * 3 + 1] += ( (-internal_y + ave_internal) * UPtr[cellIndex * 3 + 1]);
            HbyA[cellIndex * 3 + 2] += ( (-internal_z + ave_internal) * UPtr[cellIndex * 3 + 2]);
        }
    }
    Info << "prepreprocess_p ueqn_addBoundaryDiag : " << clock.timeIncrement() << endl;

#ifdef OPT_FACE2CELL_COLORING_SCHEDULE

    {
        const XYBlock1DColoringStructuredMeshSchedule& coloringMeshSchedule = XYBlock1DColoringStructuredMeshSchedule::getXYBlock1DColoringStructuredMeshSchedule();
        const labelList& face_scheduling = coloringMeshSchedule.face_scheduling();
        
        /* ueqn_lduMatrix_H */
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for(label face_scheduling_i = 0; face_scheduling_i < face_scheduling.size()-1; face_scheduling_i += 2){
            label face_start = face_scheduling[face_scheduling_i]; 
            label face_end = face_scheduling[face_scheduling_i+1];
            for (label f = face_start; f < face_end; ++f) {
                HbyA[u[f] * 3 + 0] += ( -lowerPtr[f] * UPtr[l[f] * 3 + 0]);
                HbyA[u[f] * 3 + 1] += ( -lowerPtr[f] * UPtr[l[f] * 3 + 1]);
                HbyA[u[f] * 3 + 2] += ( -lowerPtr[f] * UPtr[l[f] * 3 + 2]);
                HbyA[l[f] * 3 + 0] += ( -upperPtr[f] * UPtr[u[f] * 3 + 0]);
                HbyA[l[f] * 3 + 1] += ( -upperPtr[f] * UPtr[u[f] * 3 + 1]);
                HbyA[l[f] * 3 + 2] += ( -upperPtr[f] * UPtr[u[f] * 3 + 2]);
            }
        }
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for(label face_scheduling_i = 1; face_scheduling_i < face_scheduling.size(); face_scheduling_i += 2){
            label face_start = face_scheduling[face_scheduling_i]; 
            label face_end = face_scheduling[face_scheduling_i+1];
            for (label f = face_start; f < face_end; ++f) {
                HbyA[u[f] * 3 + 0] += ( -lowerPtr[f] * UPtr[l[f] * 3 + 0]);
                HbyA[u[f] * 3 + 1] += ( -lowerPtr[f] * UPtr[l[f] * 3 + 1]);
                HbyA[u[f] * 3 + 2] += ( -lowerPtr[f] * UPtr[l[f] * 3 + 2]);
                HbyA[l[f] * 3 + 0] += ( -upperPtr[f] * UPtr[u[f] * 3 + 0]);
                HbyA[l[f] * 3 + 1] += ( -upperPtr[f] * UPtr[u[f] * 3 + 1]);
                HbyA[l[f] * 3 + 2] += ( -upperPtr[f] * UPtr[u[f] * 3 + 2]);
            }
        }
    }

#else

    for (label f = 0; f < nFaces; ++f) {
        HbyA[u[f] * 3 + 0] += ( -lowerPtr[f] * UPtr[l[f] * 3 + 0]);
        HbyA[u[f] * 3 + 1] += ( -lowerPtr[f] * UPtr[l[f] * 3 + 1]);
        HbyA[u[f] * 3 + 2] += ( -lowerPtr[f] * UPtr[l[f] * 3 + 2]);
        HbyA[l[f] * 3 + 0] += ( -upperPtr[f] * UPtr[u[f] * 3 + 0]);
        HbyA[l[f] * 3 + 1] += ( -upperPtr[f] * UPtr[u[f] * 3 + 1]);
        HbyA[l[f] * 3 + 2] += ( -upperPtr[f] * UPtr[u[f] * 3 + 2]);
    }

#endif

    Info << "prepreprocess_p ueqn_lduMatrix_H : " << clock.timeIncrement() << endl;

    for(label patchi = 0; patchi < nPatches; ++patchi){
        if(patchTypes[patchi] == MeshSchedule::PatchType::wall){
            /* ueqn_addBoundarySrc_unCoupled */
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (label s = 0; s < patchSizes[patchi]; ++s){
                scalar boundary_x = boundaryCoeffsPtr[patchi][s * 3 + 0];
                scalar boundary_y = boundaryCoeffsPtr[patchi][s * 3 + 1];
                scalar boundary_z = boundaryCoeffsPtr[patchi][s * 3 + 2];
                label cellIndex = faceCells[patchi][s];
                HbyA[cellIndex * 3 + 0] += boundary_x;
                HbyA[cellIndex * 3 + 1] += boundary_y;
                HbyA[cellIndex * 3 + 2] += boundary_z;
            }
        }else if(patchTypes[patchi] == MeshSchedule::PatchType::processor){
            /* ueqn_addBoundarySrc_processor */
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (label s = 0; s < patchSizes[patchi]; ++s){
                scalar boundary_x = boundaryCoeffsPtr[patchi][s * 3 + 0];
                scalar boundary_y = boundaryCoeffsPtr[patchi][s * 3 + 1];
                scalar boundary_z = boundaryCoeffsPtr[patchi][s * 3 + 2];
                scalar boundary_vf_x = boundaryU[patchi][s * 3 + 0];
                scalar boundary_vf_y = boundaryU[patchi][s * 3 + 1];
                scalar boundary_vf_z = boundaryU[patchi][s * 3 + 2];
                label cellIndex = faceCells[patchi][s];
                HbyA[cellIndex * 3 + 0] += boundary_x * boundary_vf_x;
                HbyA[cellIndex * 3 + 1] += boundary_y * boundary_vf_y;
                HbyA[cellIndex * 3 + 2] += boundary_z * boundary_vf_z;
            }
        }
    }
    Info << "prepreprocess_p ueqn_addBoundarySrc : " << clock.timeIncrement() << endl;

    /* ueqn_divide_cell_volume_vec */
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (label c = 0; c < nCells; ++c){
        scalar meshVRTmp = 1. / meshVPtr[c];
        HbyA[c * 3 + 0] *= meshVRTmp;
        HbyA[c * 3 + 1] *= meshVRTmp;
        HbyA[c * 3 + 2] *= meshVRTmp;
    }

    Info << "prepreprocess_p ueqn_divide_cell_volume_vec : " << clock.timeIncrement() << endl;

    /* correct_boundary_conditions_vector */
    for(label patchi = 0; patchi < nPatches; ++patchi){
        if(patchTypes[patchi] == MeshSchedule::PatchType::wall){
            /* correct_boundary_conditions_zeroGradient_vector */
            for (label s = 0; s < patchSizes[patchi]; ++s){
                label cellIndex = faceCells[patchi][s];
                boundaryHbyA[patchi][s * 3 + 0] = HbyA[cellIndex * 3 + 0];
                boundaryHbyA[patchi][s * 3 + 1] = HbyA[cellIndex * 3 + 1];
                boundaryHbyA[patchi][s * 3 + 2] = HbyA[cellIndex * 3 + 2];
            }
        }else if(patchTypes[patchi] == MeshSchedule::PatchType::processor){
            /* correct_boundary_conditions_processor_vector */
            for (label s = 0; s < patchSizes[patchi]; ++s){
                /* coorect_internal_boundary_field_vector */
                label cellIndex = faceCells[patchi][s];
                boundaryHbyA_internal[patchi][s * 3 + 0] = HbyA[cellIndex * 3 + 0];
                boundaryHbyA_internal[patchi][s * 3 + 1] = HbyA[cellIndex * 3 + 1];
                boundaryHbyA_internal[patchi][s * 3 + 2] = HbyA[cellIndex * 3 + 2];
            }
        }
    }
    Info << "prepreprocess_p correct_boundary_conditions_vector : " << clock.timeIncrement() << endl;

    for (label patchi = 0; patchi < nPatches; ++patchi) {
        if (patchTypes[patchi] == MeshSchedule::PatchType::processor){
            MPI_Sendrecv(
                &boundaryHbyA_internal[patchi][0], patchSizes[patchi] * 3, MPI_DOUBLE, neighbProcNo[patchi], 0,
                &boundaryHbyA[patchi][0], patchSizes[patchi] * 3, MPI_DOUBLE, neighbProcNo[patchi], 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );
        }
    }

    Info << "prepreprocess_p comm : " << clock.timeIncrement() << endl;


    /* scalar_field_multiply_vector_field */

    /* scalar_multiply_vector_kernel */
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (label c = 0; c < nCells; ++c){
        HbyA[c * 3 + 0] = rAU[c] * HbyA[c * 3 + 0];
        HbyA[c * 3 + 1] = rAU[c] * HbyA[c * 3 + 1];
        HbyA[c * 3 + 2] = rAU[c] * HbyA[c * 3 + 2];
    }

    for(label patchi = 0; patchi < nPatches; ++patchi){
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (label s = 0; s < patchSizes[patchi]; ++s){
            boundaryHbyA[patchi][s * 3 + 0] = boundaryrAU[patchi][s] * boundaryHbyA[patchi][s * 3 + 0];
            boundaryHbyA[patchi][s * 3 + 1] = boundaryrAU[patchi][s] * boundaryHbyA[patchi][s * 3 + 1];
            boundaryHbyA[patchi][s * 3 + 2] = boundaryrAU[patchi][s] * boundaryHbyA[patchi][s * 3 + 2];
        }
    }
    Info << "prepreprocess_p scalar_field_multiply_vector_field : " << clock.timeIncrement() << endl;
    

    for (label patchi = 0; patchi < nPatches; ++patchi){
        if (patchTypes[patchi] == MeshSchedule::PatchType::processor){
            delete [] boundaryrAU_internal[patchi];
            delete [] boundaryHbyA_internal[patchi];
        }
    }


    delete [] internalCoeffsPtr;
    delete [] boundaryCoeffsPtr;
    delete [] boundaryU;
    delete [] faceCells;
    delete [] boundaryrAU;
    delete [] boundaryrAU_internal;
    delete [] boundaryHbyA;
    delete [] boundaryHbyA_internal;
    
    Info << "prepreprocess_p free : " << clock.timeIncrement() << endl;
    Info << "prepreprocess_p total : " << clock.elapsedTime() << endl;
}
}