#include "GenFvMatrix.H"
#include "multivariateGaussConvectionScheme.H"
#include "gaussConvectionScheme.H"
#include "snGradScheme.H"
#include "linear.H"
#include "orthogonalSnGrad.H"
namespace Foam{

tmp<fvVectorMatrix>
GenMatrix_U(
    const volScalarField& rho,
    volVectorField& U,
    const surfaceScalarField& phi,
    const volScalarField& p, 
    compressible::turbulenceModel& turbulence
){

    const fvMesh& mesh = U.mesh();
    assert(mesh.moving() == false);

    label nPatch = U.boundaryField().size();
    label *patchSize = new label[nPatch];
    for (label i = 0; i < nPatch; i++) {
        patchSize[i] = U.boundaryField()[i].size();
    }
    
    Info << "nCells : " << nCells << endl;
    Info << "nFaces : " << nFaces << endl;
    Info << "nPatch : " << nPatch << endl;

    // basic variable
    // expansive but only once

    const surfaceScalarField& weights = mesh.surfaceInterpolation::weights(); // interpolation weight (linear)
    const surfaceScalarField& deltaCoeffs = mesh.nonOrthDeltaCoeffs();
    const scalar *weightsPtr = &weights[0];
    const scalar *deltaCoeffsPtr = &deltaCoeffs[0];
    const scalar* const __restrict__ phiPtr = &phi[0];
    const scalar* const __restrict__ UOldTimePtr = &U.oldTime()[0][0];
    const scalar* const __restrict__ meshSfPtr = &mesh.Sf()[0][0];
    const scalar* const __restrict__ meshVPtr = &mesh.V()[0];
    const scalar* const __restrict__ meshMagSfPtr = &mesh.magSf()[0];
    
    const scalar *turbRhoPtr = &turbulence.rho()[0];
    tmp<volScalarField> tNuEff = turbulence.nuEff();
    volScalarField turbNuEff = tNuEff.ref();
    const scalar *turbNuEffPtr = &turbNuEff[0];
    
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

    scalar rDeltaT = 1.0/mesh.time().deltaTValue();

    const scalar* const __restrict__ rhoPtr = &rho.primitiveField()[0];
    const scalar* const __restrict__ rhoOldTimePtr = &rho.oldTime().primitiveField()[0];
    
    scalar* __restrict__ diagPtr = &fvm.diag()[0];
    scalar* __restrict__ sourcePtr = &fvm.source()[0][0];
    scalar* __restrict__ lowerPtr = &fvm.lower()[0];
    scalar* __restrict__ upperPtr = &fvm.upper()[0];

    surfaceScalarField pf(
        IOobject
        (
            "interpolate("+p.name()+')',
            p.instance(),
            p.db()
        ),
        mesh,
        mesh.Sf().dimensions()*p.dimensions()
    );
    surfaceVectorField Uf(
        IOobject
        (
            "interpolate("+U.name()+')',
            U.instance(),
            U.db()
        ),
        mesh,
        mesh.Sf().dimensions()*U.dimensions()
    );


    volScalarField gamma = (turbulence.rho()*turbulence.nuEff()).ref();

    surfaceScalarField gammaf(
        IOobject
        (
            "interpolate("+gamma.name()+')',
            gamma.instance(),
            gamma.db()
        ),
        mesh,
        mesh.Sf().dimensions()*gamma.dimensions()
    );

    surfaceScalarField gammaMagSf = (gammaf * mesh.magSf()).ref();

    const scalar* const __restrict__ pPtr = &p[0];
    const scalar* const __restrict__ gammaPtr = &gamma[0];

    const label *l = &fvm.lduAddr().lowerAddr()[0];
    const label *u = &fvm.lduAddr().upperAddr()[0];

    volVectorField gGrad(
        IOobject
        (
            "gradP",
            pf.instance(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensioned<vector>
        (
            "0",
            pf.dimensions()/dimLength,
            Zero
        ),
        extrapolatedCalculatedFvPatchField<vector>::typeName
    );
    volTensorField gGradU(
        IOobject
        (
            "gradU",
            Uf.instance(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensioned<tensor>
        (
            "0",
            Uf.dimensions()/dimLength,
            Zero
        ),
        extrapolatedCalculatedFvPatchField<tensor>::typeName
    );
    Field<vector>& igGrad = gGrad;
    scalar *gradpPtr = &gGrad[0][0];
    Field<tensor>& igGradU = gGradU;
    scalar *gradUPtr = &gGradU[0][0];

    /* --------------------------------------------------------------------------------------------------------------------------------------------- */
    
    // prepare patch_type

    label allPatchSize = 0;
    for(label patchi = 0 ; patchi < nPatch ; ++patchi){
        const fvPatchVectorField& patchU = U.boundaryField()[patchi];
        if (patchU.type() == "processor" || patchU.type() == "processorCyclic"){
            allPatchSize += 2 * patchSize[patchi];
        } else {
            allPatchSize += patchSize[patchi];
        }
    }

    scalar *boundaryPtr = new scalar[allPatchSize * 3];
    scalar *boundaryDeltaCoeffs = new scalar[allPatchSize];
    scalar *boundaryWeight = new scalar[allPatchSize];
    scalar *boundaryGammaMagSf = new scalar[allPatchSize];
    scalar *boundaryPhi = new scalar[allPatchSize];
    scalar *boundaryGradU = new scalar[allPatchSize * 9]();
    scalar *bouTurbRho = new scalar[allPatchSize];
    scalar *bouTurbNuEff = new scalar[allPatchSize];
    scalar *boundarySf = new scalar[allPatchSize * 3];
    scalar *boundaryMagSf = new scalar[allPatchSize];
    label *boundaryFaceCells = new label[allPatchSize];
    scalar *boundaryP = new scalar[allPatchSize];

    label offset = 0;

    label *patchTypePtr = new label[nPatch];
    for(label patchi = 0 ; patchi < nPatch ; ++patchi)
    {
        std::string patchTypeStr = U.boundaryField()[patchi].type();
        label *patchTypeSelector = &(patchTypePtr[patchi]);
        boundaryConditions patchCondition;
        static std::map<std::string, boundaryConditions> BCMap = {
            {"zeroGradient", zeroGradient},
            {"fixedValue", fixedValue},
            {"empty", empty},
            {"gradientEnergy", gradientEnergy},
            {"calculated", calculated},
            {"coupled", coupled},
            {"cyclic", cyclic},
            {"processor", processor},
            {"extrapolated", extrapolated},
            {"fixedEnergy", fixedEnergy},
            {"processorCyclic", processorCyclic}
        };
        auto iter = BCMap.find(patchTypeStr);
        if (iter != BCMap.end()) {
            patchCondition = iter->second;
        } else {
            throw std::runtime_error("Unknown boundary condition: " + patchTypeStr);
        }
        switch (patchCondition){
            case zeroGradient:
            {
                *patchTypeSelector = 0;
                break;
            }
            case fixedValue:
            {
                *patchTypeSelector = 1;
                break;
            }
            case coupled:
            {
                *patchTypeSelector = 2;
                break;
            }
            case empty:
            {
                *patchTypeSelector = 3;
                break;
            }
            case gradientEnergy:
            {
                *patchTypeSelector = 4;
                break;
            }
            case calculated:
            {
                *patchTypeSelector = 5;
                break;
            }
            case cyclic:
            {
                *patchTypeSelector = 6;
                break;
            }
            case processor:
            {
                *patchTypeSelector = 7;
                break;
            }
            case extrapolated:
            {
                *patchTypeSelector = 8;
                break;
            }
            case fixedEnergy:
            {
                *patchTypeSelector = 9;
                break;
            }
            case processorCyclic:
            {
                *patchTypeSelector = 10;
                break;
            }
        }

        const fvPatchVectorField& patchU = U.boundaryField()[patchi];
        const fvsPatchScalarField& pDeltaCoeffs = deltaCoeffs.boundaryField()[patchi];
        const fvsPatchScalarField& pw = weights.boundaryField()[patchi];
        const fvsPatchScalarField& pGamma = gammaMagSf.boundaryField()[patchi];
        const fvsPatchScalarField& patchFlux = phi.boundaryField()[patchi];
        const fvPatchTensorField& patchGradU = gGradU.boundaryField()[patchi];
        const fvPatchScalarField& pTurbRho = turbulence.rho().boundaryField()[patchi];
        const fvPatchScalarField& pTurbNuEff = turbNuEff.boundaryField()[patchi];
        const fvsPatchVectorField& pSf = mesh.Sf().boundaryField()[patchi];
        const fvsPatchScalarField& pMagSf = mesh.magSf().boundaryField()[patchi];
        const labelUList& pFaceCells = mesh.boundary()[patchi].faceCells();
        const fvPatchScalarField& patchP = p.boundaryField()[patchi];

        label patchsize = patchU.size();
        if (patchU.type() == "processor"
            || patchU.type() == "processorCyclic") {
            memcpy(boundaryPtr + 3*offset, &patchU[0][0], patchsize * 3 * sizeof(scalar));
            vectorField patchUInternal = 
                    dynamic_cast<const processorFvPatchField<vector>&>(patchU).patchInternalField()();
            memcpy(boundaryPtr + 3*offset + 3*patchsize, &patchUInternal[0][0], patchsize * 3 * sizeof(scalar));

            memcpy(boundaryDeltaCoeffs + offset, &pDeltaCoeffs[0], patchsize * sizeof(scalar));
            memcpy(boundaryDeltaCoeffs + offset + patchsize, &pDeltaCoeffs[0], patchsize * sizeof(scalar));

            memcpy(boundaryWeight + offset, &pw[0], patchsize * sizeof(scalar));
            memcpy(boundaryWeight + offset + patchsize, &pw[0], patchsize * sizeof(scalar));

            memcpy(boundaryGammaMagSf + offset, &pGamma[0], patchsize * sizeof(scalar));
            memcpy(boundaryGammaMagSf + offset + patchsize, &pGamma[0], patchsize * sizeof(scalar));

            memcpy(boundaryPhi + offset, &patchFlux[0], patchsize * sizeof(scalar));
            memcpy(boundaryPhi + offset + patchsize, &patchFlux[0], patchsize * sizeof(scalar));

            memcpy(boundaryGradU + 9*offset, &patchGradU[0][0], patchsize * 9 * sizeof(scalar));
            memcpy(boundaryGradU + 9*offset + 9*patchsize, &patchGradU[0][0], patchsize * 9 * sizeof(scalar));

            memcpy(bouTurbRho + offset, &pTurbRho[0], patchsize * sizeof(scalar));
            scalarField patchPTurbRhoInternal = 
                    dynamic_cast<const processorFvPatchField<scalar>&>(pTurbRho).patchInternalField()();
            memcpy(bouTurbRho + offset + patchsize, &patchPTurbRhoInternal[0], patchsize * sizeof(scalar));

            memcpy(bouTurbNuEff + offset, &pTurbNuEff[0], patchsize * sizeof(scalar));
            scalarField patchPTurbNuEffInternal = 
                    dynamic_cast<const processorFvPatchField<scalar>&>(pTurbNuEff).patchInternalField()();
            memcpy(bouTurbNuEff + offset + patchsize, &patchPTurbNuEffInternal[0], patchsize * sizeof(scalar));

            memcpy(boundarySf + 3*offset, &pSf[0][0], patchsize * 3 * sizeof(scalar));
            memcpy(boundarySf + 3*offset + 3*patchsize, &pSf[0][0], patchsize * 3 * sizeof(scalar));

            memcpy(boundaryMagSf + offset, &pMagSf[0], patchsize * sizeof(scalar));
            memcpy(boundaryMagSf + offset + patchsize, &pMagSf[0], patchsize * sizeof(scalar));

            memcpy(boundaryFaceCells + offset, &pFaceCells[0], patchsize * sizeof(label));
            memcpy(boundaryFaceCells + offset + patchsize, &pFaceCells[0], patchsize * sizeof(label));

            memcpy(boundaryP + offset, &patchP[0], patchsize * sizeof(scalar));
            scalarField patchPInternal = 
                    dynamic_cast<const processorFvPatchField<scalar>&>(patchP).patchInternalField()();
            memcpy(boundaryP + offset + patchsize, &patchPInternal[0], patchsize * sizeof(scalar));

            offset += patchsize * 2;

        } else {
            memcpy(boundaryPtr + 3*offset, &patchU[0][0], patchsize * 3 * sizeof(scalar));
            
            memcpy(boundaryDeltaCoeffs + offset, &pDeltaCoeffs[0], patchsize * sizeof(scalar));
            
            memcpy(boundaryWeight + offset, &pw[0], patchsize * sizeof(scalar));

            memcpy(boundaryGammaMagSf + offset, &pGamma[0], patchsize * sizeof(scalar));

            memcpy(boundaryPhi + offset, &patchFlux[0], patchsize * sizeof(scalar));

            memcpy(boundaryGradU + 9*offset, &patchGradU[0][0], patchsize * 9 * sizeof(scalar));

            memcpy(bouTurbRho + offset, &pTurbRho[0], patchsize * sizeof(scalar));

            memcpy(bouTurbNuEff + offset, &pTurbNuEff[0], patchsize * sizeof(scalar));

            memcpy(boundarySf + 3*offset, &pSf[0][0], patchsize * 3 * sizeof(scalar));

            memcpy(boundaryMagSf + offset, &pMagSf[0], patchsize * sizeof(scalar));

            memcpy(boundaryFaceCells + offset, &pFaceCells[0], patchsize * sizeof(label));

            memcpy(boundaryP + offset, &patchP[0], patchsize * sizeof(scalar));

            offset += patchsize;
        }
    }

    scalar *valueInternalCoeffs = new scalar[allPatchSize * 3];
    scalar *valueBoundaryCoeffs = new scalar[allPatchSize * 3];
    scalar *gradientInternalCoeffs = new scalar[allPatchSize * 3];
    scalar *gradientBoundaryCoeffs = new scalar[allPatchSize * 3];

    scalar *internalCoeffsPtr = new scalar[allPatchSize * 3]();
    scalar *boundaryCoeffsPtr = new scalar[allPatchSize * 3]();

    /* --------------------------------------------------------------------------------------------------------------------------------------------- */

    /**
     * update_boundary_coeffs_vector
    */

    offset = 0;
    #pragma omp parallel for
    for(label patchi = 0 ; patchi < nPatch ; ++patchi){

        if(patchSize[patchi] == 0) continue;  

        /* update_boundary_coeffs_zeroGradient_vector */
        if (patchTypePtr[patchi] == boundaryConditions::zeroGradient){
            for(label s = 0 ; s < patchSize[patchi] ; ++s){
                valueInternalCoeffs[(offset + s) * 3] = 1;
                valueInternalCoeffs[1 + (offset + s) * 3] = 1;
                valueInternalCoeffs[2 + (offset + s) * 3] = 1;
                valueBoundaryCoeffs[(offset + s) * 3] = 0;
                valueBoundaryCoeffs[1 + (offset + s) * 3] = 0;
                valueBoundaryCoeffs[2 + (offset + s) * 3] = 0;
                gradientInternalCoeffs[(offset + s) * 3] = 0;
                gradientInternalCoeffs[1 + (offset + s) * 3] = 0;
                gradientInternalCoeffs[2 + (offset + s) * 3] = 0;
                gradientBoundaryCoeffs[(offset + s) * 3] = 0;
                gradientBoundaryCoeffs[1 + (offset + s) * 3] = 0;
                gradientBoundaryCoeffs[2 + (offset + s) * 3] = 0;
            }
        }
        /* update_boundary_coeffs_fixedValue_vector */
        else if (patchTypePtr[patchi] == boundaryConditions::fixedValue){
            for(label s = 0 ; s < patchSize[patchi] ; ++s){
                valueInternalCoeffs[(offset + s) * 3] = 0.;
                valueInternalCoeffs[1 + (offset + s) * 3] = 0.;
                valueInternalCoeffs[2 + (offset + s) * 3] = 0.;
                valueBoundaryCoeffs[(offset + s) * 3] = boundaryPtr[(offset + s) * 3];
                valueBoundaryCoeffs[1 + (offset + s) * 3] = boundaryPtr[1 + (offset + s) * 3];
                valueBoundaryCoeffs[2 + (offset + s) * 3] = boundaryPtr[2 + (offset + s) * 3];
                gradientInternalCoeffs[(offset + s) * 3] = -1 * boundaryDeltaCoeffs[offset + s];
                gradientInternalCoeffs[1 + (offset + s) * 3] = -1 * boundaryDeltaCoeffs[offset + s];
                gradientInternalCoeffs[2 + (offset + s) * 3] = -1 * boundaryDeltaCoeffs[offset + s];
                gradientBoundaryCoeffs[(offset + s) * 3] = boundaryDeltaCoeffs[offset + s] * boundaryPtr[(offset + s) * 3];
                gradientBoundaryCoeffs[1 + (offset + s) * 3] = boundaryDeltaCoeffs[offset + s] * boundaryPtr[1 + (offset + s) * 3];
                gradientBoundaryCoeffs[2 + (offset + s) * 3] = boundaryDeltaCoeffs[offset + s] * boundaryPtr[2 + (offset + s) * 3];
            }
        }
        /* update_boundary_coeffs_processor_vector */
        else if (patchTypePtr[patchi] == boundaryConditions::cyclic){
            for(label s = 0 ; s < patchSize[patchi] ; ++s){
                valueInternalCoeffs[(offset + s) * 3] = boundaryWeight[offset + s];
                valueInternalCoeffs[1 + (offset + s) * 3] = boundaryWeight[offset + s];
                valueInternalCoeffs[2 + (offset + s) * 3] = boundaryWeight[offset + s];
                valueBoundaryCoeffs[(offset + s) * 3] = 1 - boundaryWeight[offset + s];
                valueBoundaryCoeffs[1 + (offset + s) * 3] = 1 - boundaryWeight[offset + s];
                valueBoundaryCoeffs[2 + (offset + s) * 3] = 1 - boundaryWeight[offset + s];
                gradientInternalCoeffs[(offset + s) * 3] = -1 * boundaryDeltaCoeffs[offset + s];
                gradientInternalCoeffs[1 + (offset + s) * 3] = -1 * boundaryDeltaCoeffs[offset + s];
                gradientInternalCoeffs[2 + (offset + s) * 3] = -1 * boundaryDeltaCoeffs[offset + s];
                gradientBoundaryCoeffs[(offset + s) * 3] = boundaryDeltaCoeffs[offset + s];
                gradientBoundaryCoeffs[1 + (offset + s) * 3] = boundaryDeltaCoeffs[offset + s];
                gradientBoundaryCoeffs[2 + (offset + s) * 3] = boundaryDeltaCoeffs[offset + s];
            }
        }
        /* update_boundary_coeffs_processor_vector */
        else if (patchTypePtr[patchi] == boundaryConditions::processor || patchTypePtr[patchi] == boundaryConditions::processorCyclic){
            for(label s = 0 ; s < patchSize[patchi] ; ++s){
                valueInternalCoeffs[(offset + s) * 3] = boundaryWeight[offset + s];
                valueInternalCoeffs[1 + (offset + s) * 3] = boundaryWeight[offset + s];
                valueInternalCoeffs[2 + (offset + s) * 3] = boundaryWeight[offset + s];
                valueBoundaryCoeffs[(offset + s) * 3] = 1 - boundaryWeight[offset + s];
                valueBoundaryCoeffs[1 + (offset + s) * 3] = 1 - boundaryWeight[offset + s];
                valueBoundaryCoeffs[2 + (offset + s) * 3] = 1 - boundaryWeight[offset + s];
                gradientInternalCoeffs[(offset + s) * 3] = -1 * boundaryDeltaCoeffs[offset + s];
                gradientInternalCoeffs[1 + (offset + s) * 3] = -1 * boundaryDeltaCoeffs[offset + s];
                gradientInternalCoeffs[2 + (offset + s) * 3] = -1 * boundaryDeltaCoeffs[offset + s];
                gradientBoundaryCoeffs[(offset + s) * 3] = boundaryDeltaCoeffs[offset + s];
                gradientBoundaryCoeffs[1 + (offset + s) * 3] = boundaryDeltaCoeffs[offset + s];
                gradientBoundaryCoeffs[2 + (offset + s) * 3] = boundaryDeltaCoeffs[offset + s];
            }
            offset += 2 * patchSize[patchi];
            continue;
        }
        else{
            fprintf(stderr, "%s %d, boundaryConditions other than zeroGradient are not support yet!\n", __FILE__, __LINE__);
        }
        offset += patchSize[patchi];
    }

    /* --------------------------------------------------------------------------------------------------------------------------------------------- */

    /**
     * fvm_ddt_vector
    */
    
    #pragma omp parallel for
    for(label c = 0; c < nCells; ++c){
        diagPtr[c] += rDeltaT * rhoPtr[c] * meshVPtr[c];
        sourcePtr[c * 3] += rDeltaT * rhoOldTimePtr[c] * UOldTimePtr[c * 3] * meshVPtr[c];
        sourcePtr[1 + c * 3] += rDeltaT * rhoOldTimePtr[c] * UOldTimePtr[1 + c * 3] * meshVPtr[c];
        sourcePtr[2 + c * 3] += rDeltaT * rhoOldTimePtr[c] * UOldTimePtr[2 + c * 3] * meshVPtr[c];
    }

    /* --------------------------------------------------------------------------------------------------------------------------------------------- */

    /**
     * fvm_div_vector
    */

    /* fvm_div_vector_internal */
    #pragma omp parallel for
    for(label f = 0; f < nFaces; ++f){
        scalar lower_value = - weightsPtr[f] * phiPtr[f];
        scalar upper_value = (- weightsPtr[f] + 1.) * phiPtr[f];        
        lowerPtr[f] += lower_value;
        upperPtr[f] += upper_value;
        diagPtr[l[f]] -= lower_value;
        diagPtr[u[f]] -= upper_value;
    }

    /* fvm_div_vector_boundary */
    offset = 0;
    #pragma omp parallel for
    for(label patchi= 0; patchi < nPatch ; ++patchi){

        if (patchSize[patchi] == 0) continue;
        for(label s = 0; s < patchSize[patchi] ; ++s){
            internalCoeffsPtr[(offset + s) * 3] += boundaryPhi[offset + s] * valueInternalCoeffs[(offset + s) * 3];
            internalCoeffsPtr[1 + (offset + s) * 3] += boundaryPhi[offset + s] * valueInternalCoeffs[1 + (offset + s) * 3];
            internalCoeffsPtr[2 + (offset + s) * 3] += boundaryPhi[offset + s] * valueInternalCoeffs[2 + (offset + s) * 3];
            boundaryCoeffsPtr[(offset + s) * 3] = -boundaryPhi[offset + s] * valueBoundaryCoeffs[(offset + s) * 3];
            boundaryCoeffsPtr[1 + (offset + s) * 3] = -boundaryPhi[offset + s] * valueBoundaryCoeffs[1 + (offset + s) * 3];
            boundaryCoeffsPtr[2 + (offset + s) * 3] = -boundaryPhi[offset + s] * valueBoundaryCoeffs[2 + (offset + s) * 3];
        }
        if (patchTypePtr[patchi] == boundaryConditions::processor || patchTypePtr[patchi] == boundaryConditions::processorCyclic){
            offset += 2 * patchSize[patchi];
        }
        else {
            offset += patchSize[patchi];
        }
    }

    /* --------------------------------------------------------------------------------------------------------------------------------------------- */

    /**
     * fvm_laplacian_vector 
    */

    /* fvm_laplacian_internal */
    #pragma omp parallel for
    for(label f = 0; f < nFaces; ++f){

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
    
    /*  fvm_laplacian_vector_boundary */
    offset = 0;
    #pragma omp parallel for
    for(label patchi= 0; patchi < nPatch ; ++patchi){
        if(patchSize[patchi] == 0) continue;
        for(label s = 0; s < patchSize[patchi] ; ++s){
            internalCoeffsPtr[(offset + s) * 3] += (-1) * bouTurbRho[offset + s] * bouTurbNuEff[offset + s] * boundaryMagSf[offset + s] * gradientInternalCoeffs[(offset + s) * 3];
            internalCoeffsPtr[1 + (offset + s) * 3] += (-1) * bouTurbRho[offset + s] * bouTurbNuEff[offset + s] * boundaryMagSf[offset + s] * gradientInternalCoeffs[1 + (offset + s) * 3];
            internalCoeffsPtr[2 + (offset + s) * 3] += (-1) * bouTurbRho[offset + s] * bouTurbNuEff[offset + s] * boundaryMagSf[offset + s] * gradientInternalCoeffs[2 + (offset + s) * 3];
            boundaryCoeffsPtr[(offset + s) * 3] -= (-1) * bouTurbRho[offset + s] * bouTurbNuEff[offset + s] * boundaryMagSf[offset + s] * gradientBoundaryCoeffs[(offset + s) * 3];
            boundaryCoeffsPtr[1 + (offset + s) * 3] -= (-1) * bouTurbRho[offset + s] * bouTurbNuEff[offset + s] * boundaryMagSf[offset + s] * gradientBoundaryCoeffs[1 + (offset + s) * 3];
            boundaryCoeffsPtr[2 + (offset + s) * 3] -= (-1) * bouTurbRho[offset + s] * bouTurbNuEff[offset + s] * boundaryMagSf[offset + s] * gradientBoundaryCoeffs[2 + (offset + s) * 3];
            
        }
        if (patchTypePtr[patchi] == boundaryConditions::processor || patchTypePtr[patchi] == boundaryConditions::processorCyclic){
            offset += 2 * patchSize[patchi];
        }
        else {
            offset += patchSize[patchi];
        }
    }

    /* --------------------------------------------------------------------------------------------------------------------------------------------- */

    /**
     *  fvc_grad_vector
    */

    /* fvc_grad_vector_internal */
    #pragma omp parallel for
    for(label f = 0; f < nFaces; ++f){

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


    offset = 0;
    #pragma omp parallel for
    for(label patchi = 0; patchi < nPatch ; ++patchi){

        if (patchSize[patchi] == 0) continue;
        if(patchTypePtr[patchi] == boundaryConditions::zeroGradient 
                || patchTypePtr[patchi] == boundaryConditions::fixedValue 
                || patchTypePtr[patchi] == boundaryConditions::calculated 
                || patchTypePtr[patchi] == boundaryConditions::cyclic){
            /* fvc_grad_vector_boundary_zeroGradient */
            for(label s = 0; s < patchSize[patchi] ; ++s){
                scalar bouSfx = boundarySf[3 * (offset + s)];
                scalar bouSfy = boundarySf[3 * (offset + s) + 1];
                scalar bouSfz = boundarySf[3 * (offset + s) + 2];

                scalar boussfx = boundaryPtr[3 * (offset + s)];
                scalar boussfy = boundaryPtr[3 * (offset + s) + 1];
                scalar boussfz = boundaryPtr[3 * (offset + s) + 2];

                scalar grad_xx = bouSfx * boussfx;
                scalar grad_xy = bouSfx * boussfy;
                scalar grad_xz = bouSfx * boussfz;
                scalar grad_yx = bouSfy * boussfx;
                scalar grad_yy = bouSfy * boussfy;
                scalar grad_yz = bouSfy * boussfz;
                scalar grad_zx = bouSfz * boussfx;
                scalar grad_zy = bouSfz * boussfy;
                scalar grad_zz = bouSfz * boussfz;

                gradUPtr[boundaryFaceCells[offset + s] * 9] += grad_xx;
                gradUPtr[boundaryFaceCells[offset + s] * 9 + 1] += grad_xy;
                gradUPtr[boundaryFaceCells[offset + s] * 9 + 2] += grad_xz;
                gradUPtr[boundaryFaceCells[offset + s] * 9 + 3] += grad_yx;
                gradUPtr[boundaryFaceCells[offset + s] * 9 + 4] += grad_yy;
                gradUPtr[boundaryFaceCells[offset + s] * 9 + 5] += grad_yz;
                gradUPtr[boundaryFaceCells[offset + s] * 9 + 6] += grad_zx;
                gradUPtr[boundaryFaceCells[offset + s] * 9 + 7] += grad_zy;
                gradUPtr[boundaryFaceCells[offset + s] * 9 + 8] += grad_zz;
            }

        }else if(patchTypePtr[patchi] == boundaryConditions::processor || patchTypePtr[patchi] == boundaryConditions::processorCyclic){
            /* fvc_grad_vector_boundary_processor */
            for(label s = 0; s < patchSize[patchi] ; ++s){

                label neighbor_start_index = offset + s;
                label internal_start_index = offset + patchSize[patchi] + s;

                scalar bouWeight = boundaryWeight[neighbor_start_index];

                scalar bouSfx = boundarySf[3 * neighbor_start_index];
                scalar bouSfy = boundarySf[3 * neighbor_start_index + 1];
                scalar bouSfz = boundarySf[3 * neighbor_start_index + 2];

                scalar boussfx = (1 - bouWeight) * boundaryPtr[3 * neighbor_start_index] + bouWeight * boundaryPtr[3 * internal_start_index];
                scalar boussfy = (1 - bouWeight) * boundaryPtr[3 * neighbor_start_index + 1] + bouWeight * boundaryPtr[3 * internal_start_index + 1];
                scalar boussfz = (1 - bouWeight) * boundaryPtr[3 * neighbor_start_index + 2] + bouWeight * boundaryPtr[3 * internal_start_index + 2];

                scalar grad_xx = bouSfx * boussfx;
                scalar grad_xy = bouSfx * boussfy;
                scalar grad_xz = bouSfx * boussfz;
                scalar grad_yx = bouSfy * boussfx;
                scalar grad_yy = bouSfy * boussfy;
                scalar grad_yz = bouSfy * boussfz;
                scalar grad_zx = bouSfz * boussfx;
                scalar grad_zy = bouSfz * boussfy;
                scalar grad_zz = bouSfz * boussfz;

                gradUPtr[boundaryFaceCells[offset + s] * 9] += grad_xx;
                gradUPtr[boundaryFaceCells[offset + s] * 9 + 1] += grad_xy;
                gradUPtr[boundaryFaceCells[offset + s] * 9 + 2] += grad_xz;
                gradUPtr[boundaryFaceCells[offset + s] * 9 + 3] += grad_yx;
                gradUPtr[boundaryFaceCells[offset + s] * 9 + 4] += grad_yy;
                gradUPtr[boundaryFaceCells[offset + s] * 9 + 5] += grad_yz;
                gradUPtr[boundaryFaceCells[offset + s] * 9 + 6] += grad_zx;
                gradUPtr[boundaryFaceCells[offset + s] * 9 + 7] += grad_zy;
                gradUPtr[boundaryFaceCells[offset + s] * 9 + 8] += grad_zz;

            }

            offset += 2 * patchSize[patchi];
            continue;
        }else {
            fprintf(stderr, "%s %d, boundaryConditions other than zeroGradient are not support yet!\n", __FILE__, __LINE__);
        }
        offset += patchSize[patchi];
    }

    /* divide_cell_volume_tsr */
    #pragma omp parallel for
    for(label c = 0; c < nCells; ++c){
        gradUPtr[9 * c] = gradUPtr[9 * c] / meshVPtr[c];
        gradUPtr[9 * c + 1] = gradUPtr[9 * c + 1] / meshVPtr[c];
        gradUPtr[9 * c + 2] = gradUPtr[9 * c + 2] / meshVPtr[c];
        gradUPtr[9 * c + 3] = gradUPtr[9 * c + 3] / meshVPtr[c];
        gradUPtr[9 * c + 4] = gradUPtr[9 * c + 4] / meshVPtr[c];
        gradUPtr[9 * c + 5] = gradUPtr[9 * c + 5] / meshVPtr[c];
        gradUPtr[9 * c + 6] = gradUPtr[9 * c + 6] / meshVPtr[c];
        gradUPtr[9 * c + 7] = gradUPtr[9 * c + 7] / meshVPtr[c];
        gradUPtr[9 * c + 8] = gradUPtr[9 * c + 8] / meshVPtr[c];
    }

    offset = 0;
    #pragma omp parallel for
    for(label patchi = 0; patchi < nPatch; ++patchi){

        if(patchSize[patchi] == 0) continue;
        if(patchTypePtr[patchi] == boundaryConditions::zeroGradient){
            /* fvc_grad_vector_correctBC_zeroGradient */
            for(label s = 0; s < patchSize[patchi] ; ++s){
                scalar grad_xx = gradUPtr[boundaryFaceCells[offset + s] * 9];
                scalar grad_xy = gradUPtr[boundaryFaceCells[offset + s] * 9 + 1];
                scalar grad_xz = gradUPtr[boundaryFaceCells[offset + s] * 9 + 2];
                scalar grad_yx = gradUPtr[boundaryFaceCells[offset + s] * 9 + 3];
                scalar grad_yy = gradUPtr[boundaryFaceCells[offset + s] * 9 + 4];
                scalar grad_yz = gradUPtr[boundaryFaceCells[offset + s] * 9 + 5];
                scalar grad_zx = gradUPtr[boundaryFaceCells[offset + s] * 9 + 6];
                scalar grad_zy = gradUPtr[boundaryFaceCells[offset + s] * 9 + 7];
                scalar grad_zz = gradUPtr[boundaryFaceCells[offset + s] * 9 + 8];

                scalar n_x = boundarySf[(offset + s) * 3] / boundaryMagSf[offset + s];
                scalar n_y = boundarySf[(offset + s) * 3 + 1] / boundaryMagSf[offset + s];
                scalar n_z = boundarySf[(offset + s) * 3 + 2] / boundaryMagSf[offset + s];

                scalar grad_correction_x = - (n_x * grad_xx + n_y * grad_yx + n_z * grad_zx);
                scalar grad_correction_y = - (n_x * grad_xy + n_y * grad_yy + n_z * grad_zy);
                scalar grad_correction_z = - (n_x * grad_xz + n_y * grad_yz + n_z * grad_zz);

                boundaryGradU[(offset + s) * 9] = grad_xx + n_x * grad_correction_x;
                boundaryGradU[(offset + s) * 9 + 1] = grad_xy + n_x * grad_correction_y;
                boundaryGradU[(offset + s) * 9 + 2] = grad_xz + n_x * grad_correction_z;
                boundaryGradU[(offset + s) * 9 + 3] = grad_yx + n_y * grad_correction_x;
                boundaryGradU[(offset + s) * 9 + 4] = grad_yy + n_y * grad_correction_y;
                boundaryGradU[(offset + s) * 9 + 5] = grad_yz + n_y * grad_correction_z;
                boundaryGradU[(offset + s) * 9 + 6] = grad_zx + n_z * grad_correction_x;
                boundaryGradU[(offset + s) * 9 + 7] = grad_zy + n_z * grad_correction_y;
                boundaryGradU[(offset + s) * 9 + 8] = grad_zz + n_z * grad_correction_z;
                
            }
            

        }else if(patchTypePtr[patchi] == boundaryConditions::fixedValue){
            /* fvc_grad_vector_correctBC_fixedValue */
            for(label s = 0; s < patchSize[patchi] ; ++s){

                scalar grad_xx = gradUPtr[boundaryFaceCells[offset + s] * 9];
                scalar grad_xy = gradUPtr[boundaryFaceCells[offset + s] * 9 + 1];
                scalar grad_xz = gradUPtr[boundaryFaceCells[offset + s] * 9 + 2];
                scalar grad_yx = gradUPtr[boundaryFaceCells[offset + s] * 9 + 3];
                scalar grad_yy = gradUPtr[boundaryFaceCells[offset + s] * 9 + 4];
                scalar grad_yz = gradUPtr[boundaryFaceCells[offset + s] * 9 + 5];
                scalar grad_zx = gradUPtr[boundaryFaceCells[offset + s] * 9 + 6];
                scalar grad_zy = gradUPtr[boundaryFaceCells[offset + s] * 9 + 7];
                scalar grad_zz = gradUPtr[boundaryFaceCells[offset + s] * 9 + 8];

                scalar n_x = boundarySf[(offset + s) * 3] / boundaryMagSf[offset + s];
                scalar n_y = boundarySf[(offset + s) * 3 + 1] / boundaryMagSf[offset + s];
                scalar n_z = boundarySf[(offset + s) * 3 + 2] / boundaryMagSf[offset + s];

                scalar sn_grad_x = boundaryDeltaCoeffs[offset + s] * (boundaryPtr[(offset + s) * 3] - UOldTimePtr[boundaryFaceCells[offset + s] * 3]);
                scalar sn_grad_y = boundaryDeltaCoeffs[offset + s] * (boundaryPtr[(offset + s) * 3 + 1] - UOldTimePtr[boundaryFaceCells[offset + s] * 3 + 1]);
                scalar sn_grad_z = boundaryDeltaCoeffs[offset + s] * (boundaryPtr[(offset + s) * 3 + 2] - UOldTimePtr[boundaryFaceCells[offset + s] * 3 + 2]);

                scalar grad_correction_x = sn_grad_x - (n_x * grad_xx + n_y * grad_yx + n_z * grad_zx);
                scalar grad_correction_y = sn_grad_y - (n_x * grad_xy + n_y * grad_yy + n_z * grad_zy);
                scalar grad_correction_z = sn_grad_z - (n_x * grad_xz + n_y * grad_yz + n_z * grad_zz);

                boundaryGradU[(offset + s) * 9] = grad_xx + n_x * grad_correction_x;
                boundaryGradU[(offset + s) * 9 + 1] = grad_xy + n_x * grad_correction_y;
                boundaryGradU[(offset + s) * 9 + 2] = grad_xz + n_x * grad_correction_z;
                boundaryGradU[(offset + s) * 9 + 3] = grad_yx + n_y * grad_correction_x;
                boundaryGradU[(offset + s) * 9 + 4] = grad_yy + n_y * grad_correction_y;
                boundaryGradU[(offset + s) * 9 + 5] = grad_yz + n_y * grad_correction_z;
                boundaryGradU[(offset + s) * 9 + 6] = grad_zx + n_z * grad_correction_x;
                boundaryGradU[(offset + s) * 9 + 7] = grad_zy + n_z * grad_correction_y;
                boundaryGradU[(offset + s) * 9 + 8] = grad_zz + n_z * grad_correction_z;

            }

        }else if(patchTypePtr[patchi] == boundaryConditions::processor || patchTypePtr[patchi] == boundaryConditions::processorCyclic){
            /* fvc_grad_vector_correctBC_processor */
            for(label s = 0; s < patchSize[patchi] ; ++s){
                label neighbor_start_index = offset + s;
                label internal_start_index = offset + patchSize[patchi] + s;

                boundaryGradU[internal_start_index * 9] = gradUPtr[boundaryFaceCells[neighbor_start_index] * 9];
                boundaryGradU[internal_start_index * 9 + 1] = gradUPtr[boundaryFaceCells[neighbor_start_index] * 9 + 1];
                boundaryGradU[internal_start_index * 9 + 2] = gradUPtr[boundaryFaceCells[neighbor_start_index] * 9 + 2];
                boundaryGradU[internal_start_index * 9 + 3] = gradUPtr[boundaryFaceCells[neighbor_start_index] * 9 + 3];
                boundaryGradU[internal_start_index * 9 + 4] = gradUPtr[boundaryFaceCells[neighbor_start_index] * 9 + 4];
                boundaryGradU[internal_start_index * 9 + 5] = gradUPtr[boundaryFaceCells[neighbor_start_index] * 9 + 5];
                boundaryGradU[internal_start_index * 9 + 6] = gradUPtr[boundaryFaceCells[neighbor_start_index] * 9 + 6];
                boundaryGradU[internal_start_index * 9 + 7] = gradUPtr[boundaryFaceCells[neighbor_start_index] * 9 + 7];
                boundaryGradU[internal_start_index * 9 + 8] = gradUPtr[boundaryFaceCells[neighbor_start_index] * 9 + 8];
                            
                MPI_Send(&boundaryGradU[internal_start_index * 9], 9, MPI_DOUBLE, neighbProcNo[patchi], 0, MPI_COMM_WORLD);
                MPI_Recv(&boundaryGradU[neighbor_start_index * 9], 9, MPI_DOUBLE, neighbProcNo[patchi], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            
            offset += 2 * patchSize[patchi];
            continue;
        }
        else{
            fprintf(stderr, "%s %d, boundaryConditions other than zeroGradient are not support yet!\n", __FILE__, __LINE__);
        }
        offset += patchSize[patchi];
    }

    /* --------------------------------------------------------------------------------------------------------------------------------------------- */

    /**
     *  **if use turbulence model**
     *  getTurbulenceKEpsilon_Smagorinsky
    */

   /* --------------------------------------------------------------------------------------------------------------------------------------------- */

    /**
     *  scale_dec2T_tensor
    */
    
    /* scale_dev2t_tensor_kernel(internal) */
    #pragma omp parallel for
    for (label c = 0; c < nCells; ++c)
    {
        scalar scale = gammaPtr[c]; // scalar scale = turbRhoPtr[c] * turbNuEffPtr[c]; 
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
    offset = 0;
    #pragma omp parallel for
    for (label pi = 0; pi < allPatchSize; ++pi){
        scalar scale = bouTurbRho[pi] * bouTurbNuEff[pi];
        scalar val_xx = boundaryGradU[9 * pi];
        scalar val_xy = boundaryGradU[9 * pi + 1];
        scalar val_xz = boundaryGradU[9 * pi + 2];
        scalar val_yx = boundaryGradU[9 * pi + 3];
        scalar val_yy = boundaryGradU[9 * pi + 4];
        scalar val_yz = boundaryGradU[9 * pi + 5];
        scalar val_zx = boundaryGradU[9 * pi + 6];
        scalar val_zy = boundaryGradU[9 * pi + 7];
        scalar val_zz = boundaryGradU[9 * pi + 8];
        scalar trace_coeff = (2. / 3.) * (val_xx + val_yy + val_zz);
        boundaryGradU[9 * pi] = scale * (val_xx - trace_coeff);
        boundaryGradU[9 * pi + 1] = scale * (val_yx);
        boundaryGradU[9 * pi + 2] = scale * (val_zx);
        boundaryGradU[9 * pi + 3] = scale * (val_xy);
        boundaryGradU[9 * pi + 4] = scale * (val_yy - trace_coeff);
        boundaryGradU[9 * pi + 5] = scale * (val_zy);
        boundaryGradU[9 * pi + 6] = scale * (val_xz);
        boundaryGradU[9 * pi + 7] = scale * (val_yz);
        boundaryGradU[9 * pi + 8] = scale * (val_zz - trace_coeff);
    }    

    /* --------------------------------------------------------------------------------------------------------------------------------------------- */
    
    /**
     *  fvc_div_cell_tensor
    */

    /* fvc_div_cell_tensor_internal */
    #pragma omp parallel for
    for (label f = 0; f < nFaces ; ++f){

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

    offset = 0;
    #pragma omp parallel for
    for(label patchi = 0; patchi < nPatch; ++patchi){
        if(patchSize[patchi] == 0) continue;
        if(patchTypePtr[patchi] == boundaryConditions::zeroGradient 
                || patchTypePtr[patchi] == boundaryConditions::fixedValue 
                || patchTypePtr[patchi] == boundaryConditions::calculated 
                || patchTypePtr[patchi] == boundaryConditions::cyclic){
            /* fvc_div_cell_tensor_boundary_zeroGradient */
            for(label s = 0; s < patchSize[patchi] ; ++s){

                scalar bouSfx = boundarySf[(offset + s) * 3];
                scalar bouSfy = boundarySf[(offset + s) * 3 + 1];
                scalar bouSfz = boundarySf[(offset + s) * 3 + 2];

                scalar boussf_xx = boundaryGradU[(offset + s) * 9];
                scalar boussf_xy = boundaryGradU[(offset + s) * 9 + 1];
                scalar boussf_xz = boundaryGradU[(offset + s) * 9 + 2];
                scalar boussf_yx = boundaryGradU[(offset + s) * 9 + 3];
                scalar boussf_yy = boundaryGradU[(offset + s) * 9 + 4];
                scalar boussf_yz = boundaryGradU[(offset + s) * 9 + 5];
                scalar boussf_zx = boundaryGradU[(offset + s) * 9 + 6];
                scalar boussf_zy = boundaryGradU[(offset + s) * 9 + 7];
                scalar boussf_zz = boundaryGradU[(offset + s) * 9 + 8];

                scalar bouDiv_x = (bouSfx * boussf_xx + bouSfy * boussf_yx + bouSfz * boussf_zx);
                scalar bouDiv_y = (bouSfx * boussf_xy + bouSfy * boussf_yy + bouSfz * boussf_zy);
                scalar bouDiv_z = (bouSfx * boussf_xz + bouSfy * boussf_yz + bouSfz * boussf_zz);

                sourcePtr[boundaryFaceCells[offset + s] * 3] += bouDiv_x;
                sourcePtr[boundaryFaceCells[offset + s] * 3 + 1] += bouDiv_y;
                sourcePtr[boundaryFaceCells[offset + s] * 3 + 2] += bouDiv_z;

            }
        }else if(patchTypePtr[patchi] == boundaryConditions::processor || patchTypePtr[patchi] == boundaryConditions::processorCyclic){
            /* fvc_div_cell_tensor_boundary_processor */
            for (label s = 0; s < patchSize[patchi]; ++s){
                label neighbor_start_index = offset + s;
                label internal_start_index = offset + patchSize[patchi] + s;

                scalar bouWeight = boundaryWeight[neighbor_start_index];

                scalar bouSfx = boundarySf[3 * neighbor_start_index];
                scalar bouSfy = boundarySf[3 * neighbor_start_index + 1];
                scalar bouSfz = boundarySf[3 * neighbor_start_index + 2];

                scalar boussf_xx = (1 - bouWeight) * boundaryGradU[neighbor_start_index * 9] + bouWeight * boundaryGradU[internal_start_index * 9];
                scalar boussf_xy = (1 - bouWeight) * boundaryGradU[neighbor_start_index * 9 + 1] + bouWeight * boundaryGradU[internal_start_index * 9 + 1];
                scalar boussf_xz = (1 - bouWeight) * boundaryGradU[neighbor_start_index * 9 + 2] + bouWeight * boundaryGradU[internal_start_index * 9 + 2];
                scalar boussf_yx = (1 - bouWeight) * boundaryGradU[neighbor_start_index * 9 + 3] + bouWeight * boundaryGradU[internal_start_index * 9 + 3];
                scalar boussf_yy = (1 - bouWeight) * boundaryGradU[neighbor_start_index * 9 + 4] + bouWeight * boundaryGradU[internal_start_index * 9 + 4];
                scalar boussf_yz = (1 - bouWeight) * boundaryGradU[neighbor_start_index * 9 + 5] + bouWeight * boundaryGradU[internal_start_index * 9 + 5];
                scalar boussf_zx = (1 - bouWeight) * boundaryGradU[neighbor_start_index * 9 + 6] + bouWeight * boundaryGradU[internal_start_index * 9 + 6];
                scalar boussf_zy = (1 - bouWeight) * boundaryGradU[neighbor_start_index * 9 + 7] + bouWeight * boundaryGradU[internal_start_index * 9 + 7];
                scalar boussf_zz = (1 - bouWeight) * boundaryGradU[neighbor_start_index * 9 + 8] + bouWeight * boundaryGradU[internal_start_index * 9 + 8];

                scalar bouDiv_x = (bouSfx * boussf_xx + bouSfy * boussf_yx + bouSfz * boussf_zx);
                scalar bouDiv_y = (bouSfx * boussf_xy + bouSfy * boussf_yy + bouSfz * boussf_zy);
                scalar bouDiv_z = (bouSfx * boussf_xz + bouSfy * boussf_yz + bouSfz * boussf_zz);

                label cellIndex = boundaryFaceCells[neighbor_start_index];

                sourcePtr[cellIndex * 3] += bouDiv_x;
                sourcePtr[cellIndex * 3 + 1] += bouDiv_y;
                sourcePtr[cellIndex * 3 + 2] += bouDiv_z;
            }
            offset += 2 * patchSize[patchi];
            continue;
        }else{
            fprintf(stderr, "%s %d, boundaryConditions other than zeroGradient are not support yet!\n", __FILE__, __LINE__);
        }
        offset += patchSize[patchi];
    }

    /* --------------------------------------------------------------------------------------------------------------------------------------------- */

    /**
     *  fvc_grad_cell_scalar
    */

    /* fvc_grad_scalar_internal */
    #pragma omp parallel for
    for (label f = 0; f < nFaces; ++f){

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
    offset = 0;
    #pragma omp parallel for
    for(label patchi = 0; patchi < nPatch; ++patchi){
        if(patchSize[patchi] == 0) continue;
        if(patchTypePtr[patchi] == boundaryConditions::zeroGradient 
                || patchTypePtr[patchi] == boundaryConditions::fixedValue 
                || patchTypePtr[patchi] == boundaryConditions::calculated 
                || patchTypePtr[patchi] == boundaryConditions::cyclic){
            /* fvc_grad_scalar_boundary_zeroGradient */
            for (label s = 0; s < patchSize[patchi]; ++s){

                scalar bouvf = boundaryP[offset + s];

                scalar bouSfx = boundarySf[3 * (offset + s)];
                scalar bouSfy = boundarySf[3 * (offset + s) + 1];
                scalar bouSfz = boundarySf[3 * (offset + s) + 2];

                scalar grad_x = bouSfx * bouvf;
                scalar grad_y = bouSfy * bouvf;
                scalar grad_z = bouSfz * bouvf;

                sourcePtr[boundaryFaceCells[offset + s] * 3] -= grad_x;
                sourcePtr[boundaryFaceCells[offset + s] * 3 + 1] -= grad_y;
                sourcePtr[boundaryFaceCells[offset + s] * 3 + 2] -= grad_z;
            }
        }else if(patchTypePtr[patchi] == boundaryConditions::processor || patchTypePtr[patchi] == boundaryConditions::processorCyclic){
            /* fvc_grad_scalar_boundary_processor */
            for (label s = 0; s < patchSize[patchi]; ++s){

                label neighbor_start_index = offset + s;
                label internal_start_index = offset + patchSize[patchi] + s;

                scalar bouWeight = boundaryWeight[neighbor_start_index];

                scalar bouSfx = boundarySf[3 * neighbor_start_index];
                scalar bouSfy = boundarySf[3 * neighbor_start_index + 1];
                scalar bouSfz = boundarySf[3 * neighbor_start_index + 2];

                scalar bouvf = (1 - bouWeight) * boundaryP[neighbor_start_index] + bouWeight * boundaryP[internal_start_index];

                scalar grad_x = bouSfx * bouvf;
                scalar grad_y = bouSfy * bouvf;
                scalar grad_z = bouSfz * bouvf;

                sourcePtr[boundaryFaceCells[offset + s] * 3] -= grad_x;
                sourcePtr[boundaryFaceCells[offset + s] * 3 + 1] -= grad_y;
                sourcePtr[boundaryFaceCells[offset + s] * 3 + 2] -= grad_z;

            }
            offset += 2 * patchSize[patchi];
            continue;
        }else{
            fprintf(stderr, "%s %d, boundaryConditions other than zeroGradient are not support yet!\n", __FILE__, __LINE__);
        }
        offset += patchSize[patchi];
    }


    /* --------------------------------------------------------------------------------------------------------------------------------------------- */
    
    offset = 0;
    for (label patchi = 0; patchi < nPatch; ++patchi){
        for (label s = 0; s < patchSize[patchi]; ++s){
            label start_index = offset + s;
            fvm.internalCoeffs()[patchi][s][0] = internalCoeffsPtr[start_index * 3];
            fvm.internalCoeffs()[patchi][s][1] = internalCoeffsPtr[start_index * 3 + 1];
            fvm.internalCoeffs()[patchi][s][2] = internalCoeffsPtr[start_index * 3 + 2];
            fvm.boundaryCoeffs()[patchi][s][0] = boundaryCoeffsPtr[start_index * 3];
            fvm.boundaryCoeffs()[patchi][s][1] = boundaryCoeffsPtr[start_index * 3 + 1];
            fvm.boundaryCoeffs()[patchi][s][2] = boundaryCoeffsPtr[start_index * 3 + 2];
        }
        if(patchTypePtr[patchi] == boundaryConditions::processor || patchTypePtr[patchi] == boundaryConditions::processorCyclic){
            offset += 2 * patchSize[patchi];
        }else {
            offset += patchSize[patchi];
        }
    }

    return tfvm;
}

}