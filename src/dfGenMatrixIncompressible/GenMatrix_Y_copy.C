#include "GenFvMatrix.H"
#include "multivariateGaussConvectionScheme.H"
#include "gaussConvectionScheme.H"
#include "snGradScheme.H"
#include "linear.H"
#include "orthogonalSnGrad.H"

namespace Foam{

tmp<fvScalarMatrix>
GenMatrix_Y(
    const volScalarField& rho,
    volScalarField& Yi,
    const surfaceScalarField& phi,
    const surfaceScalarField& phiUc,
    const volScalarField& rhoD,
    const volScalarField& mut,
    const Switch splitting,
    const scalar Sct,
    CombustionModel<basicThermo>& combustion,
    fv::convectionScheme<scalar>& mvConvection
){
    assert(splitting == false);

    const fvMesh& mesh = Yi.mesh();
    assert(mesh.moving() == false);

    label nCells = mesh.nCells();
    label nFaces = mesh.neighbour().size();

    // div
    auto& mgcs = dynamic_cast<fv::multivariateGaussConvectionScheme<scalar>&>(mvConvection);
    tmp<surfaceInterpolationScheme<scalar>> tinterpScheme_ = mgcs.interpolationScheme()()(Yi);
    tmp<surfaceScalarField> tweights = tinterpScheme_().weights(Yi);
    const surfaceScalarField& weights = tweights();

    // laplacian
    tmp<surfaceInterpolationScheme<scalar>> tinterpGammaScheme_(new linear<scalar>(mesh));
    tmp<fv::snGradScheme<scalar>> tsnGradScheme_(new fv::orthogonalSnGrad<scalar>(mesh));
    tmp<volScalarField> gammaScalarVol = rhoD + mut/Sct;
    Info << "gammaScalarVol size : " << gammaScalarVol().size() << endl;

    tmp<surfaceScalarField> tgamma = tinterpGammaScheme_().interpolate(gammaScalarVol);
    const surfaceScalarField& gamma = tgamma.ref();
    const surfaceScalarField& deltaCoeffs = tsnGradScheme_().deltaCoeffs(Yi)();

    Info << "gamma size : " << gamma.size() << endl;
    Info << "mesh.magSf() size : " << mesh.magSf().size() << endl;

    surfaceScalarField gammaMagSf
    (
        gamma * mesh.magSf()
    );

    // ddt div
    // -------------------------------------------------------

    tmp<fvScalarMatrix> tfvm_DDT
    (
        new fvScalarMatrix
        (
            Yi,
            rho.dimensions()*Yi.dimensions()*dimVol/dimTime
        )
    );
    fvScalarMatrix& fvm_DDT = tfvm_DDT.ref();

    scalar rDeltaT = 1.0/mesh.time().deltaTValue();

    // fvm_DDT.diag() = rDeltaT*rho.primitiveField()*mesh.Vsc();
    // fvm_DDT.source() = rDeltaT
    //     *rho.oldTime().primitiveField()
    //     *Yi.oldTime().primitiveField()*mesh.Vsc();

    scalar* __restrict__ diagPtr_ddt = fvm_DDT.diag().begin();
    scalar* __restrict__ sourcePtr_ddt = fvm_DDT.source().begin();
    scalar* __restrict__ lowerPtr_ddt = fvm_DDT.lower().begin();
    scalar* __restrict__ upperPtr_ddt = fvm_DDT.upper().begin();

    const labelUList& l = fvm_DDT.lduAddr().lowerAddr();
    const labelUList& u = fvm_DDT.lduAddr().upperAddr();

    // ddt
    const scalar* const __restrict__ rhoPtr = rho.primitiveField().begin();
    const scalar* const __restrict__ meshVscPtr = mesh.Vsc()().begin();
    const scalar* const __restrict__ rhoOldTimePtr = rho.oldTime().primitiveField().begin();
    const scalar* const __restrict__ YiOldTimePtr = Yi.oldTime().primitiveField().begin();

    // div
    const scalar* const __restrict__ weightsPtr = weights.primitiveField().begin();
    const scalar* const __restrict__ phiPtr = phi.primitiveField().begin();
    const scalar* const __restrict__ phiUcPtr = phiUc.primitiveField().begin();
    
    // laplacian
    // deltaCoeffs.primitiveField()*gammaMagSf.primitiveField();
    const scalar* const __restrict__ deltaCoeffsPtr = deltaCoeffs.primitiveField().begin();
    const scalar* const __restrict__ gammaMagSfPtr = gammaMagSf.primitiveField().begin();



    #pragma omp parallel for
    #pragma clang loop unroll_count(4)
    #pragma clang loop vectorize(enable)
    for(label c = 0; c < nCells; ++c){
        diagPtr_ddt[c] = rDeltaT * rhoPtr[c] * meshVscPtr[c];
        sourcePtr_ddt[c] = rDeltaT * rhoOldTimePtr[c] * YiOldTimePtr[c] * meshVscPtr[c];
    }

    // fvm_div1.lower() = -weights.primitiveField()*phi.primitiveField();
    // fvm_div1.upper() = fvm_div1.lower() + phi.primitiveField();

    // fvm_laplacian.upper() = deltaCoeffs.primitiveField()*gammaMagSf.primitiveField();

    // fvm_div1.negSumDiag();
    
    #pragma omp parallel for
    #pragma clang loop unroll_count(4)
    #pragma clang loop vectorize(enable)
    for(label f = 0; f < nFaces; ++f){
        lowerPtr_ddt[f] = - weightsPtr[f] * (phiPtr[f] + phiUcPtr[f]);
        upperPtr_ddt[f] = (- weightsPtr[f] + 1.) * (phiPtr[f] + phiUcPtr[f]);
        // upperPtr_ddt[f] = (- weightsPtr[f] + 1.) * (phiPtr[f] + phiUcPtr[f]) - deltaCoeffsPtr[f] * gammaMagSfPtr[f];
    }
   

    // fvm_laplacian.negSumDiag();

    // #pragma omp parallel for
    // #pragma clang loop unroll_count(4)
    // #pragma clang loop vectorize(enable)

    for (label face=0; face< nFaces; ++face)
    {
        diagPtr_ddt[l[face]] -= lowerPtr_ddt[face];
        diagPtr_ddt[u[face]] -= upperPtr_ddt[face];
    }

    forAll(Yi.boundaryField(), patchi)
    {
        const fvPatchField<scalar>& psf = Yi.boundaryField()[patchi];
        const fvsPatchScalarField& patchFlux_phi = phi.boundaryField()[patchi];
        const fvsPatchScalarField& patchFlux_phiUc = phiUc.boundaryField()[patchi];
        const fvsPatchScalarField& pw = weights.boundaryField()[patchi];

        const fvsPatchScalarField& pGamma = gammaMagSf.boundaryField()[patchi];
        const fvsPatchScalarField& pDeltaCoeffs = deltaCoeffs.boundaryField()[patchi];

        if (psf.coupled())
        {
            Info << "psf.coupled()" << endl;
            // fvm_DDT.internalCoeffs()[patchi] = (patchFlux_phi + patchFlux_phiUc) * psf.valueInternalCoeffs(pw) - pGamma * psf.gradientInternalCoeffs(pDeltaCoeffs);
            // fvm_DDT.boundaryCoeffs()[patchi] = - (patchFlux_phi + patchFlux_phiUc) * psf.valueBoundaryCoeffs(pw) + pGamma * psf.gradientBoundaryCoeffs(pDeltaCoeffs);
            fvm_DDT.internalCoeffs()[patchi] = (patchFlux_phi + patchFlux_phiUc) * psf.valueInternalCoeffs(pw);
            fvm_DDT.boundaryCoeffs()[patchi] = - (patchFlux_phi + patchFlux_phiUc) * psf.valueBoundaryCoeffs(pw);
        }
        else
        {
            Info << "not psf.coupled()" << endl;
            // fvm_DDT.internalCoeffs()[patchi] = (patchFlux_phi + patchFlux_phiUc) * psf.valueInternalCoeffs(pw) - pGamma * psf.gradientInternalCoeffs();
            // fvm_DDT.boundaryCoeffs()[patchi] = - (patchFlux_phi + patchFlux_phiUc) * psf.valueBoundaryCoeffs(pw) + pGamma * psf.gradientBoundaryCoeffs();
            fvm_DDT.internalCoeffs()[patchi] = (patchFlux_phi + patchFlux_phiUc) * psf.valueInternalCoeffs(pw);
            fvm_DDT.boundaryCoeffs()[patchi] = - (patchFlux_phi + patchFlux_phiUc) * psf.valueBoundaryCoeffs(pw);
        }
        // const auto& psfValueInternalCoeffs = psf.valueInternalCoeffs(pw)();
        // const auto& psfValueBoundaryCoeffs = psf.valueBoundaryCoeffs(pw)();

        // const scalar* const __restrict__ patchFluxPtr_div1 = patchFlux.begin();
        // const scalar* const __restrict__ psfValueInternalCoeffsPtr_div1 = psfValueInternalCoeffs.begin();
        // const scalar* const __restrict__ psfValueBoundaryCoeffsPtr_div1 = psfValueBoundaryCoeffs.begin();

        // scalar* __restrict__ internalCoeffsPtr_div1 = fvm_div1.internalCoeffs()[patchi].begin();
        // scalar* __restrict__ boundaryCoeffsPtr_div1 = fvm_div1.boundaryCoeffs()[patchi].begin();

        // #pragma omp parallel for
        // #pragma clang loop unroll_count(4)
        // #pragma clang loop vectorize(enable)
        // for(label i = 0; i < patchFlux.size(); ++i){
        //     internalCoeffsPtr_div1[i] = patchFluxPtr_div1[i] * psfValueInternalCoeffsPtr_div1[i];
        //     boundaryCoeffsPtr_div1[i] = - patchFluxPtr_div1[i] * psfValueBoundaryCoeffsPtr_div1[i];
        // }
    }

    // if (tinterpScheme_().corrected())
    // {
    //     Info << "tinterpScheme_().corrected() 1" << endl;
    //     fvm_DDT += fvc::surfaceIntegrate(phi*tinterpScheme_().correction(Yi));
    // }

    // if (tinterpScheme_().corrected())
    // {
    //     Info << "tinterpScheme_().corrected() 2" << endl;
    //     fvm_DDT += fvc::surfaceIntegrate(phiUc*tinterpScheme_().correction(Yi));
    // }

    // if (tsnGradScheme_().corrected())
    // {
    //     Info << "tsnGradScheme_().corrected()" << endl;

    //     if (mesh.fluxRequired(Yi.name()))
    //     {
    //         Info << "mesh.fluxRequired(Yi.name())" << endl;
    //         fvm_laplacian.faceFluxCorrectionPtr() = new
    //         surfaceScalarField
    //         (
    //             gammaMagSf * tsnGradScheme_().correction(Yi)
    //         );

    //         fvm_laplacian.source() -=
    //             mesh.V()*
    //             fvc::div
    //             (
    //                 *fvm_laplacian.faceFluxCorrectionPtr()
    //             )().primitiveField();
    //     }
    //     else
    //     {
    //         Info << "not mesh.fluxRequired(Yi.name())" << endl;
    //         fvm_laplacian.source() -=
    //             mesh.V()*
    //             fvc::div
    //             (
    //                 gammaMagSf*tsnGradScheme_().correction(Yi)
    //             )().primitiveField();
    //     }
    // }

    // -------------------------------------------------------

    tmp<volScalarField> DEff = rhoD + mut/Sct;

    tmp<fvScalarMatrix> tfvm
    (
        new fvScalarMatrix
        (
            (tfvm_DDT
            - gaussLaplacianSchemeFvmLaplacian(DEff(), Yi))
            // + tfvm_div1
            // + tfvm_div2
            // + mvConvection.fvmDiv(phi, Yi)
            // + mvConvection.fvmDiv(phiUc, Yi)
            // ==
            // (
            //     splitting
            // ?   gaussLaplacianSchemeFvmLaplacian(DEff(), Yi)
            // :  (gaussLaplacianSchemeFvmLaplacian(DEff(), Yi) + combustion.R(Yi))
            // )
            // (
            //     splitting
            // ?   tfvm_laplacian
            // :  (tfvm_laplacian + combustion.R(Yi))
            // )
            == 
            (combustion.R(Yi))
        )
    );

    return tfvm;
}


}
