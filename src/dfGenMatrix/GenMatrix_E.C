#include "GenFvMatrix.H"
#include "multivariateGaussConvectionScheme.H"
#include "gaussConvectionScheme.H"
#include "snGradScheme.H"
#include "linear.H"
#include "orthogonalSnGrad.H"
#include <typeinfo>
#include "ITstream.H"

#define TIME
#ifdef TIME 
#define TICK0(prefix)\
    double prefix##_tick_0 = MPI_Wtime();

#define TICK(prefix,start,end)\
    double prefix##_tick_##end = MPI_Wtime();\
    Info << #prefix << "_time_" << #end << " : " << prefix##_tick_##end - prefix##_tick_##start << endl;

#else
#define TICK0(prefix) ;
#define TICK(prefix,start,end) ;
#endif

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
    const fvMesh& mesh = he.mesh();
    assert(mesh.moving() == false);

    label nCells = mesh.nCells();
    label nFaces = mesh.neighbour().size();

    const labelUList& l = mesh.lduAddr().lowerAddr();
    const labelUList& u = mesh.lduAddr().upperAddr();

    scalar rDeltaT = 1.0/mesh.time().deltaTValue();

    const scalar* const __restrict__ rhoPtr = rho.primitiveField().begin();
    const scalar* const __restrict__ meshVscPtr = mesh.Vsc()().begin();
    const scalar* const __restrict__ meshVPtr = mesh.V().begin();
    const scalar* const __restrict__ rhoOldTimePtr = rho.oldTime().primitiveField().begin();
    const scalar* const __restrict__ heOldTimePtr = he.oldTime().primitiveField().begin();

    // ITstream ITstream_surfaceInterpolationScheme_limitedLinear_1("surfaceInterpolationScheme_limitedLinear_1", tokenList{token(word("limitedLinear")), token(label(1))});
    // tmp<surfaceInterpolationScheme<scalar>> tsurfaceInterpolationScheme_limitedLinear_1  = surfaceInterpolationScheme<scalar>::New(mesh, phi, ITstream_surfaceInterpolationScheme_limitedLinear_1);

    tmp<fv::convectionScheme<scalar>> cs_he = fv::convectionScheme<scalar>::New(mesh, phi, mesh.divScheme("div("+phi.name()+','+he.name()+')'));
    fv::gaussConvectionScheme<scalar>& gcs_he = dynamic_cast<fv::gaussConvectionScheme<scalar>&>(cs_he.ref());
    tmp<fv::convectionScheme<scalar>> cs_K = fv::convectionScheme<scalar>::New(mesh, phi, mesh.divScheme("div("+phi.name()+','+K.name()+')'));
    fv::gaussConvectionScheme<scalar>& gcs_K = dynamic_cast<fv::gaussConvectionScheme<scalar>&>(cs_K.ref());

    tmp<surfaceInterpolationScheme<vector>> surfaceInterpolationScheme_vector_linear(new linear<vector>(mesh));
    tmp<surfaceInterpolationScheme<scalar>> surfaceInterpolationScheme_linear(new linear<scalar>(mesh));
    tmp<fv::snGradScheme<scalar>> snGradScheme_orthogonal(new fv::orthogonalSnGrad<scalar>(mesh));

    // TODO expancive
    // tmp<surfaceScalarField> tweights_he = surfaceInterpolationScheme_linear().weights(he);
    // tmp<surfaceScalarField> tweights_he = tsurfaceInterpolationScheme_limitedLinear_1().weights(he);
    tmp<surfaceScalarField> tweights_he = gcs_he.interpScheme().weights(he);
    const surfaceScalarField& weights_he = tweights_he();    

    // TODO expancive
    // tmp<surfaceScalarField> tweights_K = surfaceInterpolationScheme_linear().weights(K);
    // tmp<surfaceScalarField> tweights_K = tsurfaceInterpolationScheme_limitedLinear_1().weights(K);
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
    surfaceScalarField Alphaf(
        IOobject
        (
            "interpolate("+alphaEff.name()+')',
            alphaEff.instance(),
            alphaEff.db()
        ),
        mesh,
        mesh.Sf().dimensions()*alphaEff.dimensions()
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

    scalar* KfPtr = Kf.begin();
    const scalar* const __restrict__ weights_K_Ptr = weights_K.primitiveField().begin(); // get weight
    const scalar* const __restrict__ KPtr = K.primitiveField().begin();
    
    scalar* alphafPtr = Alphaf.begin();
    const scalar* const __restrict__ laplacianWeightsPtr = laplacianWeight.primitiveField().begin();
    const scalar* const __restrict__ alphaPtr = alphaEff.primitiveField().begin();

    scalar* hDiffCorrFluxfPtr = hDiffCorrFluxf.begin();
    const scalar* const __restrict__ weightshDiffCorrFluxPtr = weightshDiffCorrFlux.primitiveField().begin(); // get weight
    
    #pragma omp parallel for
    for(label f = 0; f < nFaces; ++f){
        KfPtr[f] = weights_K_Ptr[f] * (KPtr[l[f]] - KPtr[u[f]]) + KPtr[u[f]];
        alphafPtr[f] = laplacianWeightsPtr[f] * (alphaPtr[l[f]] - alphaPtr[u[f]]) + alphaPtr[u[f]];
        hDiffCorrFluxfPtr[f] = mesh.Sf()[f] & (weightshDiffCorrFluxPtr[f] * (hDiffCorrFlux[l[f]] - hDiffCorrFlux[u[f]]) + hDiffCorrFlux[u[f]]);
    }

    forAll(weights_K.boundaryField(), Ki)
    {
        const fvsPatchScalarField& pLambda = weights_K.boundaryField()[Ki];
        fvsPatchScalarField& psf = Kf.boundaryFieldRef()[Ki];
        if (K.boundaryField()[Ki].coupled())
        {
            psf =(pLambda*K.boundaryField()[Ki].patchInternalField() + (1.0 - pLambda)*K.boundaryField()[Ki].patchNeighbourField());
        }
        else
        {
            psf = K.boundaryField()[Ki];
        }
    }

    forAll(linear_weights.boundaryField(), Ki)
    {
        const fvsPatchScalarField& pLambda = linear_weights.boundaryField()[Ki];
        fvsPatchScalarField& psf = Alphaf.boundaryFieldRef()[Ki];
        if (alphaEff.boundaryField()[Ki].coupled())
        {
            psf =(pLambda*alphaEff.boundaryField()[Ki].patchInternalField()+ (1.0 - pLambda)*alphaEff.boundaryField()[Ki].patchNeighbourField());
        }
        else
        {
            psf = alphaEff.boundaryField()[Ki];
        }
    }

    forAll(weightshDiffCorrFlux.boundaryField(), Ki)
    {
        const fvsPatchScalarField& pLambda = weightshDiffCorrFlux.boundaryField()[Ki];
        const fvsPatchVectorField& pSf = mesh.Sf().boundaryField()[Ki];
        fvsPatchScalarField& psf = hDiffCorrFluxf.boundaryFieldRef()[Ki];

        if (K.boundaryField()[Ki].coupled())
        {
            psf = pSf
              & (
                    pLambda*hDiffCorrFlux.boundaryField()[Ki].patchInternalField()
                + (1.0 - pLambda)*hDiffCorrFlux.boundaryField()[Ki].patchNeighbourField()
                );
        }
        else
        {
            psf = pSf & hDiffCorrFlux.boundaryField()[Ki];
        }
    }

    surfaceScalarField gammaMagSf(Alphaf * mesh.magSf());
    
    // expansive
    tmp<fvScalarMatrix> tfvm
    (
        new fvScalarMatrix
        (
            he,
            rho.dimensions()*he.dimensions()*dimVol/dimTime
        )
    );

    fvScalarMatrix& fvm = tfvm.ref();

    scalar* __restrict__ diagPtr = fvm.diag().begin();
    scalar* __restrict__ sourcePtr = fvm.source().begin();
    scalar* __restrict__ lowerPtr = fvm.lower().begin();
    scalar* __restrict__ upperPtr = fvm.upper().begin();

    const scalar* const __restrict__ weights_he_Ptr = weights_he.primitiveField().begin(); // get weight
    const scalar* const __restrict__ phiPtr = phi.primitiveField().begin();

    const scalar* const __restrict__ deltaCoeffsPtr = deltaCoeffs.primitiveField().begin();
    const scalar* const __restrict__ gammaMagSfPtr = gammaMagSf.primitiveField().begin();

    const scalar* const __restrict__ KOldTimePtr = K.oldTime().primitiveField().begin();

    double *fvcDivPtr = new double[nCells]{0.};
    double *fvcDiv2Ptr = new double[nCells]{0.};
    double *diagLaplacPtr = new double[nCells]{0.};
    
    for(label f = 0; f < nFaces; ++f){
        scalar var1 = - weights_he_Ptr[f] * phiPtr[f];
        scalar var2 = - weights_he_Ptr[f] * phiPtr[f] + phiPtr[f];
        lowerPtr[f] += var1;
        upperPtr[f] += var2;
        diagPtr[l[f]] -= var1;
        diagPtr[u[f]] -= var2;
        fvcDivPtr[l[f]] += phiPtr[f] * KfPtr[f];
        fvcDivPtr[u[f]] -= phiPtr[f] * KfPtr[f];
        fvcDiv2Ptr[l[f]] += hDiffCorrFluxf[f];
        fvcDiv2Ptr[u[f]] -= hDiffCorrFluxf[f];
    }

    for(label f = 0; f < nFaces; ++f){
        scalar var1 = deltaCoeffsPtr[f] * gammaMagSfPtr[f];
        lowerPtr[f] -= var1;
        upperPtr[f] -= var1;
        diagLaplacPtr[l[f]] += var1; //negSumDiag
        diagLaplacPtr[u[f]] += var1;
    }

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

    #pragma omp parallel for
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
    return tfvm;
}

}
