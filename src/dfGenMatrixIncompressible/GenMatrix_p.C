#include "GenFvMatrix.H"

#include "gaussLaplacianScheme.H"
#include "surfaceInterpolate.H"
#include "fvcDiv.H"
#include "fvcGrad.H"
#include "fvMatrices.H"
#include "snGradScheme.H"
#include "linear.H"
#include "orthogonalSnGrad.H"

#include "pimpleControl.H"
#include "fvOptions.H"
#include "turbulentFluidThermoModel.H"

#include "incompressibleTwoPhaseMixture.H"
#include "immiscibleIncompressibleTwoPhaseMixture.H"
#include "turbulentTransportModel.H"

namespace Foam{

template<class Type>
tmp<fvMatrix<Type>>
GenMatrix_p(
    const GeometricField<scalar, fvsPatchField, surfaceMesh>& rAUf,
    const GeometricField<Type, fvPatchField, volMesh>& p_rgh,
    const tmp<GeometricField<double, fvsPatchField, surfaceMesh>>& phiHbyA
){
    Info << "pEqn gaussLaplacianSchemeFvmLaplacian" << endl;
    
    const fvMesh& mesh = p_rgh.mesh();
    tmp<fv::snGradScheme<scalar>> tsnGradScheme_(new fv::correctedSnGrad<scalar>(mesh));

    GeometricField<scalar, fvsPatchField, surfaceMesh> gammaMagSf
    (
        rAUf*mesh.magSf()
    );

    const surfaceScalarField& deltaCoeffs = tsnGradScheme_().deltaCoeffs(p_rgh);
    

    // -----
    tmp<fvMatrix<Type>> tfvm_Lap
    (
        new fvScalarMatrix
        (
            p_rgh,
            deltaCoeffs.dimensions()*gammaMagSf.dimensions()*p_rgh.dimensions()
        )
    );

    fvScalarMatrix& fvm = tfvm_Lap.ref();

    // -----

    scalar* __restrict__ sourcePtrs = fvm_laplace.source().begin();
    scalar* __restrict__ diagPtrs = fvm_laplace.diag().begin();  
    scalar* __restrict__ lowerPtrs = fvm_laplace.lower().begin();
    scalar* __restrict__ upperPtrs = fvm_laplace.upper().begin();

    const labelUList& ls = fvm_laplace.lduAddr().lowerAddr();
    const labelUList& us = fvm_laplace.lduAddr().upperAddr();

    // 获取场数据指针
    const scalar* const __restrict__ gammaMagSfPtrs = gammaMagSf.primitiveField().begin();
    const scalar* const __restrict__ deltaCoeffsPtrs = deltaCoeffs.primitiveField().begin();
    const scalar* const __restrict__ meshVPtrs = mesh.V().begin();

    const label nFacess = fvm_laplace.lower().size();
    const label nCellss = fvm_laplace.diag().size();

    // 初始化矩阵
    for (label celli = 0; celli < nCellss; celli++)
    {
        diagPtrs[celli] = 0.0;
        sourcePtrs[celli] = 0.0;
    }

    // 设置内部场矩阵系数
    for (label facei = 0; facei < nFacess; facei++)
    {
        scalar coeffs = deltaCoeffsPtrs[facei] * gammaMagSfPtrs[facei];
        upperPtrs[facei] = +coeffs;
        lowerPtrs[facei] = +coeffs;
    }

    // 对角线求和
    for (label facei = 0; facei < nFacess; facei++)
    {
        diagPtrs[ls[facei]] -= lowerPtrs[facei];  
        diagPtrs[us[facei]] -= upperPtrs[facei]; 
    }
    

    // fvm.upper() = deltaCoeffs.primitiveField()*gammaMagSf.primitiveField();
    // fvm.negSumDiag();

    forAll(p_rgh.boundaryField(), patchi)
    {
        const fvPatchField<Type>& pvf = p_rgh.boundaryField()[patchi];
        const fvsPatchScalarField& pGamma = gammaMagSf.boundaryField()[patchi];
        const fvsPatchScalarField& pDeltaCoeffs =
            deltaCoeffs.boundaryField()[patchi];

        if (pvf.coupled())
        {
            fvm.internalCoeffs()[patchi] =
                pGamma*pvf.gradientInternalCoeffs(pDeltaCoeffs);
            fvm.boundaryCoeffs()[patchi] =
               -pGamma*pvf.gradientBoundaryCoeffs(pDeltaCoeffs);
        }
        else
        {
            fvm.internalCoeffs()[patchi] = pGamma*pvf.gradientInternalCoeffs();
            fvm.boundaryCoeffs()[patchi] = -pGamma*pvf.gradientBoundaryCoeffs();
        }
    }

    if (mesh.fluxRequired(p_rgh.name()))
    {
        fvm.faceFluxCorrectionPtr() = new
        GeometricField<Type, fvsPatchField, surfaceMesh>
        (
            gammaMagSf*tsnGradScheme_().correction(p_rgh)
        );

        fvm.source() -=
            mesh.V()*
            fvc::div
            (
                *fvm.faceFluxCorrectionPtr()
            )().primitiveField();
    }
    else
    {
        fvm.source() -=
            mesh.V()*
            fvc::div
            (
                gammaMagSf*tsnGradScheme_().correction(p_rgh)
            )().primitiveField();
    }
    // -------------------------------------------------------

    Info << "pEqn gaussConvectionFvcDiv" << endl;
    tmp<volScalarField> tdivPhiHbyA = tmp<volScalarField>
    (
        new volScalarField
        (
            "div(" + phiHbyA().name() + ')',
            fvc::surfaceIntegrate(phiHbyA())
        )
    );

    fvm.source() += mesh.V() * tdivPhiHbyA().primitiveField();


    
    // -------------------------------------------------------
    // interFoam   pEqn.H

    tmp<fvScalarMatrix> tfvm
    (
            tfvm_Lap
        // + fvm::laplacian(rAUf, p_rgh) 
        // - fvc::div(phiHbyA)
    );

    return tfvm;
}

}


namespace Foam {
    template tmp<fvMatrix<scalar>> GenMatrix_p<scalar>(
        const GeometricField<scalar, fvsPatchField, surfaceMesh>&,
        const GeometricField<scalar, fvPatchField, volMesh>&,
        const tmp<GeometricField<double, fvsPatchField, surfaceMesh>>&
    );
}