#include "GenFvMatrix.H"
#include "multivariateGaussConvectionScheme.H"
#include "gaussConvectionScheme.H"
#include "snGradScheme.H"
#include "linear.H"
#include "orthogonalSnGrad.H"

#include "pimpleControl.H"
#include "fvOptions.H"
#include "turbulentFluidThermoModel.H"

#include "incompressibleTwoPhaseMixture.H"
#include "immiscibleIncompressibleTwoPhaseMixture.H"
#include "turbulentTransportModel.H"

#include "turbulenceModel.H"
#include "autoPtr.H"
#include "runTimeSelectionTables.H"

namespace Foam{

tmp<fvVectorMatrix>
GenMatrix_U(
    const volScalarField& rho,
    volVectorField& U,
    const surfaceScalarField& rhoPhi,   // phi  @dfLowMachFoam
    const volScalarField& p, 
    incompressible::turbulenceModel& turbulence
){
    word name("div("+rhoPhi.name()+','+U.name()+')');

    const fvMesh& mesh = U.mesh();
    // div
    tmp<fv::convectionScheme<vector>> cs = fv::convectionScheme<vector>::New(mesh,rhoPhi,mesh.divScheme(name));
    fv::gaussConvectionScheme<vector>& gcs = dynamic_cast<fv::gaussConvectionScheme<vector>&>(cs.ref());
    tmp<surfaceScalarField> tweights = gcs.interpScheme().weights(U);
    const surfaceScalarField& weights = tweights();

    // -------------------------------------------------------

    Info << "UEqn EulerDdtSchemeFvmDdt" << endl;
    Info << "UEqn gaussConvectionSchemeFvmDiv" << endl;

    tmp<fvVectorMatrix> tfvm_DDT
    (
        new fvVectorMatrix
        (
            U,
            rho.dimensions()*U.dimensions()*dimVol/dimTime
        )
    );
    fvVectorMatrix& fvm = tfvm_DDT.ref();

    scalar rDeltaT = 1.0/mesh.time().deltaTValue();

    // interField
    // fvm.lower() = -weights.primitiveField()*rhoPhi.primitiveField();
    // // fvm.upper() = fvm.lower() + rhoPhi.primitiveField();
    // fvm.upper() = -weights.primitiveField()*rhoPhi.primitiveField() + rhoPhi.primitiveField();
    // fvm.negSumDiag();   
    // fvm.diag() += rDeltaT*rho.primitiveField()*mesh.Vsc(); 
    // fvm.source() = rDeltaT
    //     *rho.oldTime().primitiveField()
    //     *U.oldTime().primitiveField()*mesh.Vsc();


    vector* __restrict__ sourcePtr = fvm.source().begin();
    scalar* __restrict__ diagPtr = fvm.diag().begin();  
    scalar* __restrict__ lowerPtr = fvm.lower().begin();
    scalar* __restrict__ upperPtr = fvm.upper().begin();

    const labelUList& l = fvm.lduAddr().lowerAddr();
    const labelUList& u = fvm.lduAddr().upperAddr();

    const scalar* const __restrict__ weightsPtr = weights.primitiveField().begin();
    const scalar* const __restrict__ rhoPhiPtr = rhoPhi.primitiveField().begin();
    const scalar* const __restrict__ rhoPtr = rho.primitiveField().begin();
    const scalar* const __restrict__ meshVscPtr = mesh.Vsc()().begin();
    const scalar* const __restrict__ rhoOldTimePtr = rho.oldTime().primitiveField().begin();

    const vector* const __restrict__ UOldTimePtr = U.oldTime().primitiveField().begin();

    const label nFaces = fvm.lower().size();
    const label nCells = fvm.diag().size();

    for (label facei = 0; facei < nFaces; facei++)
    {
        scalar flux = weightsPtr[facei] * rhoPhiPtr[facei];
        lowerPtr[facei] = -flux;
        upperPtr[facei] = -flux + rhoPhiPtr[facei];
    }

    for (label celli = 0; celli < nCells; celli++)
    {
        diagPtr[celli] = 0.0; // vector::zero;
    }

    // 
    for (label facei = 0; facei < nFaces; facei++)
    {
        diagPtr[l[facei]] -= lowerPtr[facei];  
        diagPtr[u[facei]] -= upperPtr[facei]; 
    }

    for (label celli = 0; celli < nCells; celli++)
    {
        diagPtr[celli] += rDeltaT * rhoPtr[celli] * meshVscPtr[celli];
        sourcePtr[celli] = rDeltaT * rhoOldTimePtr[celli] * UOldTimePtr[celli] * meshVscPtr[celli];
    }

    // boundaryField
    forAll(U.boundaryField(), patchi)
    {
        const fvPatchField<vector>& psf = U.boundaryField()[patchi];
        const fvsPatchScalarField& patchFlux = rhoPhi.boundaryField()[patchi];
        const fvsPatchScalarField& pw = weights.boundaryField()[patchi];

        fvm.internalCoeffs()[patchi] = patchFlux*psf.valueInternalCoeffs(pw);
        fvm.boundaryCoeffs()[patchi] = -patchFlux*psf.valueBoundaryCoeffs(pw);
    }

    // correct
    if (gcs.interpScheme().corrected())
    {
        fvm += fvc::surfaceIntegrate(rhoPhi*gcs.interpScheme().correction(U));
    }

    // --------------------------------------------------------------------------------------------------------------

    Info << "UEqn gaussLaplacianSchemeFvmLaplacian" << endl;
     
    const volScalarField alphaEff = turbulence.alpha()*rho*turbulence.nuEff();
    // tmp<fv::snGradScheme<vector>> tsnGradScheme_(new fv::orthogonalSnGrad<vector>(mesh));  // 正交修正方案
    tmp<fv::snGradScheme<vector>> tsnGradScheme_(new fv::correctedSnGrad<vector>(mesh));  // 对应snGradSchemes{default         corrected;}

    surfaceScalarField alphaEfff = fvc::interpolate(alphaEff);    ///  
    
    // gammaMagSf = alphaEfff * magSf
    GeometricField<scalar, fvsPatchField, surfaceMesh> gammaMagSf
    (
        alphaEfff * mesh.magSf()
    );
    
    const surfaceScalarField& deltaCoeffs = tsnGradScheme_().deltaCoeffs(U);
    // 
    tmp<fvVectorMatrix> tfvm_Laplacian
    (
        new fvVectorMatrix
        (
            U,
            deltaCoeffs.dimensions()*gammaMagSf.dimensions()*U.dimensions()
        )
    );
    fvVectorMatrix& fvm_laplace = tfvm_Laplacian.ref();
    
    // interField
    // fvm_laplace.upper() = deltaCoeffs.primitiveField()*gammaMagSf.primitiveField();
    // fvm_laplace.negSumDiag();

    vector* __restrict__ sourcePtrs = fvm_laplace.source().begin();
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
        sourcePtrs[celli] = vector::zero;
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

    // ------------------------------------
    forAll(U.boundaryField(), patchi)
    {
        const fvPatchVectorField& pvf = U.boundaryField()[patchi];
        const fvsPatchScalarField& pGamma = gammaMagSf.boundaryField()[patchi];
        const fvsPatchScalarField& pDeltaCoeffs = deltaCoeffs.boundaryField()[patchi];
        
        if (pvf.coupled())
        {
            fvm_laplace.internalCoeffs()[patchi] =
                pGamma * pvf.gradientInternalCoeffs(pDeltaCoeffs);
            fvm_laplace.boundaryCoeffs()[patchi] =
               -pGamma * pvf.gradientBoundaryCoeffs(pDeltaCoeffs);
        }
        else
        {
            fvm_laplace.internalCoeffs()[patchi] = pGamma * pvf.gradientInternalCoeffs();
            fvm_laplace.boundaryCoeffs()[patchi] = -pGamma * pvf.gradientBoundaryCoeffs();
        }
    }

    // 非正交修正,否则不需要
    if (mesh.fluxRequired(U.name()))
    {
        fvm_laplace.faceFluxCorrectionPtr() = new
        GeometricField<vector, fvsPatchField, surfaceMesh>
        (
            gammaMagSf * tsnGradScheme_().correction(U)
        );
        
        fvm_laplace.source() -=
            mesh.V() *
            fvc::div
            (
                *fvm_laplace.faceFluxCorrectionPtr()
            )().primitiveField();
    }
    else
    {
        fvm_laplace.source() -=
            mesh.V() *
            fvc::div
            (
                gammaMagSf * tsnGradScheme_().correction(U)
            )().primitiveField();
    }

    fvm -= fvm_laplace;

    // --------------------------------------------------------------------------------------------------------------














    
    
    // -------------------------------------------------------
    // interFoam    
    // const alphaField& alpha;

    tmp<fvVectorMatrix> tfvm
    (
            tfvm_DDT
        // + fvm::ddt(rho, U) 
        // + fvm::div(rhoPhi, U)
        // + MRF.DDt(rho, U) 
        // + turbulence.divDevRhoReff(rho, U)  
        - fvc::div((turbulence.alpha()*rho*turbulence.nuEff())*dev2(T(fvc::grad(U))))
        
        // - fvm::laplacian(turbulence.alpha()*rho*turbulence.nuEff(), U)
        // - fvOptions(rho, U)
    );

    // @dfLowMachFoam    
    // tmp<fvVectorMatrix> tUEqn
    // (
    //     fvm::ddt(rho, U) + fvm::div(phi, U)
    //     + turbulence->divDevRhoReff(U)
    //     == 
    //     -fvc::grad(p)
    // );     

    return tfvm;
}

}

// tmp<fvVectorMatrix> tUEqn = GenMatrix_U(rho, U, phi, p, turbulence());
// fvVectorMatrix& UEqn = tUEqn.ref();

// time_monitor_UEqn_pre += UEqnClock.timeIncrement();

// tmp<fvVectorMatrix> tUEqn_answer
// (
//         fvm::ddt(rho, U) 
//     + fvm::div(rhoPhi, U)
//     + MRF.DDt(rho, U)
//     + turbulence->divDevRhoReff(rho, U)
//     ==
//         fvOptions(rho, U)
// );
// fvVectorMatrix& UEqn_answer = tUEqn_answer.ref();
// check_fvmatrix_equal(UEqn_answer, UEqn, "UEqn");