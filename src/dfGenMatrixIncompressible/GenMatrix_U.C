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
    // fvm.diag() += rDeltaT*rho.primitiveField()*mesh.Vsc(); // 再加上瞬态项
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

    // 然后从 off-diagonal 项累积到对角线
    for (label facei = 0; facei < nFaces; facei++)
    {
        diagPtr[l[facei]] -= lowerPtr[facei];  // 从下三角累积
        diagPtr[u[facei]] -= upperPtr[facei];  // 从上三角累积
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

    // -------------------------------------------------------

    

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
        - fvm::laplacian(turbulence.alpha()*rho*turbulence.nuEff(), U)
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