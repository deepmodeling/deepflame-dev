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
    fvm.lower() = -weights.primitiveField()*rhoPhi.primitiveField();
    // fvm.upper() = fvm.lower() + rhoPhi.primitiveField();
    fvm.upper() = -weights.primitiveField()*rhoPhi.primitiveField() + rhoPhi.primitiveField();
    fvm.negSumDiag();   // diag[i] = - (sum_of_lower_coeffs[i] + sum_of_upper_coeffs[i])  先设置对流项
    
    fvm.diag() += rDeltaT*rho.primitiveField()*mesh.Vsc(); // 再加上瞬态项
    fvm.source() = rDeltaT
        *rho.oldTime().primitiveField()
        *U.oldTime().primitiveField()*mesh.Vsc();

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



    // scalar* __restrict__ diagPtr_ddt = fvm.diag().begin();
    // scalar* __restrict__ sourcePtr_ddt = fvm.source().begin();
    // scalar* __restrict__ lowerPtr_ddt = fvm.lower().begin();
    // scalar* __restrict__ upperPtr_ddt = fvm.upper().begin();

    // const labelUList& l = fvm.lduAddr().lowerAddr();
    // const labelUList& u = fvm.lduAddr().upperAddr();

    // interFoam    

    tmp<fvVectorMatrix> tfvm
    (
        new fvVectorMatrix
        (
            (
                tfvm_DDT
            // + fvm::ddt(rho, U) 
            // + fvm::div(rhoPhi, U)
            )
            // + MRF.DDt(rho, U) 

        ==
            (- turbulence.divDevRhoReff(rho, U))  // fvOptions(rho, U)
        )
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