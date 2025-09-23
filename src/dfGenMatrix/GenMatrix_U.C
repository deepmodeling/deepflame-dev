#include "GenFvMatrix.H"
#include "multivariateGaussConvectionScheme.H"
#include "gaussConvectionScheme.H"
#include "snGradScheme.H"
#include "linear.H"
#include "orthogonalSnGrad.H"

#include "pimpleControl.H"
#include "fvOptions.H"
#include "turbulentFluidThermoModel.H"

namespace Foam{

tmp<fvVectorMatrix>
GenMatrix_U(
    const volScalarField& rho,
    volVectorField& U,
    const surfaceScalarField& rhoPhi,   // phi  @dfLowMachFoam
    const volScalarField& p, 
    compressible::turbulenceModel& turbulence
){

    Info << "UEqn EulerDdtSchemeFvmDdt" << endl;
    const fvMesh& mesh = U.mesh();

    tmp<fvVectorMatrix> tfvm_DDT
    (
        new fvVectorMatrix
        (
            U,
            rho.dimensions()*U.dimensions()*dimVol/dimTime
        )
    );
    fvVectorMatrix& fvm_DDT = tfvm_DDT.ref();

    scalar rDeltaT = 1.0/mesh.time().deltaTValue();

    fvm_DDT.diag() = rDeltaT*rho.primitiveField()*mesh.Vsc();
    fvm_DDT.source() = rDeltaT
        *rho.oldTime().primitiveField()
        *U.oldTime().primitiveField()*mesh.Vsc();

    // scalar* __restrict__ diagPtr_ddt = fvm_DDT.diag().begin();
    // scalar* __restrict__ sourcePtr_ddt = fvm_DDT.source().begin();
    // scalar* __restrict__ lowerPtr_ddt = fvm_DDT.lower().begin();
    // scalar* __restrict__ upperPtr_ddt = fvm_DDT.upper().begin();

    // const labelUList& l = fvm_DDT.lduAddr().lowerAddr();
    // const labelUList& u = fvm_DDT.lduAddr().upperAddr();

    // interFoam    

    tmp<fvVectorMatrix> tfvm
    (
        new fvVectorMatrix
        (
            (tfvm_DDT
        // + fvm::ddt(rho, U) 
        + fvm::div(rhoPhi, U))
        // + MRF.DDt(rho, U) 

        ==
            (- turbulence.divDevRhoReff(U))  // fvOptions(rho, U)
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