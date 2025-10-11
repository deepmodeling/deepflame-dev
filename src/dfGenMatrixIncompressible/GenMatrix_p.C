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
#include <mpi.h>

double p_rghEqn_build_pre_time = 0., p_rghEqn_build_item1_intern_time = 0., p_rghEqn_build_item1_bound_time = 0., p_rghEqn_build_item2_time = 0.;
double start, end;

namespace Foam{

template<class Type>
tmp<fvMatrix<Type>>
GenMatrix_p(
    const GeometricField<scalar, fvsPatchField, surfaceMesh>& rAUff,
    const GeometricField<Type, fvPatchField, volMesh>& p_rgh,
    const tmp<GeometricField<double, fvsPatchField, surfaceMesh>>& phiHbyA
){
    p_rghEqn_build_pre_time = 0., p_rghEqn_build_item1_intern_time = 0., p_rghEqn_build_item1_bound_time = 0., p_rghEqn_build_item2_time = 0.;
    Info << "pEqn gaussLaplacianSchemeFvmLaplacian" << endl;
    start = MPI_Wtime();
    
    const fvMesh& mesh = p_rgh.mesh();
    tmp<fv::snGradScheme<scalar>> tsnGradScheme_(new fv::correctedSnGrad<scalar>(mesh));

    surfaceScalarField rAUf = rAUff;

    GeometricField<scalar, fvsPatchField, surfaceMesh> gammaMagSf
    (
        rAUf*mesh.magSf()
    );

    const surfaceScalarField& deltaCoeffs = tsnGradScheme_().deltaCoeffs(p_rgh);

    // -----

    tmp<volScalarField> tdivPhiHbyA = tmp<volScalarField>
    (
        new volScalarField
        (
            "div(" + phiHbyA().name() + ')',
            fvc::surfaceIntegrate(phiHbyA())
        )
    );
    
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
    
    end = MPI_Wtime();
    p_rghEqn_build_pre_time += end - start;
    // -----
    start = MPI_Wtime();

    fvm.upper() = deltaCoeffs.primitiveField()*gammaMagSf.primitiveField();
    fvm.negSumDiag();

    // scalar* __restrict__ sourcePtrs = fvm.source().begin();
    // scalar* __restrict__ diagPtrs = fvm.diag().begin();  
    // scalar* __restrict__ lowerPtrs = fvm.lower().begin();
    // scalar* __restrict__ upperPtrs = fvm.upper().begin();

    // const labelUList& ls = fvm.lduAddr().lowerAddr();
    // const labelUList& us = fvm.lduAddr().upperAddr();

    // // 获取场数据指针
    // const scalar* const __restrict__ gammaMagSfPtrs = gammaMagSf.primitiveField().begin();
    // const scalar* const __restrict__ deltaCoeffsPtrs = deltaCoeffs.primitiveField().begin();
    // const scalar* const __restrict__ meshVPtrs = mesh.V().begin();

    // const label nFacess = fvm.lower().size();
    // const label nCellss = fvm.diag().size();

    // // // 构造上三角和下三角系数
    // // for (label face = 0; face < nFacess; face++)
    // // {
    // //     scalar coeff = deltaCoeffsPtrs[face] * gammaMagSfPtrs[face];
    // //     upperPtrs[face] = coeff;
    // //     lowerPtrs[face] = coeff;
    // // }

    // // 执行负对角和
    // // 首先清零对角线
    // for (label cell = 0; cell < nCellss; cell++)
    // {
    //     diagPtrs[cell] = 0.0;
    // }

    // // 累加相邻单元贡献
    // for (label face = 0; face < nFacess; face++)
    // {
    //     diagPtrs[ls[face]] += upperPtrs[face];
    //     diagPtrs[us[face]] += upperPtrs[face];
    // }

    // // 取负值
    // for (label cell = 0; cell < nCellss; cell++)
    // {
    //     diagPtrs[cell] = -diagPtrs[cell];
    // }
    

    end = MPI_Wtime();
    p_rghEqn_build_item1_intern_time += end - start;

    start = MPI_Wtime();

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

    end = MPI_Wtime();
    p_rghEqn_build_item1_bound_time += end - start;
    // -------------------------------------------------------
    
    Info << "pEqn gaussConvectionFvcDiv" << endl;
    start = MPI_Wtime();

    fvm.source() += mesh.V() * tdivPhiHbyA().primitiveField();

    end = MPI_Wtime();
    p_rghEqn_build_item2_time += end - start;

    // scalar* __restrict__ sourcePtrs = fvm.source().begin();
    // const scalar* const __restrict__ divPhiHbyAPtrs = tdivPhiHbyA().primitiveField().begin();
    // const scalar* const __restrict__ meshVPtrs = mesh.V().begin();
    // const label nCellss = fvm.diag().size();

    // for (label cell = 0; cell < nCellss; cell++)
    // {
    //     sourcePtrs[cell] = 0.0;
    //     sourcePtrs[cell] += meshVPtrs[cell]*divPhiHbyAPtrs[cell];
    // }
    
    // -------------------------------------------------------
    // interFoam   pEqn.H

    // tmp<fvScalarMatrix> tfvm
    // (
    //         tfvm_Lap
    //     // + fvm::laplacian(rAUf, p_rgh) 
    //     // - fvc::div(phiHbyA)
    // );

    Info<< "========Time Spent in pEqn build once=================="<< endl;
    Info << "p pre Time :   " << p_rghEqn_build_pre_time << endl;
    Info << "p item1 intern Time : " << p_rghEqn_build_item1_intern_time << endl;
    Info << "p item1 bound Time : " << p_rghEqn_build_item1_bound_time << endl;
    Info << "p item2 Time : " << p_rghEqn_build_item2_time << endl;
    Info<< "============================================"<<nl<< endl;

    // return tfvm;
    return tfvm_Lap;
}

}


namespace Foam {
    template tmp<fvMatrix<scalar>> GenMatrix_p<scalar>(
        const GeometricField<scalar, fvsPatchField, surfaceMesh>&,
        const GeometricField<scalar, fvPatchField, volMesh>&,
        const tmp<GeometricField<double, fvsPatchField, surfaceMesh>>&
    );
}