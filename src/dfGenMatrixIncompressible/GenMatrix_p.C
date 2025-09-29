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
    tmp<fv::snGradScheme<Type>> tsnGradScheme_(new fv::orthogonalSnGrad<Type>(mesh));

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

    fvm.upper() = deltaCoeffs.primitiveField()*gammaMagSf.primitiveField();
    fvm.negSumDiag();

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

    // -----

    if (tsnGradScheme_().corrected())
    {
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
    }

    // -------------------------------------------------------

    // tmp<GeometricField<Type, fvPatchField, volMesh>>
    // (
    //     new GeometricField<Type, fvPatchField, volMesh>
    //     (
    //         "div("+phiHbyA->name()+')',
    //         fvcSurfaceIntegrate(phiHbyA)
    //     )
    // );

    // interFoam   pEqn.H

    tmp<fvScalarMatrix> tfvm
    (
            tfvm_Lap
        // + fvm::laplacian(rAUf, p_rgh) 
        - fvc::div(phiHbyA)
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