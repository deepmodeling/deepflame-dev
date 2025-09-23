#pragma once

#include "tmp.H"
#include "dimensionedType.H"
#include "volFieldsFwd.H"
#include "surfaceFieldsFwd.H"
#include "typeInfo.H"
#include "runTimeSelectionTables.H"
#include "fvMatrices.H"
#include "fvMesh.H"
#include "turbulentFluidThermoModel.H"
#include "CombustionModel.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{


// namespace fv
// {

// fvm::ddt
template<class Type>
tmp<fvMatrix<Type>>
EulerDdtSchemeFvmDdt
(
    const volScalarField& rho,
    const GeometricField<Type, fvPatchField, volMesh>& vf
);

// fvm::div
template<class Type>
tmp<fvMatrix<Type>>
gaussConvectionSchemeFvmDiv
(
    const surfaceScalarField& faceFlux,
    const GeometricField<Type, fvPatchField, volMesh>& vf
);

template<class Type>
tmp<fvMatrix<Type>>
gaussConvectionSchemeFvmDiv
(
    const surfaceScalarField& faceFlux,
    const GeometricField<Type, fvPatchField, volMesh>& vf,
    const word& name
);

// fvc::grad
template<class Type>
tmp
<
    GeometricField
    <
        typename outerProduct<vector, Type>::type,
        fvPatchField,
        volMesh
    >
>
gaussGradSchemeGrad
(
    const GeometricField<Type, fvPatchField, volMesh>& vsf
);


// fvm::laplacian

template<class Type>
tmp<fvMatrix<Type>>
gaussLaplacianSchemeFvmLaplacianUncorrected
(
    const surfaceScalarField& gammaMagSf,
    const surfaceScalarField& deltaCoeffs,
    const GeometricField<Type, fvPatchField, volMesh>& vf
);

template<class Type>
tmp<fvMatrix<Type>>
gaussLaplacianSchemeFvmLaplacian
(
    const GeometricField<scalar, fvPatchField, volMesh>& gammaScalarVol,
    const GeometricField<Type, fvPatchField, volMesh>& vf
);

// turbulence->divDevRhoReff(U)
tmp<fvVectorMatrix>
turbulenceModelLinearViscousStressDivDevRhoReff
(
    volVectorField& U,
    compressible::turbulenceModel& turbulence
);


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
);



} // End namespace Foam


// ************************************************************************* //