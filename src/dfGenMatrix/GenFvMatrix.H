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
#include <mpi.h>
// #include <algorithm>
// #include "PstreamGlobals.H"
#include "StructuredMeshSchedule.H"
#include "processorFvPatchField.H"

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

template<class Type>
tmp<fvMatrix<Type>>
EulerDdtSchemeFvmDdt
(
    const GeometricField<Type, fvPatchField, volMesh>& vf
);

// fvc::ddt
template<class Type>
tmp<GeometricField<Type, fvPatchField, volMesh>>
EulerDdtSchemeFvcDdt
(
    const volScalarField& rho,
    const GeometricField<Type, fvPatchField, volMesh>& vf
);

template<class Type>
tmp<GeometricField<Type, fvPatchField, volMesh>>
EulerDdtSchemeFvcDdt
(
    const GeometricField<Type, fvPatchField, volMesh>& vf
);

// fvc::ddtCorr
tmp<surfaceScalarField>
EulerDdtSchemeFvcDdtCorr
(
    const volScalarField& rho,
    const volVectorField& U,
    const surfaceScalarField& phi,
    const autoPtr<surfaceVectorField>& Uf
);

tmp<surfaceScalarField>
EulerDdtSchemeFvcDdtPhiCoeff
(
    const volVectorField& U,
    const surfaceScalarField& phi,
    const surfaceScalarField& phiCorr,
    const volScalarField& rho
);

template<class Type>
Foam::tmp<Foam::GeometricField<Type, Foam::fvPatchField, Foam::volMesh>>
UEqn_H
(
    fvMatrix<Type>& UEqn
);

tmp<volScalarField>
rAUConstructor
(
    fvMatrix<vector>& UEqn
);

tmp<surfaceScalarField>
rhorAUfConstructor
(
    const volScalarField& rhorAU,
    const surfaceScalarField& linear_weights
);

tmp<surfaceScalarField>
phiHbyAConstructor
(
    const volScalarField& rho,
    const volVectorField& HbyA,
    const surfaceScalarField& rhorAUf,
    const surfaceScalarField& tddtCorr,
    const surfaceScalarField& linear_weights
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
tmp<GeometricField<Type, fvPatchField, volMesh>>
fvcSurfaceIntegrate
(
    const GeometricField<Type, fvsPatchField, surfaceMesh>& ssf
);

template<class Type>
tmp<GeometricField<Type, fvPatchField, volMesh>>
fvcSurfaceIntegrate
(
    const tmp<GeometricField<Type, fvsPatchField, surfaceMesh>>& tssf
);

template<class Type>
tmp<fvMatrix<Type>>
gaussConvectionSchemeFvmDiv
(
    const surfaceScalarField& faceFlux,
    const GeometricField<Type, fvPatchField, volMesh>& vf,
    const word& name
);

// fvc::div
template<class Type>
tmp<GeometricField<Type, fvPatchField, volMesh>>
gaussConvectionSchemeFvcDiv
(
    const surfaceScalarField& faceFlux,
    const GeometricField<Type, fvPatchField, volMesh>& vf
);

template<class Type>
tmp<GeometricField<Type, fvPatchField, volMesh>>
gaussConvectionSchemeFvcDiv
(
    const GeometricField<Type, fvsPatchField, surfaceMesh>& ssf
);

template<class Type>
tmp<GeometricField<Type, fvPatchField, volMesh>>
gaussConvectionSchemeFvcDiv
(
    const tmp<GeometricField<Type, fvsPatchField, surfaceMesh>>& tssf
);

template<class Type>
tmp
<
    GeometricField
    <
        typename innerProduct<vector, Type>::type, fvPatchField, volMesh
    >
>
gaussDivFvcdiv
(
    const GeometricField<Type, fvPatchField, volMesh>& vf
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


template<class Type>
tmp<fvMatrix<Type>>
gaussLaplacianSchemeFvmLaplacianUncorrected
(
    const surfaceScalarField& gammaMagSf,
    const surfaceScalarField& deltaCoeffs,
    const GeometricField<Type, fvPatchField, volMesh>& vf
);

// fvm::laplacian
template<class Type>
tmp<fvMatrix<Type>>
gaussLaplacianSchemeFvmLaplacian
(
    const GeometricField<scalar, fvPatchField, volMesh>& gammaScalarVol,
    const GeometricField<Type, fvPatchField, volMesh>& vf
);

template<class Type>
tmp<fvMatrix<Type>>
gaussLaplacianSchemeFvmLaplacian
(
    const GeometricField<scalar, fvsPatchField, surfaceMesh>& gamma,
    const GeometricField<Type, fvPatchField, volMesh>& vf
);

// fvc::laplacian
template<class Type>
tmp<GeometricField<Type, fvPatchField, volMesh>>
gaussLaplacianSchemeFvcLaplacian
(
    const GeometricField<scalar, fvPatchField, volMesh>& gamma,
    const GeometricField<Type, fvPatchField, volMesh>& vf
);

// turbulence->divDevRhoReff(U)
tmp<fvVectorMatrix>
turbulenceModelLinearViscousStressDivDevRhoReff
(
    volVectorField& U,
    compressible::turbulenceModel& turbulence
);

tmp<fvVectorMatrix>
GenMatrix_U(
    const volScalarField& rho,
    volVectorField& U,
    const surfaceScalarField& phi,
    const volScalarField& p, 
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
    const surfaceScalarField& upwindWeights
);

void preProcess_Y(
    PtrList<volScalarField>& Y,
    surfaceScalarField& phiUc,
    volVectorField& sumYDiffError,
    // std::vector<volVectorField>& gradResult,
    dfChemistryModel<basicThermo>* chemistry,
    surfaceScalarField& upwindWeights,
    const volScalarField& alpha,
    const surfaceScalarField& phi
);

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
);

tmp<fvScalarMatrix>
GenMatrix_p(
    const volScalarField& rho,
    volScalarField& p,
    const surfaceScalarField& phiHbyA,
    const surfaceScalarField& rhorAUf,
    const volScalarField& phi
);

tmp<fvScalarMatrix>
GenMatrix_p(
    const volScalarField& rho,
    volScalarField& p,
    const surfaceScalarField& phi,
    const volScalarField& rAU,
    const volVectorField& U,
    const volVectorField& HbyA,
    const volScalarField& thermoPsi,
    // debug
    surfaceScalarField& rhorAUf,
    surfaceScalarField& phiHbyA
);

// thermo

// construct new variables
// thermo_rho
// thermo_

// correctThermo
void correctThermo
(
    const PtrList<volScalarField>& Y,
    volScalarField& T,
    volScalarField& ha,
    volScalarField& rho,
    volScalarField& psi,
    volScalarField& alpha,
    volScalarField& mu,
    volScalarField& p,
    dfChemistryModel<basicThermo>* chemistry
);

void check_fvmatrix_equal(const fvScalarMatrix& answer,const fvScalarMatrix& check, const word& name);
void check_fvmatrix_equal(const fvVectorMatrix& answer,const fvVectorMatrix& check, const word& name);

void check_field_equal(const Field<scalar>& answer, const Field<scalar>& check, const word& name);
void check_field_boundary_equal(const volScalarField& answer, const volScalarField& check, const word& name);
void check_field_boundary_equal(const volVectorField& answer, const volVectorField& check, const word& name);
void check_field_boundary_equal(const surfaceScalarField& answer, const surfaceScalarField& check, const word& name);
void check_field_error(const Field<scalar>& answer, const Field<scalar>& check, const word& name);

} // End namespace Foam

void init_const_coeff_ptr(std::string mechanism_file, Foam::PtrList<Foam::volScalarField>& Y);
extern Foam::label nSpecies;
extern Foam::label nCells;
extern Foam::label nFaces;
extern Foam::label nBoundarySurfaces;
extern Foam::label nBoundaryPatches;
extern Foam::label nProcessBoundarySurfaces;
extern Foam::label *surfacePerPatch;
extern int *neighbProcNo;

enum boundaryConditions{
    zeroGradient,
    fixedValue,
    coupled,
    empty,
    gradientEnergy,
    calculated,
    cyclic,
    processor,
    extrapolated,
    fixedEnergy,
    processorCyclic
};

// ************************************************************************* //
