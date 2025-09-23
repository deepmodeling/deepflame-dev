/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2018 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "GenFvMatrix.H"
#include "fvcSurfaceIntegrate.H"
#include "fvMatrices.H"
#include "gaussConvectionScheme.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

template<class Type>
tmp<fvMatrix<Type>>
gaussConvectionSchemeFvmDiv
(
    const surfaceScalarField& faceFlux,
    const GeometricField<Type, fvPatchField, volMesh>& vf,
    const word& name
)
{
    Info << "gaussConvectionSchemeFvmDiv start" << endl;

    const fvMesh& mesh = vf.mesh();

    tmp<fv::convectionScheme<Type>> cs = fv::convectionScheme<Type>::New(mesh,faceFlux,mesh.divScheme(name));
    fv::gaussConvectionScheme<Type>& gcs = dynamic_cast<fv::gaussConvectionScheme<Type>&>(cs.ref());

    tmp<surfaceScalarField> tweights = gcs.interpScheme().weights(vf);
    const surfaceScalarField& weights = tweights();

    tmp<fvMatrix<Type>> tfvm
    (
        new fvMatrix<Type>
        (
            vf,
            faceFlux.dimensions()*vf.dimensions()
        )
    );
    fvMatrix<Type>& fvm = tfvm.ref();
    fvm.lower() = -weights.primitiveField()*faceFlux.primitiveField();
    fvm.upper() = fvm.lower() + faceFlux.primitiveField();
    fvm.negSumDiag();
    forAll(vf.boundaryField(), patchi)
    {
        const fvPatchField<Type>& psf = vf.boundaryField()[patchi];
        const fvsPatchScalarField& patchFlux = faceFlux.boundaryField()[patchi];
        const fvsPatchScalarField& pw = weights.boundaryField()[patchi];

        fvm.internalCoeffs()[patchi] = patchFlux*psf.valueInternalCoeffs(pw);
        fvm.boundaryCoeffs()[patchi] = -patchFlux*psf.valueBoundaryCoeffs(pw);
    }
    if (gcs.interpScheme().corrected())
    {
        fvm += fvc::surfaceIntegrate(faceFlux*gcs.interpScheme().correction(vf));
    }
    Info << "gaussConvectionSchemeFvmDiv end" << endl;
    return tfvm;
}

template<class Type>
tmp<fvMatrix<Type>>
gaussConvectionSchemeFvmDiv
(
    const surfaceScalarField& faceFlux,
    const GeometricField<Type, fvPatchField, volMesh>& vf
)
{
    word name("div("+faceFlux.name()+','+vf.name()+')');
    return gaussConvectionSchemeFvmDiv(faceFlux,vf,name);
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

template
tmp<fvMatrix<scalar>>
gaussConvectionSchemeFvmDiv
(
    const surfaceScalarField& faceFlux,
    const GeometricField<scalar, fvPatchField, volMesh>& vf
);

template
tmp<fvMatrix<vector>>
gaussConvectionSchemeFvmDiv
(
    const surfaceScalarField& faceFlux,
    const GeometricField<vector, fvPatchField, volMesh>& vf
);

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //