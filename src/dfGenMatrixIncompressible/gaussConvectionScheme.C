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
        fvm += fvcSurfaceIntegrate(faceFlux*gcs.interpScheme().correction(vf));
    }
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
    return gaussConvectionSchemeFvmDiv(faceFlux, vf, name);
}

template<class Type>
tmp<GeometricField<Type, fvPatchField, volMesh>>
gaussConvectionSchemeFvcDiv
(
    const surfaceScalarField& faceFlux,
    const GeometricField<Type, fvPatchField, volMesh>& vf
)
{
    word name("div("+faceFlux.name()+','+vf.name()+')');
    return gaussConvectionSchemeFvcDiv(faceFlux, vf, name);
}

template<class Type>
tmp<GeometricField<Type, fvPatchField, volMesh>>
gaussConvectionSchemeFvcDiv
(
    const surfaceScalarField& faceFlux,
    const GeometricField<Type, fvPatchField, volMesh>& vf,
    const word& name
)
{
    const fvMesh& mesh = vf.mesh();

    Istream& divIntScheme = mesh.divScheme(name);
    word divScheme(divIntScheme);
    
    tmp<surfaceInterpolationScheme<Type>> tinterpScheme_ = surfaceInterpolationScheme<Type>::New(mesh, faceFlux, divIntScheme);

    tmp<GeometricField<Type, fvPatchField, volMesh>> tConvection
    (
        fvcSurfaceIntegrate(gaussConvectionSchemeFlux(faceFlux, vf, tinterpScheme_))
    );

    tConvection.ref().rename
    (
        "convection(" + faceFlux.name() + ',' + vf.name() + ')'
    );

    return tConvection;
}

template<class Type>
tmp<GeometricField<Type, fvPatchField, volMesh>>
gaussConvectionSchemeFvcDiv
(
    const GeometricField<Type, fvsPatchField, surfaceMesh>& ssf
)
{
    return tmp<GeometricField<Type, fvPatchField, volMesh>>
    (
        new GeometricField<Type, fvPatchField, volMesh>
        (
            "div("+ssf.name()+')',
            fvcSurfaceIntegrate(ssf)
        )
    );
}

template<class Type>
tmp<GeometricField<Type, fvPatchField, volMesh>>
gaussConvectionSchemeFvcDiv
(
    const tmp<GeometricField<Type, fvsPatchField, surfaceMesh>>& tssf
)
{
    // const GeometricField<Type, fvsPatchField, surfaceMesh>& ssf = tssf();
    return tmp<GeometricField<Type, fvPatchField, volMesh>>
    (
        new GeometricField<Type, fvPatchField, volMesh>
        (
            "div("+tssf->name()+')',
            fvcSurfaceIntegrate(tssf)
        )
    );
}

template<class Type>
tmp<GeometricField<Type, fvsPatchField, surfaceMesh>>
gaussConvectionSchemeFlux
(
    const surfaceScalarField& faceFlux,
    const GeometricField<Type, fvPatchField, volMesh>& vf,
    tmp<surfaceInterpolationScheme<Type>> tinterpScheme
)
{
    return faceFlux*tinterpScheme().interpolate(vf);
}

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
)
{
    const fvMesh& mesh = vf.mesh();
    Istream& divIntScheme = mesh.divScheme("div("+vf.name()+')');
    word divScheme(divIntScheme);

    tmp<surfaceInterpolationScheme<Type>> tinterpScheme_ = surfaceInterpolationScheme<Type>::New(mesh, divIntScheme);

    tmp<GeometricField<typename innerProduct<vector, Type>::type, fvPatchField, volMesh>> tDiv
    (
        fvcSurfaceIntegrate
        (
            tinterpScheme_().dotInterpolate(mesh.Sf(), vf)
        )
    );
    
    return tDiv;
}

template<class Type>
tmp<GeometricField<Type, fvPatchField, volMesh>>
fvcSurfaceIntegrate
(
    const tmp<GeometricField<Type, fvsPatchField, surfaceMesh>>& tssf
)
{
    return fvcSurfaceIntegrate(tssf());
}


template<class Type>
tmp<GeometricField<Type, fvPatchField, volMesh>>
fvcSurfaceIntegrate
(
    const GeometricField<Type, fvsPatchField, surfaceMesh>& ssf
)
{
    const fvMesh& mesh = ssf.mesh();

    tmp<GeometricField<Type, fvPatchField, volMesh>> tvf
    (
        new GeometricField<Type, fvPatchField, volMesh>
        (
            IOobject
            (
                "surfaceIntegrate("+ssf.name()+')',
                ssf.instance(),
                mesh,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            mesh,
            dimensioned<Type>
            (
                "0",
                ssf.dimensions()/dimVol,
                Zero
            ),
            extrapolatedCalculatedFvPatchField<Type>::typeName
        )
    );
    GeometricField<Type, fvPatchField, volMesh>& vf = tvf.ref();
    Field<Type>& ivf = vf.primitiveFieldRef();

    const labelUList& owner = mesh.owner();
    const labelUList& neighbour = mesh.neighbour();

    const Field<Type>& issf = ssf;

    forAll(owner, facei)
    {
        ivf[owner[facei]] += issf[facei];
        ivf[neighbour[facei]] -= issf[facei];
    }

    forAll(mesh.boundary(), patchi)
    {
        const labelUList& pFaceCells =
            mesh.boundary()[patchi].faceCells();

        const fvsPatchField<Type>& pssf = ssf.boundaryField()[patchi];

        forAll(mesh.boundary()[patchi], facei)
        {
            ivf[pFaceCells[facei]] += pssf[facei];
        }
    }

    // ivf /= mesh.Vsc();
    auto meshVsc = mesh.Vsc();
    #pragma omp parallel for
    for(label c = 0; c < ivf.size(); ++c){
        ivf[c] /= meshVsc()[c];
    }

    vf.correctBoundaryConditions();

    return tvf;
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

template
tmp<GeometricField<scalar, fvPatchField, volMesh>>
gaussConvectionSchemeFvcDiv
(
    const surfaceScalarField& faceFlux,
    const GeometricField<scalar, fvPatchField, volMesh>& vf
);


template
tmp<GeometricField<scalar, fvPatchField, volMesh>>
gaussConvectionSchemeFvcDiv
(
    const GeometricField<scalar, fvsPatchField, surfaceMesh>& ssf
);

template
tmp
<
    GeometricField
    <
        typename innerProduct<vector, vector>::type, fvPatchField, volMesh
    >
>
gaussDivFvcdiv
(
    const GeometricField<vector, fvPatchField, volMesh>& vf
);

template
tmp<GeometricField<scalar, fvPatchField, volMesh>>
gaussConvectionSchemeFvcDiv
(
    const tmp<GeometricField<scalar, fvsPatchField, surfaceMesh>>& tssf
);

template
tmp<GeometricField<scalar, fvPatchField, volMesh>>
fvcSurfaceIntegrate
(
    const tmp<GeometricField<scalar, fvsPatchField, surfaceMesh>>& tssf
);
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //