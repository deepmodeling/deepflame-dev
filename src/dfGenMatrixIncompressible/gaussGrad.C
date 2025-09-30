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

#include "gaussGrad.H"
#include "extrapolatedCalculatedFvPatchField.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{



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
)
{
    return gaussGradSchemeGrad(vsf, "grad(" + vsf.name() + ')');
}

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
    const GeometricField<Type, fvPatchField, volMesh>& vsf,
    const word& name
)
{
    const fvMesh& mesh = vsf.mesh();

    typedef typename outerProduct<vector, Type>::type GradType;
    typedef GeometricField<GradType, fvPatchField, volMesh> GradFieldType;

    if (!mesh.changing() && mesh.cache(name))
    {
        if (!mesh.objectRegistry::template foundObject<GradFieldType>(name))
        {
            solution::cachePrintMessage("Calculating and caching", name, vsf);
            tmp<GradFieldType> tgGrad = gaussGradCalcGrad(vsf, name);
            regIOobject::store(tgGrad.ptr());
        }

        solution::cachePrintMessage("Retrieving", name, vsf);
        GradFieldType& gGrad =
            mesh.objectRegistry::template lookupObjectRef<GradFieldType>
            (
                name
            );

        if (gGrad.upToDate(vsf))
        {
            return gGrad;
        }
        else
        {
            solution::cachePrintMessage("Deleting", name, vsf);
            gGrad.release();
            delete &gGrad;

            solution::cachePrintMessage("Recalculating", name, vsf);
            tmp<GradFieldType> tgGrad = gaussGradCalcGrad(vsf, name);

            solution::cachePrintMessage("Storing", name, vsf);
            regIOobject::store(tgGrad.ptr());
            GradFieldType& gGrad =
                mesh.objectRegistry::template lookupObjectRef<GradFieldType>
                (
                    name
                );

            return gGrad;
        }
    }
    else
    {
        if (mesh.objectRegistry::template foundObject<GradFieldType>(name))
        {
            GradFieldType& gGrad =
                mesh.objectRegistry::template lookupObjectRef<GradFieldType>
                (
                    name
                );

            if (gGrad.ownedByRegistry())
            {
                solution::cachePrintMessage("Deleting", name, vsf);
                gGrad.release();
                delete &gGrad;
            }
        }

        solution::cachePrintMessage("Calculating", name, vsf);
        return gaussGradCalcGrad(vsf, name);
    }
}

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
gaussGradCalcGrad
(
    const GeometricField<Type, fvPatchField, volMesh>& vsf,
    const word& name
)
{
    const fvMesh& mesh = vsf.mesh();

    tmp<surfaceInterpolationScheme<Type>> tinterpScheme_ =
    tmp<surfaceInterpolationScheme<Type>>
    (
        new linear<Type>(mesh)
    );

    typedef typename outerProduct<vector, Type>::type GradType;

    tmp<GeometricField<Type, fvsPatchField, surfaceMesh>> tinterpolate = tinterpScheme_().interpolate(vsf);

    tmp<GeometricField<GradType, fvPatchField, volMesh>> tgGrad
    (
        gaussGradGradf(tinterpolate.ref(), name)
    );
    GeometricField<GradType, fvPatchField, volMesh>& gGrad = tgGrad.ref();

    gaussGradCorrectBoundaryConditions(vsf, gGrad);

    return tgGrad;
}

template<class Type>
void gaussGradCorrectBoundaryConditions
(
    const GeometricField<Type, fvPatchField, volMesh>& vsf,
    GeometricField
    <
        typename outerProduct<vector, Type>::type, fvPatchField, volMesh
    >& gGrad
)
{
    typename GeometricField
    <
        typename outerProduct<vector, Type>::type, fvPatchField, volMesh
    >::Boundary& gGradbf = gGrad.boundaryFieldRef();

    forAll(vsf.boundaryField(), patchi)
    {
        if (!vsf.boundaryField()[patchi].coupled())
        {
            const vectorField n
            (
                vsf.mesh().Sf().boundaryField()[patchi]
              / vsf.mesh().magSf().boundaryField()[patchi]
            );

            gGradbf[patchi] += n *
            (
                vsf.boundaryField()[patchi].snGrad()
              - (n & gGradbf[patchi])
            );
        }
     }
}

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
gaussGradGradf
(
    const GeometricField<Type, fvsPatchField, surfaceMesh>& ssf,
    const word& name
)
{
    typedef typename outerProduct<vector, Type>::type GradType;

    const fvMesh& mesh = ssf.mesh();

    tmp<GeometricField<GradType, fvPatchField, volMesh>> tgGrad
    (
        new GeometricField<GradType, fvPatchField, volMesh>
        (
            IOobject
            (
                name,
                ssf.instance(),
                mesh,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            mesh,
            dimensioned<GradType>
            (
                "0",
                ssf.dimensions()/dimLength,
                Zero
            ),
            extrapolatedCalculatedFvPatchField<GradType>::typeName
        )
    );
    GeometricField<GradType, fvPatchField, volMesh>& gGrad = tgGrad.ref();

    const labelUList& owner = mesh.owner();
    const labelUList& neighbour = mesh.neighbour();
    const vectorField& Sf = mesh.Sf();

    Field<GradType>& igGrad = gGrad;
    const Field<Type>& issf = ssf;

    forAll(owner, facei)
    {
        GradType Sfssf = Sf[facei]*issf[facei];

        igGrad[owner[facei]] += Sfssf;
        igGrad[neighbour[facei]] -= Sfssf;
    }

    forAll(mesh.boundary(), patchi)
    {
        const labelUList& pFaceCells =
            mesh.boundary()[patchi].faceCells();

        const vectorField& pSf = mesh.Sf().boundaryField()[patchi];

        const fvsPatchField<Type>& pssf = ssf.boundaryField()[patchi];

        forAll(mesh.boundary()[patchi], facei)
        {
            igGrad[pFaceCells[facei]] += pSf[facei]*pssf[facei];
        }
    }

    igGrad /= mesh.V();

    gGrad.correctBoundaryConditions();

    return tgGrad;
}


template
tmp
<
    GeometricField
    <
        typename outerProduct<vector, scalar>::type,
        fvPatchField,
        volMesh
    >
>
gaussGradSchemeGrad
(
    const GeometricField<scalar, fvPatchField, volMesh>& vsf
);


template
tmp
<
    GeometricField
    <
        typename outerProduct<vector, vector>::type,
        fvPatchField,
        volMesh
    >
>
gaussGradSchemeGrad
(
    const GeometricField<vector, fvPatchField, volMesh>& vsf
);

} // End namespace Foam

// ************************************************************************* //

// template<class Type>
// tmp
// <
//     GeometricField
//     <
//         typename outerProduct<vector,Type>::type, fvPatchField, volMesh
//     >
// >
// grad
// (
//     const GeometricField<Type, fvPatchField, volMesh>& vf
// )
// {
//     return fvc::grad(vf, "grad(" + vf.name() + ')');
// }

// template<class Type>
// tmp
// <
//     GeometricField
//     <
//         typename outerProduct<vector,Type>::type, fvPatchField, volMesh
//     >
// >
// grad
// (
//     const GeometricField<Type, fvPatchField, volMesh>& vf,
//     const word& name
// )
// {
//     return fv::gradScheme<Type>::New
//     (
//         vf.mesh(),
//         vf.mesh().gradScheme(name)
//     )().grad(vf, name);
// }