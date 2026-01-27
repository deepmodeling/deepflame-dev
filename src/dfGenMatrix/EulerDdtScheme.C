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

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// namespace fv
// {

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

template<class Type>
tmp<fvMatrix<Type>>
EulerDdtSchemeFvmDdt
(
    const volScalarField& rho,
    const GeometricField<Type, fvPatchField, volMesh>& vf
)
{
    Info << "EulerDdtSchemeFvmDdt start" << endl;

    const fvMesh& mesh = vf.mesh();

    tmp<fvMatrix<Type>> tfvm
    (
        new fvMatrix<Type>
        (
            vf,
            rho.dimensions()*vf.dimensions()*dimVol/dimTime
        )
    );
    fvMatrix<Type>& fvm = tfvm.ref();

    scalar rDeltaT = 1.0/mesh.time().deltaTValue();

    fvm.diag() = rDeltaT*rho.primitiveField()*mesh.Vsc();

    if (mesh.moving())
    {
        fvm.source() = rDeltaT
            *rho.oldTime().primitiveField()
            *vf.oldTime().primitiveField()*mesh.Vsc0();
    }
    else
    {
        fvm.source() = rDeltaT
            *rho.oldTime().primitiveField()
            *vf.oldTime().primitiveField()*mesh.Vsc();
    }

    Info << "EulerDdtSchemeFvmDdt end" << endl;
    return tfvm;
}

template<class Type>
tmp<GeometricField<Type, fvPatchField, volMesh>>
EulerDdtSchemeFvcDdt
(
    const volScalarField& rho,
    const GeometricField<Type, fvPatchField, volMesh>& vf
)
{
    Info << "EulerDdtSchemeFvmDdt start" << endl;

    const fvMesh& mesh = vf.mesh();

    dimensionedScalar rDeltaT = 1.0/mesh.time().deltaT();

    IOobject ddtIOobject
    (
        "ddt("+rho.name()+','+vf.name()+')',
        mesh.time().timeName(),
        mesh
    );

    if (mesh.moving())
    {
        return tmp<GeometricField<Type, fvPatchField, volMesh>>
        (
            new GeometricField<Type, fvPatchField, volMesh>
            (
                ddtIOobject,
                rDeltaT*
                (
                    rho()*vf()
                  - rho.oldTime()()
                   *vf.oldTime()()*mesh.Vsc0()/mesh.Vsc()
                ),
                rDeltaT.value()*
                (
                    rho.boundaryField()*vf.boundaryField()
                  - rho.oldTime().boundaryField()
                   *vf.oldTime().boundaryField()
                )
            )
        );
    }
    else
    {
        return tmp<GeometricField<Type, fvPatchField, volMesh>>
        (
            new GeometricField<Type, fvPatchField, volMesh>
            (
                ddtIOobject,
                rDeltaT*(rho*vf - rho.oldTime()*vf.oldTime())
            )
        );
    }
}


tmp<surfaceScalarField>
EulerDdtSchemeFvcDdtCorr
(
    const volScalarField& rho,
    const volVectorField& U,
    const surfaceScalarField& phi,
    const autoPtr<surfaceVectorField>& Uf
)
{
    Info << "EulerDdtSchemeFvcDdtCorr start" << endl;

    const fvMesh& mesh = U.mesh();

    dimensionedScalar rDeltaT = 1.0/mesh.time().deltaT();

    GeometricField<vector, fvPatchField, volMesh> rhoU0
    (
        rho.oldTime() * U.oldTime()
    );

    surfaceScalarField phiCorr
    (
        phi.oldTime() - fvc::dotInterpolate(mesh.Sf(), rhoU0)
    );

    return tmp<surfaceScalarField>
    (
        new surfaceScalarField
        (
            IOobject
            (
                "ddtCorr("
                + rho.name() + ',' + U.name() + ',' + phi.name() + ')',
                mesh.time().timeName(),
                mesh
            ),
            EulerDdtSchemeFvcDdtPhiCoeff
            (
                rhoU0,
                phi.oldTime(),
                phiCorr,
                rho.oldTime()
            )*rDeltaT*phiCorr
        )
    );

}

tmp<surfaceScalarField>
EulerDdtSchemeFvcDdtPhiCoeff
(
    const volVectorField& U,
    const surfaceScalarField& phi,
    const surfaceScalarField& phiCorr,
    const volScalarField& rho
)
{
    const fvMesh& mesh = U.mesh();
    tmp<surfaceScalarField> tddtCouplingCoeff = scalar(1) - min(mag(phiCorr)/(mag(phi) + dimensionedScalar("small", phi.dimensions(), SMALL)),scalar(1));

    surfaceScalarField& ddtCouplingCoeff = tddtCouplingCoeff.ref();

    surfaceScalarField::Boundary& ccbf = ddtCouplingCoeff.boundaryFieldRef();

    forAll(U.boundaryField(), patchi)
    {
        if
        ( U.boundaryField()[patchi].fixesValue()
         || isA<cyclicAMIFvPatch>(mesh.boundary()[patchi])
        )
        {
            ccbf[patchi] = 0.0;
        }
    }

    return tddtCouplingCoeff;
}

template<class Type>
tmp<fvMatrix<Type>>
EulerDdtSchemeFvmDdt
(
    const GeometricField<Type, fvPatchField, volMesh>& vf
)
{
    const fvMesh& mesh = vf.mesh();

    tmp<fvMatrix<Type>> tfvm
    (
        new fvMatrix<Type>
        (
            vf,
            vf.dimensions()*dimVol/dimTime
        )
    );

    fvMatrix<Type>& fvm = tfvm.ref();

    scalar rDeltaT = 1.0/mesh.time().deltaTValue();

    fvm.diag() = rDeltaT*mesh.Vsc();

    if (mesh.moving())
    {
        fvm.source() = rDeltaT*vf.oldTime().primitiveField()*mesh.Vsc0();
    }
    else
    {
        fvm.source() = rDeltaT*vf.oldTime().primitiveField()*mesh.Vsc();
    }

    return tfvm;
}

template<class Type>
tmp<GeometricField<Type, fvPatchField, volMesh>>
EulerDdtSchemeFvcDdt
(
    const GeometricField<Type, fvPatchField, volMesh>& vf
)
{
    const fvMesh& mesh = vf.mesh();

    dimensionedScalar rDeltaT = 1.0/mesh.time().deltaT();

    IOobject ddtIOobject
    (
        "ddt("+vf.name()+')',
        mesh.time().timeName(),
        mesh
    );

    return tmp<GeometricField<Type, fvPatchField, volMesh>>
    (
        new GeometricField<Type, fvPatchField, volMesh>
        (
            ddtIOobject,
            rDeltaT*(vf - vf.oldTime())
        )
    );
}

template
tmp<fvMatrix<scalar>>
EulerDdtSchemeFvmDdt<scalar>
(
    const volScalarField& rho,
    const GeometricField<scalar, fvPatchField, volMesh>& vf
);

template
tmp<fvMatrix<vector>>
EulerDdtSchemeFvmDdt<vector>
(
    const volScalarField& rho,
    const GeometricField<vector, fvPatchField, volMesh>& vf
);

template
tmp<GeometricField<scalar, fvPatchField, volMesh>>
EulerDdtSchemeFvcDdt<scalar>
(
    const volScalarField& rho,
    const GeometricField<scalar, fvPatchField, volMesh>& vf
);

template
tmp<fvMatrix<scalar>>
EulerDdtSchemeFvmDdt
(
    const GeometricField<scalar, fvPatchField, volMesh>& vf
);

template
tmp<GeometricField<scalar, fvPatchField, volMesh>>
EulerDdtSchemeFvcDdt
(
    const GeometricField<scalar, fvPatchField, volMesh>& vf
);

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// } // End namespace fv

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
