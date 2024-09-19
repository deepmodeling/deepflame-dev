/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2019 Synthetik Applied Technologies
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is derivative work of OpenFOAM.

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

#include "fluxScheme.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(fluxScheme, 0);
    defineRunTimeSelectionTable(fluxScheme, dictionary);
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::fluxScheme::fluxScheme(const fvMesh& mesh)
:
    regIOobject
    (
        IOobject
        (
            "fluxScheme",
            mesh.time().timeName(),
            mesh
        )
    ),
    mesh_(mesh)
{}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::fluxScheme::~fluxScheme()
{}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::fluxScheme::clear()
{
    own_.clear();
    nei_.clear();
    rhoOwn_.clear();
    rhoNei_.clear();
}

void Foam::fluxScheme::createSavedFields()
{
    if (own_.valid())
    {
        return;
    }
    own_ = tmp<surfaceScalarField>
    (
        new surfaceScalarField
        (
            IOobject
            (
                "fluxScheme::own",
                mesh_.time().timeName(),
                mesh_
            ),
            mesh_,
            dimensionedScalar("1", dimless, 1.0)
        )
    );
    nei_ = tmp<surfaceScalarField>
    (
        new surfaceScalarField
        (
            IOobject
            (
                "fluxScheme::nei",
                mesh_.time().timeName(),
                mesh_
            ),
            mesh_,
            dimensionedScalar("-1", dimless, -1.0)
        )
    );
}

void Foam::fluxScheme::update
(
    const volScalarField& rho,
    const PtrList<volScalarField>& rhoYi,
    const scalar& nspecies,
    const volVectorField& U,
    const volScalarField& e,
    const volScalarField& p,
    const volScalarField& c,
    surfaceScalarField& phi,
    surfaceScalarField& rhoPhi,
    PtrList<surfaceScalarField>& rhoPhiYi,
    surfaceVectorField& rhoUPhi,
    surfaceScalarField& rhoEPhi
)
{
    createSavedFields();

    rhoOwn_ = fvc::interpolate(rho, own_(), scheme("rho"));
    rhoNei_ = fvc::interpolate(rho, nei_(), scheme("rho"));

    PtrList<surfaceScalarField> rhoYiOwn(nspecies);
    PtrList<surfaceScalarField> rhoYiNei(nspecies);

    forAll(rhoYiOwn,i)
    {
        rhoYiOwn.set
        (
            i,
            new surfaceScalarField
            (
                IOobject
                (
                    "rhoYiOwn" + rhoYi[i].name(),
                    mesh_.time().timeName(),
                    mesh_,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE
                ),
                fvc::interpolate(rhoYi[i], own_(), scheme("Yi"))
            )
        );
    }

    forAll(rhoYiNei,i)
    {
        rhoYiNei.set
        (
            i,
            new surfaceScalarField
            (
                IOobject
                (
                    "rhoYiNei" + rhoYi[i].name(),
                    mesh_.time().timeName(),
                    mesh_,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE
                ),
                fvc::interpolate(rhoYi[i], nei_(), scheme("Yi"))
            )
        );
    }

    surfaceVectorField UOwn(fvc::interpolate(U, own_(), scheme("U")));
    surfaceVectorField UNei(fvc::interpolate(U, nei_(), scheme("U")));

    surfaceScalarField pOwn(fvc::interpolate(p, own_(), scheme("p")));
    surfaceScalarField pNei(fvc::interpolate(p, nei_(), scheme("p")));

    surfaceScalarField cOwn(fvc::interpolate(c, own_(), scheme("c")));
    surfaceScalarField cNei(fvc::interpolate(c, nei_(), scheme("c")));

    surfaceScalarField eOwn(fvc::interpolate(e, own_(), scheme("T")));
    surfaceScalarField eNei(fvc::interpolate(e, nei_(), scheme("T")));

    preUpdate(p);
    forAll(UOwn, facei)
    {
        scalarList rhoYiOwn_face(nspecies);
        scalarList rhoYiNei_face(nspecies);
        scalarList rhoPhiYi_face(nspecies);

        forAll(rhoYiOwn_face,i)
        {
            rhoYiOwn_face[i] = rhoYiOwn[i][facei];
        }

        forAll(rhoYiNei_face,i)
        {
            rhoYiNei_face[i] = rhoYiNei[i][facei];
        }

        calculateFluxes
        (
            rhoOwn_()[facei], rhoNei_()[facei],
            rhoYiOwn_face, rhoYiNei_face,
            UOwn[facei], UNei[facei],
            eOwn[facei], eNei[facei],
            pOwn[facei], pNei[facei],
            cOwn[facei], cNei[facei],
            mesh_.Sf()[facei],
            phi[facei],
            rhoPhi[facei],
            rhoPhiYi_face,
            rhoUPhi[facei],
            rhoEPhi[facei],
            facei
        );

        forAll(rhoPhiYi_face,i)
        {
            rhoPhiYi[i][facei] = rhoPhiYi_face[i];
        }
    }

    forAll(U.boundaryField(), patchi)
    {
        forAll(U.boundaryField()[patchi], facei)
        {

            scalarList rhoYiOwn_bf(nspecies);
            scalarList rhoYiNei_bf(nspecies);
            scalarList rhoPhiYi_bf(nspecies);

            forAll(rhoYiOwn_bf,i)
            {
                rhoYiOwn_bf[i] = rhoYiOwn[i].boundaryField()[patchi][facei];
            }

            forAll(rhoYiNei_bf,i)
            {
                rhoYiNei_bf[i] = rhoYiNei[i].boundaryField()[patchi][facei];
            }

            calculateFluxes
            (
                rhoOwn_().boundaryField()[patchi][facei],
                rhoNei_().boundaryField()[patchi][facei],
                rhoYiOwn_bf,
                rhoYiNei_bf,
                UOwn.boundaryField()[patchi][facei],
                UNei.boundaryField()[patchi][facei],
                eOwn.boundaryField()[patchi][facei],
                eNei.boundaryField()[patchi][facei],
                pOwn.boundaryField()[patchi][facei],
                pNei.boundaryField()[patchi][facei],
                cOwn.boundaryField()[patchi][facei],
                cNei.boundaryField()[patchi][facei],
                mesh_.Sf().boundaryField()[patchi][facei],
                phi.boundaryFieldRef()[patchi][facei],
                rhoPhi.boundaryFieldRef()[patchi][facei],
                rhoPhiYi_bf,
                rhoUPhi.boundaryFieldRef()[patchi][facei],
                rhoEPhi.boundaryFieldRef()[patchi][facei],
                facei, patchi
            );

            forAll(rhoPhiYi_bf,i)
            {
                rhoPhiYi[i].boundaryFieldRef()[patchi][facei] = rhoPhiYi_bf[i];
            }
        }
    }
    postUpdate();
}

bool Foam::fluxScheme::writeData(Ostream& os) const
{
    return os.good();
}

// ************************************************************************* //