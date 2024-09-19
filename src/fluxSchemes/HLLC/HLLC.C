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

#include "HLLC.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace fluxSchemes
{
    defineTypeNameAndDebug(HLLC, 0);
    addToRunTimeSelectionTable(fluxScheme, HLLC, dictionary);
}
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::fluxSchemes::HLLC::HLLC
(
    const fvMesh& mesh
)
:
    fluxScheme(mesh)
{}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::fluxSchemes::HLLC::~HLLC()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::fluxSchemes::HLLC::clear()
{
    fluxScheme::clear();
}

void Foam::fluxSchemes::HLLC::createSavedFields()
{
    fluxScheme::createSavedFields();
}

void Foam::fluxSchemes::HLLC::calculateFluxes
(
    const scalar& rhoOwn, const scalar& rhoNei,
    const scalarList& rhoYiOwn,
    const scalarList& rhoYiNei,
    const vector& UOwn, const vector& UNei,
    const scalar& eOwn, const scalar& eNei,
    const scalar& pOwn, const scalar& pNei,
    const scalar& cOwn, const scalar& cNei,
    const vector& Sf,
    scalar& phi,
    scalar& rhoPhi,
    scalarList& rhoPhiYi,
    vector& rhoUPhi,
    scalar& rhoEPhi,
    const label facei, const label patchi
)
{
    scalar magSf = mag(Sf);
    vector normal = Sf/magSf;

    scalar EOwn = eOwn + 0.5*magSqr(UOwn);
    scalar ENei = eNei + 0.5*magSqr(UNei);

    const scalar vMesh(meshPhi(facei, patchi)/magSf);
    scalar UvOwn((UOwn & normal) - vMesh);
    scalar UvNei((UNei & normal) - vMesh);

    scalar wOwn(sqrt(rhoOwn)/(sqrt(rhoOwn) + sqrt(rhoNei)));
    scalar wNei(1.0 - wOwn);

    scalar cTilde(cOwn*wOwn + cNei*wNei);
    scalar UvTilde(UvOwn*wOwn + UvNei*wNei);

    scalar SOwn(min(UvOwn - cOwn, UvTilde - cTilde));
    scalar SNei(max(UvNei + cNei, UvTilde + cTilde));

    scalar SStar
    (
        (
            pNei - pOwn
          + rhoOwn*UvOwn*(SOwn - UvOwn)
          - rhoNei*UvNei*(SNei - UvNei)
        )
       /(rhoOwn*(SOwn - UvOwn) - rhoNei*(SNei - UvNei))
    );

    scalar pStarOwn(pOwn + rhoOwn*(SOwn - UvOwn)*(SStar - UvOwn));
    scalar pStarNei(pNei + rhoNei*(SNei - UvNei)*(SStar - UvNei));

    // this->save(facei, patchi, SOwn, SOwn_);
    // this->save(facei, patchi, SNei, SNei_);
    // this->save(facei, patchi, SStar, SStar_);
    // this->save(facei, patchi, pStarOwn, pStarOwn_);
    // this->save(facei, patchi, pStarNei, pStarNei_);
    // this->save(facei, patchi, UvOwn, UvOwn_);
    // this->save(facei, patchi, UvNei, UvNei_);

    // Owner values
    const vector rhoUOwn = rhoOwn*UOwn;
    const scalar rhoEOwn = rhoOwn*EOwn;

    const vector rhoUPhiOwn = rhoUOwn*UvOwn + pOwn*normal;
    const scalar rhoEPhiOwn = (rhoEOwn + pOwn)*UvOwn;

    // Neighbour values
    const vector rhoUNei = rhoNei*UNei;
    const scalar rhoENei = rhoNei*ENei;

    const vector rhoUPhiNei = rhoUNei*UvNei + pNei*normal;
    const scalar rhoEPhiNei = (rhoENei + pNei)*UvNei;

    scalar p;
    if (SOwn > 0)
    {
        // this->save(facei, patchi, UOwn, Uf_);
        phi = UvOwn;
        rhoPhi = rhoOwn*UvOwn;
        forAll(rhoPhiYi,i)
        {
            rhoPhiYi[i] = rhoYiOwn[i]*UvOwn;
        }
        p = pOwn;
        rhoUPhi = rhoUPhiOwn;
        rhoEPhi = rhoEPhiOwn;
    }
    else if (SStar > 0)
    {
        const scalar dS = SOwn - SStar;

        // this->save
        // (
        //     facei,
        //     patchi,
        //     (SOwn*rhoUOwn - rhoUPhiOwn + pStarOwn*normal)
        //    /(rhoOwn*(SOwn - UvOwn)),
        //     Uf_
        // );
        phi = SStar;
        rhoPhi = phi*rhoOwn*(SOwn - UvOwn)/dS;
        forAll(rhoPhiYi,i)
        {
            rhoPhiYi[i] = phi*rhoYiOwn[i]*(SOwn - UvOwn)/dS;
        }
        p = 0.5*(pStarNei + pStarOwn);
        rhoUPhi =
            (SStar*(SOwn*rhoUOwn - rhoUPhiOwn) + SOwn*pStarOwn*normal)/dS;
        rhoEPhi = SStar*(SOwn*rhoEOwn - rhoEPhiOwn + SOwn*pStarOwn)/dS;
    }
    else if (SNei > 0)
    {
        const scalar dS = SNei - SStar;

        // this->save
        // (
        //     facei,
        //     patchi,
        //     (SNei*rhoUNei - rhoUPhiNei + pStarNei*normal)
        //    /(rhoNei*(SNei - UvNei)),
        //     Uf_
        // );
        phi = SStar;
        rhoPhi = phi*rhoNei*(SNei - UvNei)/dS;
        forAll(rhoPhiYi,i)
        {
            rhoPhiYi[i] = phi*rhoYiNei[i]*(SNei - UvNei)/dS;
        }
        p = 0.5*(pStarNei + pStarOwn);
        rhoUPhi =
            (SStar*(SNei*rhoUNei - rhoUPhiNei) + SNei*pStarNei*normal)/dS;
        rhoEPhi = SStar*(SNei*rhoENei - rhoEPhiNei + SNei*pStarNei)/dS;
    }
    else
    {
        // this->save(facei, patchi, UNei, Uf_);
        phi = UvNei;
        rhoPhi = rhoNei*UvNei;
        forAll(rhoPhiYi,i)
        {
            rhoPhiYi[i] = rhoYiNei[i]*UvNei;
        }
        p = pNei;
        rhoUPhi = rhoUPhiNei;
        rhoEPhi = rhoEPhiNei;
    }
    phi *= magSf;
    rhoPhi *= magSf;
    forAll(rhoPhiYi,i)
    {
        rhoPhiYi[i] *= magSf;
    }
    rhoUPhi *= magSf;
    rhoEPhi *= magSf;
    rhoEPhi += vMesh*magSf*p;
}

// ************************************************************************* //