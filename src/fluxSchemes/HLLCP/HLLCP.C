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

#include "HLLCP.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace fluxSchemes
{
    defineTypeNameAndDebug(HLLCP, 0);
    addToRunTimeSelectionTable(fluxScheme, HLLCP, dictionary);
}
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::fluxSchemes::HLLCP::HLLCP
(
    const fvMesh& mesh
)
:
    fluxScheme(mesh)
{}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::fluxSchemes::HLLCP::~HLLCP()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::fluxSchemes::HLLCP::clear()
{
    fluxScheme::clear();
}

void Foam::fluxSchemes::HLLCP::createSavedFields()
{
    fluxScheme::createSavedFields();
}

void Foam::fluxSchemes::HLLCP::preUpdate(const volScalarField& p)
{
    volScalarField fCells
    (
        IOobject
        (
            "fCells",
            p.time().timeName(),
            mesh_
        ),
        mesh_,
        dimensionedScalar("0", dimless, 1.0)
    );

    const labelListList& cellCells = mesh_.cellCells();
    forAll(fCells, celli)
    {
        forAll(cellCells[celli], cellj)
        {
            fCells[celli] =
                min
                (
                    fCells[celli],
                    min
                    (
                        p[celli]/p[cellCells[celli][cellj]],
                        p[cellCells[celli][cellj]]/p[celli]
                    )
                );
        }
    }
    f_ = fvc::interpolate(pow3(fCells));
}


void Foam::fluxSchemes::HLLCP::postUpdate()
{
    f_.clear();
}


void Foam::fluxSchemes::HLLCP::calculateFluxes
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

    scalar MaOwn(UvOwn/cOwn);
    scalar MaNei(UvNei/cNei);

    scalar wOwn(sqrt(rhoOwn)/(sqrt(rhoOwn) + sqrt(rhoNei)));
    scalar wNei(sqrt(rhoNei)/(sqrt(rhoOwn) + sqrt(rhoNei)));
    scalar MaTilde(wOwn*MaOwn + wNei*MaNei);
    vector UTilde(wOwn*UOwn + wNei*UNei);
    scalar UvTilde(wOwn*UvOwn + wNei*UvNei);
    scalar cTilde(wOwn*cOwn + wNei*cNei);

    scalar SOwn(min(UvOwn - cOwn, UvTilde - cTilde));
    scalar SNei(max(UvNei + cNei, UvTilde + cTilde));

    scalar aOwn(rhoOwn*(SOwn - UvOwn));
    scalar aNei(rhoNei*(SNei - UvNei));

    scalar SStar((aNei*UvNei - aOwn*UvOwn + pOwn - pNei)/(aNei - aOwn));
    scalar pStar
    (
        (
            aNei*pOwn - aOwn*pNei - aOwn*aNei*(UvOwn - UvNei)
        )/(aNei - aOwn)
    );

    scalar theta(min(max(mag(MaOwn), mag(MaNei)), 1.0));
    scalar pAvg(0.5*(pOwn + pNei));
    scalar pStarStar(pStar*theta + (1.0 - theta)*pAvg);

    scalar f(f_()[facei]);
    scalar pStarStarStar(f*pStarStar + (1.0 - f)*pStar);

    scalar phip =
        (f - 1.0)
       *SOwn*SNei/(SNei - SOwn)
       /(1.0 + mag(MaTilde))
       *(pNei - pOwn)/sqr(cTilde);

    // this->save(facei, patchi, SOwn, SOwn_);
    // this->save(facei, patchi, SNei, SNei_);
    // this->save(facei, patchi, SStar, SStar_);
    // this->save(facei, patchi, pStar, pStar_);
    // this->save(facei, patchi, phip, phip_);
    // this->save(facei, patchi, UTilde, UTilde_);
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
        rhoUPhi = rhoUPhiOwn;
        rhoEPhi = rhoEPhiOwn;
        p = pOwn;
    }
    else if (SStar > 0)
    {
        const scalar dS = SOwn - SStar;

        // this->save
        // (
        //     facei,
        //     patchi,
        //     (SOwn*rhoUOwn - rhoUPhiOwn + pStarStarStar*normal)
        //    /(rhoOwn*(SOwn - UvOwn)),
        //     Uf_
        // );
        phi = SStar*(SOwn - UvOwn)/dS;

        rhoPhi = phi*rhoOwn + phip;
        forAll(rhoPhiYi,i)
        {
            rhoPhiYi[i] = (phi*rhoOwn+ phip)*(rhoYiOwn[i]/rhoOwn);
        }
        rhoUPhi =
            (
                SStar*(SOwn*rhoUOwn - rhoUPhiOwn) + SOwn*pStarStarStar*normal
            )/dS
          + phip*UTilde;
        rhoEPhi =
            SStar*(SOwn*rhoEOwn - rhoEPhiOwn + SOwn*pStar)/dS
          + 0.5*phip*magSqr(UTilde);
        p = pStar;
    }
    else if (SNei > 0)
    {
        const scalar dS = SNei - SStar;

        // this->save
        // (
        //     facei,
        //     patchi,
        //     (SNei*rhoUNei - rhoUPhiNei + pStarStarStar*normal)
        //    /(rhoNei*(SNei - UvNei)),
        //     Uf_
        // );
        phi = SStar*(SNei - UvNei)/dS;
        rhoPhi = phi*rhoNei + phip;
        forAll(rhoPhiYi,i)
        {
            rhoPhiYi[i] = (phi*rhoNei+ phip)*(rhoYiNei[i]/rhoNei);
        }
        rhoUPhi =
            (
                SStar*(SNei*rhoUNei - rhoUPhiNei) + SNei*pStarStarStar*normal
            )/dS
          + phip*UTilde;
        rhoEPhi =
            SStar*(SNei*rhoENei - rhoEPhiNei + SNei*pStar)/dS
          + 0.5*phip*magSqr(UTilde);
        p = pStar;
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
        rhoUPhi = rhoUPhiNei;
        rhoEPhi = rhoEPhiNei;
        p = pNei;
    }

    phi *= magSf;
    rhoPhi *= magSf;
    forAll(rhoPhiYi,i)
    {
        rhoPhiYi[i] *= magSf;
    }
    rhoUPhi *= magSf;
    rhoEPhi *= magSf;
    rhoEPhi += meshPhi(facei, patchi)*p;
}

// ************************************************************************* //