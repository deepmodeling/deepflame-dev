/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2018 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
2019-10-21  Jeff Heylmun:   Moved from rhoCentralFoam to runtime selectable
                            method.
-------------------------------------------------------------------------------License
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

#include "AUSMDV.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace fluxSchemes
{
    defineTypeNameAndDebug(AUSMDV, 0);
    addToRunTimeSelectionTable(fluxScheme, AUSMDV, dictionary);
}
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::fluxSchemes::AUSMDV::AUSMDV
(
    const fvMesh& mesh
)
:
    fluxScheme(mesh)
{}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::fluxSchemes::AUSMDV::~AUSMDV()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::fluxSchemes::AUSMDV::clear()
{
    fluxScheme::clear();
}


void Foam::fluxSchemes::AUSMDV::createSavedFields()
{
    fluxScheme::createSavedFields();
}

void Foam::fluxSchemes::AUSMDV::calculateFluxes
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
    scalar  norP = 1.0;
    vector normal(Sf/magSf);
    const scalar vMesh(meshPhi(facei, patchi));

    scalar UvOwn((UOwn & normal) - vMesh/magSf);
    scalar UvNei((UNei & normal) - vMesh/magSf);

    // Compute split velocity
    scalar alphaOwn(2*(pOwn/rhoOwn)/(pOwn/rhoOwn + pNei/rhoNei));
    scalar alphaNei(2 - alphaOwn);

    scalar cm(max(cOwn , cNei));

    scalar uPlus(
      neg0(mag(UvOwn/cm) - 1)*(alphaOwn*(sqr(UvOwn + cm)/(4*cm) - 0.5*(UvOwn + mag(UvOwn))))
       + 0.5*(UvOwn + mag(UvOwn))
    );
    scalar uMinus(
      neg0(mag(UvNei/cm) - 1)*(alphaNei*(-sqr(UvNei - cm)/(4*cm) - 0.5*(UvNei - mag(UvNei))))
       + 0.5*(UvNei - mag(UvNei))
    );

    // scalar U12 = (alphaOwn*UOwn + alphaNei*UNei)/2.0;

    scalar pPlus(
       neg0(mag(UvOwn/cm) - 1)*pOwn*sqr(UvOwn/cm + 1.0)*(2.0 - UvOwn/cm)/4.0
       + pos(mag(UvOwn/cm) - 1)*pOwn*0.5*(1 + sign(UvOwn))    //pos(mag(UvNei))
    );
    scalar pMinus(
       neg0(mag(UvNei/cm) - 1)*pNei*sqr(UvNei/cm - 1.0)*(2.0 + UvNei/cm)/4.0
       + pos(mag(UvNei/cm) - 1)*pNei*0.5*(1 - sign(UvNei))    //max(UvNei,minU)
    );

    scalar P12(pPlus + pMinus);
    scalar s(0.5*min(norP , 10.0*mag(pNei - pOwn)/min(pOwn,pNei)));

    scalar caseA(neg(UvOwn - cOwn)*pos(UvNei - cNei));
    scalar caseB(neg(UvOwn + cOwn)*pos(UvNei + cNei));

    rhoPhi =   (uPlus*rhoOwn + uMinus*rhoNei)*magSf
             - (1 - caseA*caseB)*(caseA*0.125*(UvNei - cNei - UvOwn + cOwn)*(rhoNei - rhoOwn)*magSf
                    + (1 - caseA)*caseB*0.125*(UvNei + cNei - UvOwn - cOwn)*(rhoNei - rhoOwn)*magSf);

    forAll(rhoPhiYi,i)
    {
        rhoPhiYi[i] =  (uPlus*rhoYiOwn[i] + uMinus*rhoYiNei[i])*magSf
                     - (1 - caseA*caseB)*(caseA*0.125*(UvNei - cNei - UvOwn + cOwn)*(rhoYiNei[i] - rhoYiOwn[i])*magSf
                            + (1 - caseA)*caseB*0.125*(UvNei + cNei - UvOwn - cOwn)*(rhoYiNei[i] - rhoYiOwn[i])*magSf);
    }

    vector AUSMV((uPlus*rhoOwn*UOwn + uMinus*rhoNei*UNei)*magSf);
    vector AUSMD(0.5*(rhoPhi*(UOwn+UNei) - mag(rhoPhi)*(UNei-UOwn)));

    rhoUPhi =   (0.5 + s)*AUSMV + (0.5 - s)*AUSMD + P12*normal*magSf
              - (1 - caseA*caseB)*(caseA*0.125*(UvNei - cNei - UvOwn + cOwn)*(rhoNei*UNei - rhoOwn*UOwn)*magSf
                     + (1 - caseA)*caseB*0.125*(UvNei + cNei - UvOwn - cOwn)*(rhoNei*UNei - rhoOwn*UOwn)*magSf);

    scalar rhoEOwn = rhoOwn*(eOwn + 0.5*magSqr(UOwn));
    scalar rhoENei = rhoNei*(eNei + 0.5*magSqr(UNei));
    
    scalar hOwn((rhoEOwn+pOwn)/rhoOwn);
    scalar hNei((rhoENei+pNei)/rhoNei);

    rhoEPhi =    0.5*(rhoPhi*(hOwn + hNei) - mag(rhoPhi)*(hNei-hOwn)) + vMesh*P12
               - (1 - caseA*caseB)*(caseA*0.125*(UvNei - cNei - UvOwn + cOwn)*(rhoNei*hNei - rhoOwn*hOwn)*magSf
                      + (1 - caseA)*caseB*0.125*(UvNei + cNei - UvOwn - cOwn)*(rhoNei*hNei - rhoOwn*hOwn)*magSf);
}

// ************************************************************************* //