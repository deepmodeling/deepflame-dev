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

Function
    AUSMDVFlux

Description
    A function for solving convective flux based on AUSMDV, which mixes the AUSMD and AUSMV scheme

Author
    Daoping Zhang (zdp19980509@163.com)

\*---------------------------------------------------------------------------*/

void AUSMDVFlux(
    const surfaceScalarField& meshPhi,
    const surfaceScalarField& magSf,
    const surfaceVectorField& normal,
    const surfaceScalarField& rhoOwn,
    const surfaceScalarField& rhoNei,
    const surfaceVectorField& UOwn,
    const surfaceVectorField& UNei,
    const surfaceScalarField& HOwn,
    const surfaceScalarField& HNei,
    const surfaceScalarField& pOwn,
    const surfaceScalarField& pNei,
    const surfaceScalarField& gammaOwn,
    const surfaceScalarField& gammaNei,
    const surfaceScalarField& aOwn,
    const surfaceScalarField& aNei,
    surfaceScalarField& massFlux,
    surfaceVectorField& momentumFlux,
    surfaceScalarField& energyFlux
)
{
   
    dimensionedScalar norP("norP", dimless, 1.0);
    dimensionedScalar minU("minU", dimVelocity, SMALL);
    dimensionedScalar zeroU("minU", dimVelocity, 0.0);

    surfaceScalarField UvOwn((UOwn & normal) - meshPhi/magSf);

    surfaceScalarField UvNei((UNei & normal) - meshPhi/magSf);

    // Compute split velocity
    surfaceScalarField alphaOwn(2*(pOwn/rhoOwn)/(pOwn/rhoOwn + pNei/rhoNei));
    surfaceScalarField alphaNei(2*(1 - 0.5*alphaOwn));

    surfaceScalarField cm(max(aOwn , aNei));
 
    surfaceScalarField uPlus(
      neg0(mag(UvOwn/cm) - 1)*(alphaOwn*(sqr(UvOwn + cm)/(4*cm) - 0.5*(UvOwn + mag(UvOwn)))
       + 0.5*(UvOwn + mag(UvOwn)))
       + Foam::pos(mag(UvOwn/cm) - 1)*0.5*(UvOwn + mag(UvOwn))
    );
    surfaceScalarField uMinus(
      neg0(mag(UvNei/cm) - 1)*(alphaNei*(-sqr(UvNei - cm)/(4*cm) - 0.5*(UvNei - mag(UvNei)))
       + 0.5*(UvNei - mag(UvNei)))
       + Foam::pos(mag(UvNei/cm) - 1)*0.5*(UvNei - mag(UvNei))
    );
 
    surfaceScalarField pPlus(
       neg0(mag(UvOwn/cm) - 1)*pOwn*sqr(UvOwn/cm + 1.0)*(2.0 - UvOwn/cm)/4.0
       + Foam::pos(mag(UvOwn/cm) - 1)*pOwn*0.5*(1 + Foam::sign(UvOwn))//pos(mag(UvNei))
    );
    surfaceScalarField pMinus(
       neg0(mag(UvNei/cm) - 1)*pNei*sqr(UvNei/cm - 1.0)*(2.0 + UvNei/cm)/4.0
       + Foam::pos(mag(UvNei/cm) - 1)*pNei*0.5*(1 - Foam::sign(UvNei))//max(UvNei,minU)
    );

    surfaceScalarField P12(pPlus + pMinus);
    surfaceScalarField s(0.5*min(norP , 10.0*mag(pNei - pOwn)/min(pOwn,pNei)));

    surfaceScalarField caseA(Foam::neg(UvOwn - aOwn)*Foam::pos(UvNei - aNei));
    surfaceScalarField caseB(Foam::neg(UvOwn + aOwn)*Foam::pos(UvNei + aNei));

    massFlux = (uPlus*rhoOwn + uMinus*rhoNei)*magSf
               -(1-caseA*caseB)*(caseA*0.125*(UvNei - aNei - UvOwn + aOwn)*(rhoNei - rhoOwn)*magSf
               + (1-caseA)*caseB*0.125*(UvNei + aNei - UvOwn - aOwn)*(rhoNei - rhoOwn)*magSf);

    surfaceVectorField AUSMV((uPlus*rhoOwn*UOwn + uMinus*rhoNei*UNei)*magSf);
    surfaceVectorField AUSMD(0.5*(massFlux*(UOwn+UNei) - mag(massFlux)*(UNei-UOwn)));

    momentumFlux = (0.5+s)*AUSMV + (0.5-s)*AUSMD + P12*normal*magSf
                  -(1-caseA*caseB)*(caseA*0.125*(UvNei - aNei - UvOwn + aOwn)*(rhoNei*UNei - rhoOwn*UOwn)*magSf
                  + (1-caseA)*caseB*0.125*(UvNei + aNei - UvOwn - aOwn)*(rhoNei*UNei - rhoOwn*UOwn)*magSf);

    energyFlux = 0.5*(massFlux*(HOwn+HNei) - mag(massFlux)*(HNei-HOwn)) + meshPhi*P12
                  -(1-caseA*caseB)*(caseA*0.125*(UvNei - aNei - UvOwn + aOwn)*(rhoNei*HNei - rhoOwn*HOwn)*magSf
                  + (1-caseA)*caseB*0.125*(UvNei + aNei - UvOwn - aOwn)*(rhoNei*HNei - rhoOwn*HOwn)*magSf);

    
}