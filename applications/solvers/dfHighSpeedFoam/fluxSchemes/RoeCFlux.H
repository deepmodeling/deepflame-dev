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
    RoeCFlux

Description
    A function for solving convective flux based on Roe's approximate solver

Author
    Daoping Zhang (zdp19980509@163.com)

\*---------------------------------------------------------------------------*/

void RoeCFlux(
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
    surfaceVectorField UO = UOwn - (meshPhi/magSf)*normal;
    surfaceVectorField UN = UNei - (meshPhi/magSf)*normal;
    surfaceScalarField wOwn(sqrt(rhoOwn)/(sqrt(rhoOwn) + sqrt(rhoNei)));
    surfaceScalarField wNei(1.0 - wOwn);
    surfaceScalarField wTilde(sqrt(rhoNei/rhoOwn));
    surfaceScalarField gm1Own( gammaOwn - 1.0 );
    surfaceScalarField gm1Nei( gammaNei - 1.0 );

    surfaceScalarField rhoTilde
    (
        "rhoTilde",
        sqrt
        (
            max
            (
                rhoOwn*rhoNei,
                dimensionedScalar("SMALL", sqr(dimDensity), SMALL)
            )
        )
    );

    surfaceScalarField gm1Tilde("gm1Tilde", (gm1Own + gm1Nei*wTilde)/(1.0+wTilde));

    surfaceVectorField UTilde
    (
        "UTilde",
        (UO + UN*wTilde)/(1.0+wTilde)
    );

    surfaceScalarField UvTilde(UTilde & normal);

    surfaceScalarField HTilde
    (
        "HTilde",
        (HOwn + HNei*wTilde)/(1.0+wTilde)
    );

    surfaceScalarField aTilde
    (
        "aTilde",
        (aOwn + aNei*wTilde)/(1.0+wTilde)   //
    );

    surfaceScalarField deltaRho(rhoNei - rhoOwn);
    surfaceVectorField deltaU(UNei - UOwn);

    surfaceScalarField deltaUv(deltaU & normal);
    //deltaUv.setOriented(false);

    surfaceScalarField deltaP(pNei - pOwn);
    vector iV(1,0,0);
    vector jV(0,1,0);
    vector kV(0,0,1);

    surfaceScalarField eps("eps",
        0.4*(mag(UTilde & iV) + mag(UTilde & jV) + mag(UTilde & kV)
        + aTilde)
    );
        
    surfaceScalarField lambda1(
        Foam::pos0(mag(UvTilde) - eps) * mag(UvTilde)
        + Foam::neg(mag(UvTilde) - eps) 
        *(sqr(UvTilde)+sqr(eps))/(2.0*eps)
        );
    surfaceScalarField lambda2(
        Foam::pos0(mag(UvTilde + aTilde) - eps) * mag(UvTilde + aTilde)
        + Foam::neg(mag(UvTilde + aTilde) - eps) 
        *(sqr(UvTilde + aTilde)+sqr(eps))/(2.0*eps)
        );
    surfaceScalarField lambda3(
        Foam::pos0(mag(UvTilde - aTilde) - eps) * mag(UvTilde - aTilde)
        + Foam::neg(mag(UvTilde - aTilde) - eps) 
        *(sqr(UvTilde - aTilde)+sqr(eps))/(2.0*eps)
        );

    surfaceScalarField alpha1
    (
        deltaRho - deltaP/sqr(aTilde)
    );

    surfaceScalarField alpha2
    (
        (deltaP + rhoTilde*aTilde*deltaUv)/(2.0*sqr(aTilde))
    );
    
    surfaceScalarField alpha3
    (
        (deltaP - rhoTilde*aTilde*deltaUv)/(2.0*sqr(aTilde))
    );

    surfaceScalarField alpha4
    (
        rhoTilde
    );

    // U Row
    surfaceVectorField K21(UTilde);
    surfaceVectorField K224((UTilde + aTilde*normal));
    surfaceVectorField K25((UTilde - aTilde*normal));
    surfaceVectorField K26(deltaU - deltaUv*normal);
    //surfaceVectorField K26(UTilde - UvTilde*normal);

    // E row
    //surfaceScalarField K31(0.5*magSqr(UTilde));
    surfaceScalarField K31(HTilde - sqr(aTilde)/gm1Tilde);
    surfaceScalarField K324((HTilde + aTilde*UvTilde));
    surfaceScalarField K35((HTilde - aTilde*UvTilde));
    surfaceScalarField K36((normal^deltaU) & (normal^UTilde));
    //surfaceScalarField K36(magSqr(UTilde - UvTilde*normal));

    // Compute owner and neighbour fluxes
    surfaceScalarField UvOwn(UO & normal);
    surfaceScalarField UvNei(UN & normal);
    

    surfaceScalarField massFluxOwn(rhoOwn*UvOwn);
    surfaceScalarField massFluxNei(rhoNei*UvNei);

    surfaceVectorField momentumFluxOwn
    (
        UOwn*massFluxOwn + pOwn*normal
    );

    surfaceVectorField momentumFluxNei
    (
        UOwn*massFluxNei + pNei*normal
    );

    surfaceScalarField energyFluxOwn(HOwn*massFluxOwn);
    surfaceScalarField energyFluxNei(HNei*massFluxNei);

    surfaceScalarField lambdaAlpha
    (
        mag(lambda1)*alpha1
      + mag(lambda2)*alpha2
      + mag(lambda3)*alpha3
     // + mag(lambda1)*alpha1
    );

    // Compute fluxes
    massFlux = 0.5*magSf*(massFluxOwn + massFluxNei - lambdaAlpha);

    surfaceVectorField lambdaAlphaK
    (
        mag(lambda1)*alpha1*K21
      + mag(lambda2)*alpha2*K224
      + mag(lambda3)*alpha3*K25
      + mag(lambda1)*alpha4*K26
    );

    //lambdaAlphaK.setOriented(true);

    momentumFlux =
        0.5*magSf*(momentumFluxOwn + momentumFluxNei - lambdaAlphaK);

    //momentumFlux.setOriented(true);

    surfaceScalarField lambdaAlphaKE
    (
        mag(lambda1)*alpha1*K31
      + mag(lambda2)*alpha2*K324
      + mag(lambda3)*alpha3*K35
      + mag(lambda1)*alpha4*K36
    );

    //lambdaAlphaKE.setOriented(true);

    energyFlux =
        0.5*magSf*(energyFluxOwn + energyFluxNei - lambdaAlphaKE) + 0.5*meshPhi*(pNei + pOwn);

    //energyFlux.setOriented(true);
}