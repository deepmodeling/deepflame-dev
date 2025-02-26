/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2023 OpenCFD Ltd.
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

#include "radiationModel.H"
#include "fvDOM.H"
#include "specularRadiationMixedFvPatchScalarField.H"
#include "addToRunTimeSelectionTable.H"
#include "wedgePolyPatch.H"
#include "symmetryPlanePolyPatch.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace radiation
{

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

scalar specularRadiationMixedFvPatchScalarField::azimuthAngle
(
    const vector& d
) const
{
    return sign(d.y())*Foam::acos(d.x()/Foam::sqrt(sqr(d.x()) + sqr(d.y())));
}


scalar specularRadiationMixedFvPatchScalarField::polarAngle
(
    const vector& d
) const
{
    return Foam::acos(d.z()/mag(d));
}


tmp<scalarField> specularRadiationMixedFvPatchScalarField::interpolateI
(
    const fvDOM& dom,
    const label closestRayi
) const
{
    // Calculate the reflected ray for this ray (KE:p. 2)
    const vector dAve(normalised(dom.IRay(rayID_).dAve()));
    const vector dSpe(normalised(dAve - 2*(dAve & n_)*n_));


    // Fetch the number of polar and azimuthal segments
    const label nPolar = dom.nTheta();
    const label nAzimuth = 4*dom.nPhi();


    // Find the neighbouring ray indices of the reflected ray in the east
    // and west
    // Go to the east
    const label polari = std::floor(closestRayi/nAzimuth) + 1;
    label eastRayi = closestRayi + 1;
    if (eastRayi == nAzimuth*polari)
    {
        eastRayi -= nAzimuth;
    }
    // Go to the west
    label westRayi = closestRayi - 1;
    if (westRayi < nAzimuth*(polari - 1))
    {
        westRayi += nAzimuth;
    }

    // Find the ray index closest to the reflected ray in the azimuthal
    // direction
    label azimuthRayi = -1;
    bool east = false;
    const vector dEast(normalised(dom.IRay(eastRayi).dAve()));
    const vector dWest(normalised(dom.IRay(westRayi).dAve()));
    if (mag(dSpe - dEast) < mag(dSpe - dWest))
    {
        azimuthRayi = eastRayi;
        east = true;
    }
    else
    {
        azimuthRayi = westRayi;
    }


    // Find the neighbouring ray indices of the reflected ray in the north
    // and south
    // Go to the north
    label northRayi = closestRayi - nAzimuth;
    if (northRayi < 0)
    {
        // The pole is inside the polar segment - skip
        northRayi = -1;
    }
    // Go to the south
    label southRayi = closestRayi + nAzimuth;
    if (southRayi > nPolar*nAzimuth-1)
    {
        // The pole is inside the polar segment - skip
        southRayi = -1;
    }


    // Find the ray index closest to the reflected ray in the polar direction
    label polarRayi = -1;
    if (northRayi != -1 && southRayi != -1)
    {
        const vector dNorth(normalised(dom.IRay(northRayi).dAve()));
        const vector dSouth(normalised(dom.IRay(southRayi).dAve()));

        if (mag(dSpe - dNorth) < mag(dSpe - dSouth))
        {
            polarRayi = northRayi;
        }
        else
        {
            polarRayi = southRayi;
        }
    }
    else if (northRayi != -1)
    {
        polarRayi = northRayi;
    }
    else if (southRayi != -1)
    {
        polarRayi = southRayi;
    }


    // Find the ray index neighbouring the azimuthal and polar neighbour rays
    label cornerRayi = -1;
    if (polarRayi != -1)
    {
        if (east)
        {
            cornerRayi = polarRayi + 1;

            if (!(cornerRayi % nAzimuth))
            {
                cornerRayi -= nAzimuth;
            }
        }
        else
        {
            cornerRayi = polarRayi - 1;
            if (!(polarRayi % nAzimuth))
            {
                cornerRayi += nAzimuth;
            }
        }
    }


    // Interpolate the ray intensity of the reflected complementary ray from
    // the neighbouring rays
    const label patchi = this->patch().index();
    auto tIc = tmp<scalarField>::New(this->patch().size(), Zero);
    auto& Ic = tIc.ref();


    if (polarRayi == -1)
    {
        // Linear interpolation only in the azimuth direction
        const vector d1(normalised(dom.IRay(closestRayi).dAve()));
        const vector d2(normalised(dom.IRay(azimuthRayi).dAve()));

        const scalar phic = azimuthAngle(dSpe);
        const scalar phi1 = azimuthAngle(d1);
        const scalar phi2 = azimuthAngle(d2);

        const auto& I1 =
            dom.IRayLambda(closestRayi, lambdaID_).boundaryField()[patchi];
        const auto& I2 =
            dom.IRayLambda(azimuthRayi, lambdaID_).boundaryField()[patchi];

        Ic = lerp(I1, I2, (phic - phi1)/(phi2 - phi1));
    }
    else
    {
        // Bilinear interpolation in the azimuth and polar directions
        const vector d1(normalised(dom.IRay(closestRayi).dAve()));
        const vector d2(normalised(dom.IRay(azimuthRayi).dAve()));
        const vector d3(normalised(dom.IRay(polarRayi).dAve()));
        const vector d4(normalised(dom.IRay(cornerRayi).dAve()));

        const scalar phic = azimuthAngle(dSpe);
        const scalar phi1 = azimuthAngle(d1);
        const scalar phi2 = azimuthAngle(d2);
        const scalar phi3 = azimuthAngle(d3);
        const scalar phi4 = azimuthAngle(d4);

        const scalar thetac = polarAngle(dSpe);
        const scalar theta1 = polarAngle(d1);
        const scalar theta3 = polarAngle(d3);

        const auto& I1 =
            dom.IRayLambda(closestRayi, lambdaID_).boundaryField()[patchi];
        const auto& I2 =
            dom.IRayLambda(azimuthRayi, lambdaID_).boundaryField()[patchi];
        const auto& I3 =
            dom.IRayLambda(polarRayi, lambdaID_).boundaryField()[patchi];
        const auto& I4 =
            dom.IRayLambda(cornerRayi, lambdaID_).boundaryField()[patchi];

        const scalarField Ia(lerp(I1, I2, (phic - phi1)/(phi2 - phi1)));
        const scalarField Ib(lerp(I3, I4, (phic - phi3)/(phi4 - phi3)));

        Ic = lerp(Ia, Ib, (thetac - theta1)/(theta3 - theta1));
    }

    return tIc;
}


label specularRadiationMixedFvPatchScalarField::calcComplementaryRayID
(
    const fvDOM& dom
) const
{
    vectorList dAve(dom.nRay());
    forAll(dAve, rayi)
    {
        dAve[rayi] = normalised(dom.IRay(rayi).dAve());
    }


    // Check if the ray goes out of the domain, ie. no reflection
    if ((dAve[rayID_] & n_) > 0)
    {
        return -1;
    }


    // Calculate the reflected ray direction for rayID (KE:p. 2)
    const vector dSpe
    (
        normalised(dAve[rayID_] - 2*(dAve[rayID_] & n_)*n_)
    );


    // Find the closest ray to the reflected ray
    label complementaryRayID = -1;
    scalar dotProductThisRay = -GREAT;
    forAll(dAve, i)
    {
        const scalar dotProductOtherRay = dAve[i] & dSpe;

        if (dotProductThisRay < dotProductOtherRay)
        {
            complementaryRayID = i;
            dotProductThisRay = dotProductOtherRay;
        }
    }

    return complementaryRayID;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

specularRadiationMixedFvPatchScalarField::
specularRadiationMixedFvPatchScalarField
(
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF
)
:
    mixedFvPatchScalarField(p, iF),
    n_(),
    rayID_(-1),
    lambdaID_(-1),
    interpolate_(false)
{
    this->refValue() = Zero;
    this->refGrad() = Zero;
    this->valueFraction() = Zero;
}


specularRadiationMixedFvPatchScalarField::
specularRadiationMixedFvPatchScalarField
(
    const specularRadiationMixedFvPatchScalarField& ptf,
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    mixedFvPatchScalarField(ptf, p, iF, mapper),
    n_(ptf.n_),
    rayID_(ptf.rayID_),
    lambdaID_(ptf.lambdaID_),
    interpolate_(ptf.interpolate_)
{}


specularRadiationMixedFvPatchScalarField::
specularRadiationMixedFvPatchScalarField
(
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF,
    const dictionary& dict
)
:
    mixedFvPatchScalarField(p, iF),
    n_(),
    rayID_(-1),
    lambdaID_(-1),
    interpolate_(dict.getOrDefault("interpolate", false))
{
    this->refValue() = Zero;
    this->refGrad() = Zero;
    this->valueFraction() = Zero;

    if (!this->readValueEntry(dict))
    {
        fvPatchScalarField::operator=(this->refValue());
    }


    if (isA<wedgePolyPatch>(p.patch()))
    {
        const auto& wp = refCast<const wedgePolyPatch>(p.patch());
        n_ = wp.n();
    }
    else if (isA<symmetryPlanePolyPatch>(p.patch()))
    {
        const auto& sp = refCast<const symmetryPlanePolyPatch>(p.patch());
        n_ = sp.n();
    }
    else
    {
        FatalErrorInFunction
            << " specularRadiation boundary condition is limited to "
            << "wedge or symmetry-plane geometries." << nl
            << exit(FatalError);
    }
}


specularRadiationMixedFvPatchScalarField::
specularRadiationMixedFvPatchScalarField
(
    const specularRadiationMixedFvPatchScalarField& ptf,
    const DimensionedField<scalar, volMesh>& iF
)
:
    mixedFvPatchScalarField(ptf, iF),
    n_(ptf.n_),
    rayID_(ptf.rayID_),
    lambdaID_(ptf.lambdaID_),
    interpolate_(ptf.interpolate_)
{}


specularRadiationMixedFvPatchScalarField::
specularRadiationMixedFvPatchScalarField
(
    const specularRadiationMixedFvPatchScalarField& ptf
)
:
    mixedFvPatchScalarField(ptf),
    n_(ptf.n_),
    rayID_(ptf.rayID_),
    lambdaID_(ptf.lambdaID_),
    interpolate_(ptf.interpolate_)
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void specularRadiationMixedFvPatchScalarField::updateCoeffs()
{
    if (this->updated())
    {
        return;
    }

    const fvDOM& dom = db().lookupObject<fvDOM>("radiationProperties");

    // Get rayID and lambdaID for this ray
    if (rayID_ == -1 && lambdaID_ == -1)
    {
        dom.setRayIdLambdaId(internalField().name(), rayID_, lambdaID_);
    }


    // Find the ray index closest to the reflected ray
    const label compID = calcComplementaryRayID(dom);


    if (compID == -1)
    {
        // Apply zero-gradient condition for rays outgoing from the domain
        this->valueFraction() = 0;
    }
    else
    {
        // Apply fixed condition for rays incoming to the domain
        this->valueFraction() = 1;

        if (!interpolate_)
        {
            // Fetch the existing ray closest to the reflected ray (KE:Eq. 4)
            this->refValue() =
                dom.IRayLambda
                (
                    compID,
                    lambdaID_
                ).internalField();
        }
        else
        {
            // Interpolate the ray intensity from neighbouring rays (KE:p. 2)
            this->refValue() = interpolateI(dom, compID);
        }
    }

    mixedFvPatchScalarField::updateCoeffs();
}


void specularRadiationMixedFvPatchScalarField::write(Ostream& os) const
{
    mixedFvPatchScalarField::write(os);
    os.writeEntryIfDifferent<bool>("interpolate", false, interpolate_);
    this->writeValueEntry(os);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

makePatchTypeField
(
   fvPatchScalarField,
   specularRadiationMixedFvPatchScalarField
);


} // End namespace radiation
} // End namespace Foam

// ************************************************************************* //
