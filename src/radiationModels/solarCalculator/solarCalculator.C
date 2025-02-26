/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2015-2024 OpenCFD Ltd.
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

#include "solarCalculator.H"
#include "Time.H"
#include "unitConversion.H"
#include "constants.H"

using namespace Foam::constant;

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(solarCalculator, 0);
}


const Foam::NamedEnum
<
    Foam::solarCalculator::sunDirModel,
    2
>
Foam::solarCalculator::sunDirectionModelTypeNames_
({
    { sunDirModel::mSunDirConstant, "constant" },
    { sunDirModel::mSunDirTracking, "tracking" },

    // old long names (v2012 and earlier)
    { sunDirModel::mSunDirConstant, "sunDirConstant" },
    { sunDirModel::mSunDirTracking, "sunDirTracking" }
});


const Foam::NamedEnum
<
    Foam::solarCalculator::sunLModel,
    4
>
Foam::solarCalculator::sunLModelTypeNames_
({
    { sunLModel::mSunLoadConstant, "constant" },
    { sunLModel::mSunLoadTimeDependent, "timeDependent" },
    { sunLModel::mSunLoadFairWeatherConditions, "fairWeather" },
    { sunLModel::mSunLoadTheoreticalMaximum, "theoreticalMaximum" },

    // old long names (v2012 and earlier)
    { sunLModel::mSunLoadConstant, "sunLoadConstant" },
    {
        sunLModel::mSunLoadFairWeatherConditions,
        "sunLoadFairWeatherConditions"
    },
    { sunLModel::mSunLoadTheoreticalMaximum, "sunLoadTheoreticalMaximum" }
});


// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

void Foam::solarCalculator::calculateBetaTheta()
{
    scalar runTime = 0;

    if (sunDirectionModel_ == mSunDirTracking)
    {
        runTime = mesh_.time().value();
    }

    const scalar LSM = 15.0*(dict_.get<scalar>("localStandardMeridian"));

    const scalar D = dict_.get<scalar>("startDay") + runTime/86400.0;
    const scalar M = 6.24004 + 0.0172*D;
    const scalar EOT = -7.659*sin(M) + 9.863*sin(2*M + 3.5932);

    dict_.readEntry("startTime", startTime_);

    const scalar LST = startTime_ + std::fmod(runTime/3600.0, 24);

    const scalar LON = dict_.get<scalar>("longitude");

    const scalar AST = LST + EOT/60.0 + (LON - LSM)/15;

    const scalar delta = 23.45*sin(degToRad((360*(284 + D))/365));

    const scalar H = degToRad(15*(AST - 12));

    const scalar L = degToRad(dict_.get<scalar>("latitude"));

    const scalar deltaRad = degToRad(delta);
    beta_ = max(asin(cos(L)*cos(deltaRad)*cos(H) + sin(L)*sin(deltaRad)), 1e-3);
    theta_ = acos((sin(beta_)*sin(L) - sin(deltaRad))/(cos(beta_)*cos(L)));

    // theta is the angle between the SOUTH axis and the Sun
    // If the hour angle is lower than zero (morning) the Sun is positioned
    // on the East side.
    if (H < 0)
    {
        theta_ += 2*(constant::mathematical::pi - theta_);
    }

    DebugInfo
        << tab << "altitude : " << radToDeg(beta_) << nl
        << tab << "azimuth  : " << radToDeg(theta_) << endl;
}


void Foam::solarCalculator::calculateSunDirection()
{
    gridUp_  = normalised(dict_.get<vector>("gridUp"));
    eastDir_ = normalised(dict_.get<vector>("gridEast"));

    coord_.reset
    (
        new coordinateSystem("grid", Zero, gridUp_, eastDir_)
    );

    // Assuming 'z' vertical, 'y' North and 'x' East
    direction_.z() = -sin(beta_);
    direction_.y() =  cos(beta_)*cos(theta_); // South axis
    direction_.x() =  cos(beta_)*sin(theta_); // West axis

    direction_.normalise();

    DebugInfo
        << "Sun direction in absolute coordinates : " << direction_ <<endl;

    // Transform to actual coordinate system
    direction_ = coord_->transform(direction_);

    DebugInfo
        << "Sun direction in the Grid coordinates : " << direction_ <<endl;
}


void Foam::solarCalculator::initialise()
{
    switch (sunDirectionModel_)
    {
        case mSunDirConstant:
        {
            if (dict_.readIfPresent("sunDirection", direction_))
            {
                direction_.normalise();
            }
            else
            {
                calculateBetaTheta();
                calculateSunDirection();
            }
            break;
        }
        case mSunDirTracking:
        {
            if (word(mesh_.ddtScheme("default")) == "steadyState")
            {
                FatalErrorInFunction
                    << " Sun direction model can not be sunDirtracking if the "
                    << " case is steady " << nl << exit(FatalError);
            }

            dict_.readEntry
            (
                "sunTrackingUpdateInterval",
                sunTrackingUpdateInterval_
            );

            calculateBetaTheta();
            calculateSunDirection();
            break;
        }
    }

    switch (sunLoadModel_)
    {
        case mSunLoadConstant:
        {
            dict_.readEntry("directSolarRad", directSolarRad_);
            dict_.readEntry("diffuseSolarRad", diffuseSolarRad_);
            break;
        }
        case mSunLoadTimeDependent:
        {
            directSolarRads_.reset
            (
                Function1<scalar>::New
                (
                    "directSolarRad",
                    dict_,
                    &mesh_
                )
            );

            diffuseSolarRads_.reset
            (
                Function1<scalar>::New
                (
                    "diffuseSolarRad",
                    dict_,
                    &mesh_
                )
            );

            directSolarRad_ =
                directSolarRads_->value(mesh_.time().timeOutputValue());
            diffuseSolarRad_ =
                diffuseSolarRads_->value(mesh_.time().timeOutputValue());
            break;
        }
        case mSunLoadFairWeatherConditions:
        {
            dict_.readIfPresent
            (
                "skyCloudCoverFraction",
                skyCloudCoverFraction_
            );

            dict_.readEntry("A", A_);
            dict_.readEntry("B", B_);
            dict_.readEntry("C", C_);
            dict_.readEntry("groundReflectivity", groundReflectivity_);
            if (!dict_.readIfPresent("beta", beta_))
            {
                calculateBetaTheta();
            }

            directSolarRad_ =
                (1.0 - 0.75*pow(skyCloudCoverFraction_, 3.0))
              * A_/exp(B_/sin(beta_));
            break;
        }
        case mSunLoadTheoreticalMaximum:
        {
            dict_.readEntry("Setrn", Setrn_);
            dict_.readEntry("SunPrime", SunPrime_);
            dict_.readEntry("groundReflectivity", groundReflectivity_);
            dict_.readEntry("C", C_);

            directSolarRad_ = Setrn_*SunPrime_;
            break;
        }
    }
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::solarCalculator::solarCalculator
(
    const dictionary& dict,
    const fvMesh& mesh
)
:
    mesh_(mesh),
    dict_(dict),
    sunDirectionModel_
    (
        sunDirectionModelTypeNames_.get("sunDirectionModel", dict)
    ),
    sunLoadModel_(sunLModelTypeNames_.get("sunLoadModel", dict)),
    direction_(Zero),
    sunTrackingUpdateInterval_(0),
    startTime_(0),
    gridUp_(Zero),
    eastDir_(Zero),
    coord_(),
    directSolarRad_(0),
    diffuseSolarRad_(0),
    directSolarRads_(),
    diffuseSolarRads_(),
    skyCloudCoverFraction_(0),
    groundReflectivity_(0),
    A_(0),
    B_(0),
    beta_(0),
    theta_(0),
    C_(0.058),
    Setrn_(0),
    SunPrime_(0)
{
    initialise();
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::solarCalculator::correctSunDirection()
{
    if (sunDirectionModel_ == mSunDirTracking)
    {
        calculateBetaTheta();
        calculateSunDirection();
        directSolarRad_ = A_/exp(B_/sin(max(beta_, ROOTVSMALL)));
    }
}


void Foam::solarCalculator::correctDirectSolarRad()
{
    if (sunLoadModel_ == mSunLoadTimeDependent)
    {
        directSolarRad_ = directSolarRads_->value(mesh_.time().value());
    }
}


void Foam::solarCalculator::correctDiffuseSolarRad()
{
    if (sunLoadModel_ == mSunLoadTimeDependent)
    {
        diffuseSolarRad_ = diffuseSolarRads_->value(mesh_.time().value());
    }
}


Foam::tmp<Foam::scalarField> Foam::solarCalculator::diffuseSolarRad
(
    const vectorField& n
) const
{
    auto tload = tmp<scalarField>::New(n.size());
    auto& load = tload.ref();

    forAll(n, facei)
    {
        const scalar cosEpsilon(gridUp_ & -n[facei]);

        scalar Ed = 0;
        scalar Er = 0;
        const scalar cosTheta(direction_ & -n[facei]);

        // Above the horizon
        if (cosEpsilon == 0.0)
        {
            // Vertical walls
            scalar Y = 0;

            if (cosTheta > -0.2)
            {
                Y = 0.55+0.437*cosTheta + 0.313*sqr(cosTheta);
            }
            else
            {
                Y = 0.45;
            }

            Ed = C_*Y*directSolarRad_;
        }
        else
        {
            //Other than vertical walls
            Ed =
                C_
              * directSolarRad_
              * 0.5*(1.0 + cosEpsilon);
        }

        // Ground reflected
        Er =
            directSolarRad_
            * (C_ + Foam::sin(beta_))
            * groundReflectivity_
            * 0.5*(1.0 - cosEpsilon);

        load[facei] = Ed + Er;
    }

    return tload;
}


// ************************************************************************* //
