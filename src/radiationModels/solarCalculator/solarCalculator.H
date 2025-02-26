/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2015-2022 OpenCFD Ltd.
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

Class
    Foam::solarCalculator

Description
    A solar calculator model providing models
    for the solar direction and solar loads.

    Available models for the solar direction:
    - \c constant: Constant sunbeam direction.
    - \c tracking: Transient model calculating sunbeam direction
    based on a given set of parameters.

    Available models for the solar load:
    - \c constant: Constant solar load.
    - \c timeDependent: Time-dependent solar load.
    - \c fairWeather: Solar fluxes are calculated following
       the "Fair Weather Conditions Method from the ASHRAE Handbook".
    - \c theoreticalMaximum: Theoretically maximum solar load.

Usage
    Minimal examples by using \c constant/radiationProperties:

    \c sunDirectionModel - Option-1:
    \verbatim
    solarLoadCoeffs
    {
        sunDirectionModel           constant;
        sunDirection                (1 0 -1);
    }
    \endverbatim

    where the entries mean:
    \table
      Property     | Description                  | Type   | Reqd | Deflt
      sunDirection | Sunbeam direction            | vector | no   | calculated
    \endtable


    \c sunDirectionModel - Option-2:
    \verbatim
    solarLoadCoeffs
    {
        sunDirectionModel           tracking;
        sunTrackingUpdateInterval   800;
        localStandardMeridian       9;
        startDay                    204;
        startTime                   15;
        longitude                   139.74;
        latitude                    35.658;
        gridUp                      (0 0 1);
        gridEast                    (1 0 0);
    }
    \endverbatim

    where the entries mean:
    \table
      Property     | Description                       | Type   | Reqd | Deflt
      sunTrackingUpdateInterval  | Interval to update the Sun direction  <!--
                   --> [decimal hours]                 | scalar | yes  | -
      localStandardMeridian | GMT (Local Zone Meridian) [decimal hours]<!--
                   -->                                 | scalar | yes  | -
      startDay     | Day from 1 to 365                 | scalar | yes  | -
      startTime    | Start time for the Sun position [decimal hours]   <!--
                   -->                                 | scalar | yes  | -
      longitude    | Geographic coordinate specifying the east–west <!--
                   --> position of a point on the surface of a planetary <!--
                   --> body [degree]                   | scalar | yes  | -
      latitude     | Geographic coordinate specifying the north–south  <!--
                   --> position of a point on the surface of a planetary <!--
                   --> body [degree]                   | scalar | yes  | -
      gridUp       | Grid orientation upwards          | vector | yes  | -
      gridEast     | Grid orientation eastwards        | vector | yes  | -
    \endtable


    \c sunLoadModel - Option-1:
    \verbatim
    solarLoadCoeffs
    {
        sunLoadModel                constant;
        directSolarRad              100;
        diffuseSolarRad             0;
    }
    \endverbatim

    where the entries mean:
    \table
      Property     | Description                       | Type   | Reqd | Deflt
      directSolarRad | Direct solar irradiation [W/m2] | scalar | yes  | -
      diffuseSolarRad | Diffuse solar irradiation on vertical surfaces <!--
                      --> [W/m2]                       | scalar | yes  | -
    \endtable


    \c sunLoadModel - Option-2:
    \verbatim
    solarLoadCoeffs
    {
        sunLoadModel                timeDependent;
        directSolarRad              <Function1<scalar>>;
        diffuseSolarRad             <Function1<scalar>>;
    }
    \endverbatim

    where the entries mean:
    \table
      Property     | Description                       | Type    | Reqd | Deflt
      directSolarRad  | Time-series of direct solar irradiation        <!--
                   --> [W/m2]          | Function1\<scalar\> | yes  | -
      diffuseSolarRad | Time-series of diffuse solar irradiation on    <!--
                   --> vertical surfaces [W/m2]                        <!--
                   -->                  | Function1\<scalar\> | yes | -
    \endtable


    \c sunLoadModel - Option-3:
    \verbatim
    solarLoadCoeffs
    {
        sunLoadModel                fairWeather;
        skyCloudCoverFraction       0.25;
        groundReflectivity          1.0;
        A                           0.1;
        B                           0.2;
        C                           0.058;
        beta                        0.15;
    }
    \endverbatim

    where the entries mean:
    \table
      Property  | Description                       | Type   | Reqd | Deflt
      A         | Apparent solar irradiation at air mass m = 0      <!--
                -->                                 | scalar | yes  | -
      B         | Atmospheric extinction coefficient                <!--
                -->                                 | scalar | yes  | -
      C         | Solar diffusivity constant        | scalar | yes  | -
      groundReflectivity | Ground reflectivity      | scalar | yes  | -
      skyCloudCoverFraction | Fraction of sky covered by clouds [0,1]  <!--
                -->                                 | scalar | no   | 0
      beta      | Solar altitude (in degrees) above the horizontal  <!--
                -->                                 | scalar | no   | calculated
    \endtable

    In this model the flux is calculated as:
    \verbatim
    directSolarRad = (1 - 0.75*skyCloudCoverFraction^3)*A/exp(B/sin(beta));
    \endverbatim


    \c sunLoadModel - Option-4:
    \verbatim
    solarLoadCoeffs
    {
        sunLoadModel                theoreticalMaximum;
        Setrn                       1.0;
        SunPrime                    4.0;
        groundReflectivity          1.0;
        C                           0.058;
    }
    \endverbatim

    where the entries mean:
    \table
      Property     | Description                       | Type   | Reqd | Deflt
      Setrn        | Parameter in maximum theoretical direct solar    <!--
                   --> model                           | scalar | yes  | -
      SunPrime     | Parameter in maximum theoretical direct solar    <!--
                   --> model                           | scalar | yes  | -
      groundReflectivity | Ground reflectivity         | scalar | yes  | -
      C            | Solar diffusivity constant        | scalar | yes  | -
    \endtable

    In this model the flux is calculated as:
    \verbatim
    directSolarRad = Setrn*SunPrime;
    \endverbatim

Note
  - The \c sunDirectionModel:tracking can only be used
  in transient calculations.
  - The keyword \c sunTrackingUpdateInterval (in hours) specifies on which
    interval is the Sun direction updated.
  - The diffuse on vertical/horizontal walls and ground-reflected radiation are
    calculated following the ASHRAE Handbook.
  - The range of \c skyCloudCoverFraction is [0,1].

SourceFiles
    solarCalculator.C

\*---------------------------------------------------------------------------*/

#ifndef solarCalculator_H
#define solarCalculator_H

#include "fvMesh.H"
#include "meshTools.H"
#include "DynamicField.H"
#include "HashSet.H"
#include "coordinateSystem.H"
#include "Function1.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class solarCalculator Declaration
\*---------------------------------------------------------------------------*/

class solarCalculator
{
public:

    //  Public Enumeration

        //- Options for the Sun direction models
        enum sunDirModel
        {
            mSunDirConstant,
            mSunDirTracking
        };

        //- Names for sunDirModel
        static const NamedEnum<sunDirModel, 2> sunDirectionModelTypeNames_;

        //- Options for the Sun load models
        enum sunLModel
        {
            mSunLoadConstant,
            mSunLoadTimeDependent,
            mSunLoadFairWeatherConditions,
            mSunLoadTheoreticalMaximum
        };

        //- Names for sunLModel
        static const NamedEnum<sunLModel, 4> sunLModelTypeNames_;


private:

    // Private Data

        //- Reference to mesh
        const fvMesh& mesh_;

        //- Dictionary
        dictionary dict_;

        //- Sun direction model
        sunDirModel sunDirectionModel_;

        //- Sun load model
        sunLModel sunLoadModel_;


        // sunDirectionModel = constant

            //- Sunbeam direction
            vector direction_;


        // sunDirectionModel = tracking

            //- Interval to update the Sun direction [decimal hours]
            scalar sunTrackingUpdateInterval_;

            //- Start time for the Sun position [decimal hours]
            scalar startTime_;

            //- Up grid orientation
            vector gridUp_;

            //- East grid orientation
            vector eastDir_;

            //- Grid coordinate system
            autoPtr<coordinateSystem> coord_;


        // sunLoadModel = constant

            //- Direct solar irradiation
            scalar directSolarRad_;

            //- Diffuse solar irradiation on vertical surfaces
            scalar diffuseSolarRad_;


        // sunLoadModel = timeDependent

            //- Time-series of direct solar irradiation
            autoPtr<Function1<scalar>> directSolarRads_;

            //- Time-series of diffuse solar irradiation on vertical surfaces
            autoPtr<Function1<scalar>> diffuseSolarRads_;


        // sunLoadModel = fairWeather

            //- Sky cloud cover fraction [0-1]
            scalar skyCloudCoverFraction_;

            //- Ground reflectivity
            scalar groundReflectivity_;

            //- Fair weather direct solar load model parameters
            scalar A_;
            scalar B_;
            scalar beta_;
            scalar theta_;

            //- Diffusive solar load model parameter
            scalar C_;


        // sunLoadModel = theoreticalMaximum

            //- Maximum theoretical direct solar load model parameters
            scalar Setrn_;
            scalar SunPrime_;


        //- No copy construct
        solarCalculator(const solarCalculator&) = delete;

        //- No copy assignment
        void operator=(const solarCalculator&) = delete;


    // Private Member Functions

        //- Initialise model parameters
        void initialise();

        //- Calculate beta and theta angles
        void calculateBetaTheta();

        //- Calculate the Sun direction
        void calculateSunDirection();


public:

    // Declare name of the class and its debug switch
    ClassName("solarCalculator");


    // Constructors

        //- Construct from dictionary
        solarCalculator(const dictionary&, const fvMesh&);


    //- Destructor
    ~solarCalculator() = default;


    // Member Functions

        // Evaluation

            //- Correct the Sun direction
            void correctSunDirection();

            //- Correct direct solar irradiation
            void correctDirectSolarRad();

            //- Correct diffuse solar irradiation
            void correctDiffuseSolarRad();

            //- Return the diffusive solar irradiation
            //- for sunLoadModel = fairWeather
            //- or for sunLoadModel = theoreticalMaximum
            tmp<scalarField> diffuseSolarRad(const vectorField& n) const;


        // Access

            //- Return const access to the Sun direction model
            const sunDirModel& sunDirectionModel() const noexcept
            {
                return sunDirectionModel_;
            }

            //- Return const access to the Sun load model
            const sunLModel& sunLoadModel() const noexcept
            {
                return sunLoadModel_;
            }

            //- Return non-const access to the Sun direction
            vector& direction()
            {
                return direction_;
            }

            //- Return const access to the Sun direction
            const vector& direction() const noexcept
            {
                return direction_;
            }

            //- Return non-const access to the direct solar irradiation
            scalar& directSolarRad()
            {
                return directSolarRad_;
            }

            //- Return const access to the direct solar irradiation
            const scalar& directSolarRad() const noexcept
            {
                return directSolarRad_;
            }

            //- Return non-const access to the diffuse solar irradiation
            scalar& diffuseSolarRad()
            {
                return diffuseSolarRad_;
            }

            //- Return const access to the diffuse solar irradiation
            const scalar& diffuseSolarRad() const noexcept
            {
                return diffuseSolarRad_;
            }

            //- Return const access to the C constant
            scalar C() const noexcept
            {
                return C_;
            }

            //- Return const access to beta
            scalar beta() const noexcept
            {
                return beta_;
            }

            //- Return const access to theta
            scalar theta() const noexcept
            {
                return theta_;
            }

            //- Return const access to the ground reflectivity
            scalar groundReflectivity() const noexcept
            {
                return groundReflectivity_;
            }

            //- Return const access to the coordinate system
            const coordinateSystem& coord() const noexcept
            {
                return *coord_;
            }

            //- Return const access to sunTrackingUpdateInterval
            scalar sunTrackingUpdateInterval() const noexcept
            {
                return sunTrackingUpdateInterval_;
            }

            //- Return const access to startTime
            scalar startTime() const noexcept
            {
                return startTime_;
            }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
