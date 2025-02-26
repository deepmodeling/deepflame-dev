/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2015 OpenFOAM Foundation
    Copyright (C) 2018-2021 OpenCFD Ltd.
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
    Foam::radiation::solarLoad

Group
    grpRadiationModels

Description
    The \c solarLoad radiation model includes Sun primary hits, their
    reflective fluxes and diffusive sky radiative fluxes.

    The primary hit rays are calculated using a face shading algorithm.
    The first reflected fluxes can be optionally included. A view factors
    method is needed in order to include diffusive surface to surface fluxes.

    The energy is included on "visible" walls by default. The sky diffusive
    radiation for horizontal and vertical walls is calculated following the
    Fair Weather Conditions Method from the ASHRAE Handbook.

    By default the energy is included in cells adjacent to the patches into
    the energy equation (\c wallCoupled=false). On coupled patches the flux is
    by default added to the wall and considered into the solid
    (\c solidCoupled=true).

    The \c solarLoad model can be used in conjuntion with \c fvDOM and
    \c viewFactor radiation models. The flag \c useSolarLoad must be
    \c true on the \c radiationProperties dictionary.

Usage
    Minimal examples by using \c constant/radiationProperties:

    \verbatim
    solarLoadCoeffs
    {
        // Mandatory entries
        useReflectedRays            true;
        spectralDistribution        (1 5 1 2);

        // Optional entries
        solidCoupled                true;
        wallCoupled                 false;
        updateAbsorptivity          true;

        // Mandatory/Optional (inherited) entries
        ...
    }
    \endverbatim

    where the entries mean:
    \table
      Property     | Description                      | Type  | Reqd   | Deflt
      useReflectedRays | Flag to use reflected rays   | bool  | yes    | -
      spectralDistribution | Spectral distribution for the integrated <!--
                   --> solar heat flux | Function1\<scalarField\> | yes | -
      solidCoupled     | Flag to couple solids through mapped         <!--
                   --> boundary patch using qr        | bool  | no     | true
      wallCoupled      | Flag to couple wall patches using qr         <!--
                   -->                                | bool  | no     | false
      updateAbsorptivity | Flag to enable absorptivity updates        <!--
                   -->                                | bool  | no     | false
    \endtable

    The inherited entries are elaborated in:
     - \link radiationModel.H \endlink
     - \link solarCalculator.H \endlink
     - \link Function1.H \endlink

SourceFiles
    solarLoad.C

\*---------------------------------------------------------------------------*/

#ifndef radiation_solarLoad_H
#define radiation_solarLoad_H

#include "radiationModel.H"
#include "solarLoadBase.H"
#include "volFields.H"
#include "faceShading.H"
#include "faceReflecting.H"
#include "solarCalculator.H"
#include "Function1.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace radiation
{

/*---------------------------------------------------------------------------*\
                           Class solarLoad Declaration
\*---------------------------------------------------------------------------*/

class solarLoad
:
    public radiationModel,
    public solarLoadBase
{
    // Private Data

        //- Solar calculator
        solarCalculator solarCalc_;

        //- Dictionary
        dictionary dict_;

        //- Net radiative heat flux [W/m2]
        volScalarField qr_;

        //- Direct hit faces Ids
        autoPtr<faceShading> hitFaces_;

        //- Reflected faces
        autoPtr<faceReflecting> reflectedFaces_;

        //- Source term for cells next to patches with flags solidCoupled
        //- and wallCoupled false
        DimensionedField<scalar, volMesh> Ru_;

        //- Absorptivity list
        List<List<tmp<scalarField>>> absorptivity_;

        //- Spectral distribution for the integrated solar heat flux
        scalarList spectralDistribution_;

        //- Time-dependent spectral distributions
        autoPtr<Function1<scalarList>> spectralDistributions_;

        //- Primary solar radiative heat flux per band [W/m2]
        PtrList<volScalarField> qprimaryRad_;

        //- Number of bands
        label nBands_;

        //- Update Sun position index
        label updateTimeIndex_;

        //- Couple solids through mapped boundary patch using qr
        bool solidCoupled_;

        //- Couple wall patches using qr
        bool wallCoupled_;

        //- Update absorptivity
        bool updateAbsorptivity_;

        //- Include reflected rays from specular surfaces
        bool useReflectedRays_;

        //- First iteration
        bool firstIter_;


    // Private Member Functions

        //- Initialise model parameters
        void initialise(const dictionary&);

        //- Update direct hit faces radiation
        void updateDirectHitRadiation(const labelList&, const labelHashSet&);

        //- Update reflected heat flux
        void updateReflectedRays(const labelHashSet&);

        //- Calculate diffusive heat flux
        //void calculateQdiff(const labelHashSet&, const labelHashSet&);

        //- Update Sky diffusive radiation
        void updateSkyDiffusiveRadiation
        (
            const labelHashSet&,
            const labelHashSet&
        );

        //- Update hit faces
        bool updateHitFaces();

        //- Update absorptivity
        void updateAbsorptivity(const labelHashSet& includePatches);


        //- No copy construct
        solarLoad(const solarLoad&) = delete;

        //- No copy assignment
        void operator=(const solarLoad&) = delete;


public:

    //- Runtime type information
    TypeName("solarLoad");


    // Constructors

        //- Construct from volScalarField
        solarLoad(const volScalarField& T);

        //- Construct from dictionary and volScalarField
        solarLoad(const dictionary& dict, const volScalarField& T);


    //- Destructor
    virtual ~solarLoad() = default;


    // Member Functions

        // Evaluation

            //- Read radiationProperties dictionary
            bool read();

            //- Solve radiation equations
            void calculate();


        // Access

            //- Source term component (for power of T^4)
            virtual tmp<volScalarField> Rp() const;

            //- Source term component (constant)
            virtual tmp<DimensionedField<scalar, volMesh>> Ru() const;

            //- Return const access to the number of bands
            label nBands() const noexcept
            {
                return nBands_;
            }

            //- Return const access to the primary solar heat flux
            const volScalarField& qprimaryRad(const label bandI) const
            {
                return qprimaryRad_[bandI];
            }

            //- Return const reference to the solar calculator
            virtual const solarCalculator& solarCalculatorRef() const;

            //- Return const reference to the face shading calculator
            virtual const faceShading& faceShadingRef() const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace radiation
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
