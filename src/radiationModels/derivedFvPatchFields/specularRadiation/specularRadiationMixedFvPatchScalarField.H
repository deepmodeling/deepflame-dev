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

Class
    Foam::radiation::specularRadiationMixedFvPatchScalarField

Description
    This boundary condition provides a specular radiation condition for
    axisymmetric and symmetry-plane \c fvDOM computations.

    References:
    \verbatim
        Standard model (tag:KE):
            Kumar, P., & Eswaran, V. (2013).
            A methodology to solve 2D and axisymmetric radiative
            transfer problems using a general 3D solver.
            Journal of heat transfer, 135(12).
            DOI:10.1115/1.4024674
    \endverbatim

Usage
    Example of the boundary condition specification:
    \verbatim
    <patchName>
    {
        // Mandatory entries
        type                specularRadiation;

        // Optional entries
        interpolate         <bool>;

        // Inherited entries
        patchType           <word>;
        ...
    }
    \endverbatim

    where the entries mean:
    \table
      Property | Description                            | Type  | Reqd | Deflt
      type     | Type name: turbulentDigitalFilterInlet | word  | yes  | -
      interpolate | Flag to enable ray-intensity interp | bool  | no   | false
    \endtable

    The inherited entries are elaborated in:
      - \link mixedFvPatchFields.H \endlink

Note
  - The condition is limited to the underlying \c wedge and \c symmetryPlane
    conditions.

SourceFiles
    specularRadiationMixedFvPatchScalarField.C

\*---------------------------------------------------------------------------*/

#ifndef Foam_specularRadiationMixedFvPatchScalarField_H
#define Foam_specularRadiationMixedFvPatchScalarField_H

#include "mixedFvPatchFields.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace radiation
{

/*---------------------------------------------------------------------------*\
          Class specularRadiationMixedFvPatchScalarField Declaration
\*---------------------------------------------------------------------------*/

class specularRadiationMixedFvPatchScalarField
:
    public mixedFvPatchScalarField
{
    // Private Data

        //- Patch normal vector
        vector n_;

        //- Ray index of this ray
        label rayID_;

        //- Band index of this ray
        label lambdaID_;

        //- Flag to enable ray-intensity interpolation
        const bool interpolate_;


    // Private Member Functions

        //- Return the corresponding azimuth angle of a given Cartesian vector
        scalar azimuthAngle(const vector& d) const;

        //- Return the corresponding polar angle of a given Cartesian vector
        scalar polarAngle(const vector& d) const;

        //- Return interpolated ray intensity
        tmp<scalarField> interpolateI
        (
            const fvDOM& dom,
            const label closestRayID
        ) const;

        //- Return the index of complementary ray
        label calcComplementaryRayID(const fvDOM& dom) const;


public:

    //- Runtime type information
    TypeName("specularRadiation");


    // Constructors

        //- Construct from patch and internal field
        specularRadiationMixedFvPatchScalarField
        (
            const fvPatch&,
            const DimensionedField<scalar, volMesh>&
        );

        //- Construct from patch, internal field and dictionary
        specularRadiationMixedFvPatchScalarField
        (
            const fvPatch&,
            const DimensionedField<scalar, volMesh>&,
            const dictionary&
        );

        //- Construct by mapping given
        //- specularRadiationMixedFvPatchScalarField onto a new patch
        specularRadiationMixedFvPatchScalarField
        (
            const specularRadiationMixedFvPatchScalarField&,
            const fvPatch&,
            const DimensionedField<scalar, volMesh>&,
            const fvPatchFieldMapper&
        );

        //- Construct as copy
        specularRadiationMixedFvPatchScalarField
        (
            const specularRadiationMixedFvPatchScalarField&
        );

        //- Construct as copy setting internal field reference
        specularRadiationMixedFvPatchScalarField
        (
            const specularRadiationMixedFvPatchScalarField&,
            const DimensionedField<scalar, volMesh>&
        );

        //- Return a clone
        virtual tmp<fvPatchField<scalar>> clone() const
        {
            return fvPatchField<scalar>::Clone(*this);
        }

        //- Clone with an internal field reference
        virtual tmp<fvPatchField<scalar>> clone
        (
            const DimensionedField<scalar, volMesh>& iF
        ) const
        {
            return fvPatchField<scalar>::Clone(*this, iF);
        }


    // Member Functions

        //- Update the coefficients associated with the patch field
        virtual void updateCoeffs();

        //- Write
        virtual void write(Ostream&) const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace radiation
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //


#endif

// ************************************************************************* //
