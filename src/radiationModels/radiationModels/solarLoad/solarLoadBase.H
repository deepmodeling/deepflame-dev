/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2022 OpenCFD Ltd.
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
    Foam::radiation::solarLoadBase

Group
    grpRadiationModels

Description
    Base class for solarLoad models.

\*---------------------------------------------------------------------------*/

#ifndef radiation_solarLoadBase_H
#define radiation_solarLoadBase_H

#include "regIOobject.H"
#include "fvMesh.H"
#include "solarCalculator.H"
#include "faceShading.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace radiation
{

/*---------------------------------------------------------------------------*\
                        Class solarLoadBase Declaration
\*---------------------------------------------------------------------------*/

class solarLoadBase
:
    public regIOobject
{
public:

    //- Runtime type information
    TypeName("solarLoadBase");


    // Constructors

        //- Construct
        solarLoadBase(const fvMesh& mesh);


    //- Destructor
    virtual ~solarLoadBase() = default;


    // Member Functions

    // Access

        //- Return const reference to the solar calculator
        virtual const solarCalculator& solarCalculatorRef() const = 0;

        //- Return const reference to the face shading calculator
        virtual const faceShading& faceShadingRef() const = 0;


    // IO

        bool writeData(Foam::Ostream&) const
        {
            return true;
        }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace radiation
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
