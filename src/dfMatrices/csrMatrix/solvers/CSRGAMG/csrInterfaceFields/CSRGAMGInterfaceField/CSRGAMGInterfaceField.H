/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2019 OpenFOAM Foundation
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

Class
    Foam::CSRGAMGInterfaceField

Description
     Abstract base class for GAMG agglomerated interface fields.

SourceFiles
    CSRGAMGInterfaceField.C
    CSRGAMGInterfaceFieldNew.C

\*---------------------------------------------------------------------------*/

#ifndef CSRGAMGInterfaceField_H
#define CSRGAMGInterfaceField_H

#include "lduInterfaceField.H"
#include "CSRGAMGInterface.H"
#include "autoPtr.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                      Class CSRGAMGInterfaceField Declaration
\*---------------------------------------------------------------------------*/

class CSRGAMGInterfaceField
:
    public lduInterfaceField
{
    // Private Data

        //- Local reference cast into the interface
        const CSRGAMGInterface& interface_;


public:

    //- Runtime type information
    TypeName("CSRGAMGInterfaceField");


    // Declare run-time constructor selection tables

        declareRunTimeSelectionTable
        (
            autoPtr,
            CSRGAMGInterfaceField,
            lduInterfaceField,
            (
                const CSRGAMGInterface& GAMGCp,
                const lduInterfaceField& fineInterface
            ),
            (GAMGCp, fineInterface)
        );

        declareRunTimeSelectionTable
        (
            autoPtr,
            CSRGAMGInterfaceField,
            lduInterface,
            (
                const CSRGAMGInterface& GAMGCp,
                const bool doTransform,
                const int rank
            ),
            (GAMGCp, doTransform, rank)
        );


    // Selectors

        //- Return a pointer to a new interface created on freestore given
        //  the fine interface
        static autoPtr<CSRGAMGInterfaceField> New
        (
            const CSRGAMGInterface& GAMGCp,
            const lduInterfaceField& fineInterface
        );

        //- Return a pointer to a new interface created on freestore given
        //  the fine interface
        static autoPtr<CSRGAMGInterfaceField> New
        (
            const CSRGAMGInterface& GAMGCp,
            const bool doTransform,
            const int rank
        );


    // Constructors

        //- Construct from GAMG interface and fine level interface field
        CSRGAMGInterfaceField
        (
            const CSRGAMGInterface& GAMGCp,
            const lduInterfaceField&
        )
        :
            lduInterfaceField(GAMGCp),
            interface_(GAMGCp)
        {}

        //- Construct from GAMG interface and fine level interface field
        CSRGAMGInterfaceField
        (
            const CSRGAMGInterface& GAMGCp,
            const bool doTransform,
            const int rank
        )
        :
            lduInterfaceField(GAMGCp),
            interface_(GAMGCp)
        {}

        //- Disallow default bitwise copy construction
        CSRGAMGInterfaceField(const CSRGAMGInterfaceField&) = delete;


    // Member Functions

        // Access

            //- Return interface
            const CSRGAMGInterface& interface() const
            {
                return interface_;
            }


    // Member Operators

        //- Disallow default bitwise assignment
        void operator=(const CSRGAMGInterfaceField&) = delete;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
