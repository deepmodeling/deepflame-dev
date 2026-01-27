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
    Foam::dfGAMGInterfaceField

Description
     Abstract base class for GAMG agglomerated interface fields.

SourceFiles
    dfGAMGInterfaceField.C
    dfGAMGInterfaceFieldNew.C

\*---------------------------------------------------------------------------*/

#ifndef dfGAMGInterfaceField_H
#define dfGAMGInterfaceField_H

#include "lduInterfaceField.H"
#include "dfGAMGInterface.H"
#include "autoPtr.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                      Class dfGAMGInterfaceField Declaration
\*---------------------------------------------------------------------------*/

class dfGAMGInterfaceField
:
    public lduInterfaceField
{
    // Private Data

        //- Local reference cast into the interface
        const dfGAMGInterface& interface_;


public:

    //- Runtime type information
    TypeName("dfGAMGInterfaceField");


    // Declare run-time constructor selection tables

        declareRunTimeSelectionTable
        (
            autoPtr,
            dfGAMGInterfaceField,
            lduInterfaceField,
            (
                const dfGAMGInterface& GAMGCp,
                const lduInterfaceField& fineInterface
            ),
            (GAMGCp, fineInterface)
        );

        declareRunTimeSelectionTable
        (
            autoPtr,
            dfGAMGInterfaceField,
            lduInterface,
            (
                const dfGAMGInterface& GAMGCp,
                const bool doTransform,
                const int rank
            ),
            (GAMGCp, doTransform, rank)
        );


    // Selectors

        //- Return a pointer to a new interface created on freestore given
        //  the fine interface
        static autoPtr<dfGAMGInterfaceField> New
        (
            const dfGAMGInterface& GAMGCp,
            const lduInterfaceField& fineInterface
        );

        //- Return a pointer to a new interface created on freestore given
        //  the fine interface
        static autoPtr<dfGAMGInterfaceField> New
        (
            const dfGAMGInterface& GAMGCp,
            const bool doTransform,
            const int rank
        );


    // Constructors

        //- Construct from GAMG interface and fine level interface field
        dfGAMGInterfaceField
        (
            const dfGAMGInterface& GAMGCp,
            const lduInterfaceField&
        )
        :
            lduInterfaceField(GAMGCp),
            interface_(GAMGCp)
        {}

        //- Construct from GAMG interface and fine level interface field
        dfGAMGInterfaceField
        (
            const dfGAMGInterface& GAMGCp,
            const bool doTransform,
            const int rank
        )
        :
            lduInterfaceField(GAMGCp),
            interface_(GAMGCp)
        {}

        //- Disallow default bitwise copy construction
        dfGAMGInterfaceField(const dfGAMGInterfaceField&) = delete;


    // Member Functions

        // Access

            //- Return interface
            const dfGAMGInterface& interface() const
            {
                return interface_;
            }


    // Member Operators

        //- Disallow default bitwise assignment
        void operator=(const dfGAMGInterfaceField&) = delete;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
