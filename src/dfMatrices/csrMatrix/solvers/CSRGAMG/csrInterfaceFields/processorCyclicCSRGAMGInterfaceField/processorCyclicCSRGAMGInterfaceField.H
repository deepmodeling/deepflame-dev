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
    Foam::processorCyclicCSRGAMGInterfaceField

Description
    GAMG agglomerated processor interface field.

SourceFiles
    processorCyclicCSRGAMGInterfaceField.C

\*---------------------------------------------------------------------------*/

#ifndef processorCyclicCSRGAMGInterfaceField_H
#define processorCyclicCSRGAMGInterfaceField_H

#include "processorCSRGAMGInterfaceField.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
               Class processorCyclicCSRGAMGInterfaceField Declaration
\*---------------------------------------------------------------------------*/

class processorCyclicCSRGAMGInterfaceField
:
    public processorCSRGAMGInterfaceField
{
    // Private Member Functions

        //- Disallow default bitwise copy construction
        processorCyclicCSRGAMGInterfaceField
        (
            const processorCyclicCSRGAMGInterfaceField&
        );

        //- Disallow default bitwise assignment
        void operator=(const processorCyclicCSRGAMGInterfaceField&) = delete;


public:

    //- Runtime type information
    TypeName("processorCyclic");


    // Constructors

        //- Construct from GAMG interface and fine level interface field
        processorCyclicCSRGAMGInterfaceField
        (
            const CSRGAMGInterface& GAMGCp,
            const lduInterfaceField& fineInterface
        );

        //- Construct from GAMG interface and fine level interface field
        processorCyclicCSRGAMGInterfaceField
        (
            const CSRGAMGInterface& GAMGCp,
            const bool doTransform,
            const int rank
        );


    // Destructor

        virtual ~processorCyclicCSRGAMGInterfaceField();
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
