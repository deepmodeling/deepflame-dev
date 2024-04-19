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
    Foam::processorCyclicLDUGAMGInterfaceField

Description
    GAMG agglomerated processor interface field.

SourceFiles
    processorCyclicLDUGAMGInterfaceField.C

\*---------------------------------------------------------------------------*/

#ifndef processorCyclicLDUGAMGInterfaceField_H
#define processorCyclicLDUGAMGInterfaceField_H

#include "processorLDUGAMGInterfaceField.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
               Class processorCyclicLDUGAMGInterfaceField Declaration
\*---------------------------------------------------------------------------*/

class processorCyclicLDUGAMGInterfaceField
:
    public processorLDUGAMGInterfaceField
{
    // Private Member Functions

        //- Disallow default bitwise copy construction
        processorCyclicLDUGAMGInterfaceField
        (
            const processorCyclicLDUGAMGInterfaceField&
        );

        //- Disallow default bitwise assignment
        void operator=(const processorCyclicLDUGAMGInterfaceField&) = delete;


public:

    //- Runtime type information
    TypeName("processorCyclic");


    // Constructors

        //- Construct from GAMG interface and fine level interface field
        processorCyclicLDUGAMGInterfaceField
        (
            const LDUGAMGInterface& GAMGCp,
            const lduInterfaceField& fineInterface
        );

        //- Construct from GAMG interface and fine level interface field
        processorCyclicLDUGAMGInterfaceField
        (
            const LDUGAMGInterface& GAMGCp,
            const bool doTransform,
            const int rank
        );


    // Destructor

        virtual ~processorCyclicLDUGAMGInterfaceField();
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
