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
    Foam::processorCyclicELLGAMGInterfaceField

Description
    GAMG agglomerated processor interface field.

SourceFiles
    processorCyclicELLGAMGInterfaceField.C

\*---------------------------------------------------------------------------*/

#ifndef processorCyclicELLGAMGInterfaceField_H
#define processorCyclicELLGAMGInterfaceField_H

#include "processorELLGAMGInterfaceField.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
               Class processorCyclicELLGAMGInterfaceField Declaration
\*---------------------------------------------------------------------------*/

class processorCyclicELLGAMGInterfaceField
:
    public processorELLGAMGInterfaceField
{
    // Private Member Functions

        //- Disallow default bitwise copy construction
        processorCyclicELLGAMGInterfaceField
        (
            const processorCyclicELLGAMGInterfaceField&
        );

        //- Disallow default bitwise assignment
        void operator=(const processorCyclicELLGAMGInterfaceField&) = delete;


public:

    //- Runtime type information
    TypeName("processorCyclic");


    // Constructors

        //- Construct from GAMG interface and fine level interface field
        processorCyclicELLGAMGInterfaceField
        (
            const ELLGAMGInterface& GAMGCp,
            const lduInterfaceField& fineInterface
        );

        //- Construct from GAMG interface and fine level interface field
        processorCyclicELLGAMGInterfaceField
        (
            const ELLGAMGInterface& GAMGCp,
            const bool doTransform,
            const int rank
        );


    // Destructor

        virtual ~processorCyclicELLGAMGInterfaceField();
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
