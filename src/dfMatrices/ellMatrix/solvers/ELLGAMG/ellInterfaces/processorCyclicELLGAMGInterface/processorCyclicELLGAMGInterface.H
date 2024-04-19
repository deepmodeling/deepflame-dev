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
    Foam::processorCyclicELLGAMGInterface

Description
    GAMG agglomerated processor interface.

SourceFiles
    processorCyclicELLGAMGInterface.C
    processorCyclicELLGAMGInterfaceTemplates.C

\*---------------------------------------------------------------------------*/

#ifndef processorCyclicELLGAMGInterface_H
#define processorCyclicELLGAMGInterface_H

#include "processorELLGAMGInterface.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                  Class processorCyclicELLGAMGInterface Declaration
\*---------------------------------------------------------------------------*/

class processorCyclicELLGAMGInterface
:
    public processorELLGAMGInterface
{

public:

    //- Runtime type information
    TypeName("processorCyclic");

    // Constructors

        //- Construct from fine-level interface,
        //  local and neighbour restrict addressing
        processorCyclicELLGAMGInterface
        (
            const label index,
            const lduInterfacePtrsList& coarseInterfaces,
            const lduInterface& fineInterface,
            const labelField& restrictAddressing,
            const labelField& neighbourRestrictAddressing,
            const label fineLevelIndex,
            const label coarseComm
        );

        //- Construct from Istream
        processorCyclicELLGAMGInterface
        (
            const label index,
            const lduInterfacePtrsList& coarseInterfaces,
            Istream& is
        );

        //- Disallow default bitwise copy construction
        processorCyclicELLGAMGInterface
        (
            const processorCyclicELLGAMGInterface&
        ) = delete;


    // Destructor

        virtual ~processorCyclicELLGAMGInterface();


    // Member Operators

        //- Disallow default bitwise assignment
        void operator=(const processorCyclicELLGAMGInterface&) = delete;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
