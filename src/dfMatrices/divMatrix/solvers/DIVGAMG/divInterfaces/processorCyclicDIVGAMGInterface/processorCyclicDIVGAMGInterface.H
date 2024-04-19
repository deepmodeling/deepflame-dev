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
    Foam::processorCyclicDIVGAMGInterface

Description
    GAMG agglomerated processor interface.

SourceFiles
    processorCyclicDIVGAMGInterface.C
    processorCyclicDIVGAMGInterfaceTemplates.C

\*---------------------------------------------------------------------------*/

#ifndef processorCyclicDIVGAMGInterface_H
#define processorCyclicDIVGAMGInterface_H

#include "processorDIVGAMGInterface.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                  Class processorCyclicDIVGAMGInterface Declaration
\*---------------------------------------------------------------------------*/

class processorCyclicDIVGAMGInterface
:
    public processorDIVGAMGInterface
{

public:

    //- Runtime type information
    TypeName("processorCyclic");

    // Constructors

        //- Construct from fine-level interface,
        //  local and neighbour restrict addressing
        processorCyclicDIVGAMGInterface
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
        processorCyclicDIVGAMGInterface
        (
            const label index,
            const lduInterfacePtrsList& coarseInterfaces,
            Istream& is
        );

        //- Disallow default bitwise copy construction
        processorCyclicDIVGAMGInterface
        (
            const processorCyclicDIVGAMGInterface&
        ) = delete;


    // Destructor

        virtual ~processorCyclicDIVGAMGInterface();


    // Member Operators

        //- Disallow default bitwise assignment
        void operator=(const processorCyclicDIVGAMGInterface&) = delete;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
