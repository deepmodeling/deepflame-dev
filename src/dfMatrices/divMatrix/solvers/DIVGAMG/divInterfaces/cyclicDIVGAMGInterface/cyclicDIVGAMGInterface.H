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
    Foam::cyclicDIVGAMGInterface

Description
    GAMG agglomerated cyclic interface.

SourceFiles
    cyclicDIVGAMGInterface.C

\*---------------------------------------------------------------------------*/

#ifndef cyclicDIVGAMGInterface_H
#define cyclicDIVGAMGInterface_H

#include "DIVGAMGInterface.H"
#include "cyclicLduInterface.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                  Class cyclicDIVGAMGInterface Declaration
\*---------------------------------------------------------------------------*/

class cyclicDIVGAMGInterface
:
    public DIVGAMGInterface,
    public cyclicLduInterface
{
    // Private Data

        //- Neighbour patch number
        label neighbPatchID_;

        //- Am I owner?
        bool owner_;

        //- Transformation tensor
        tensorField forwardT_;

        //- Transformation tensor
        tensorField reverseT_;


public:

    //- Runtime type information
    TypeName("cyclic");


    // Constructors

        //- Construct from fine level interface,
        //  local and neighbour restrict addressing
        cyclicDIVGAMGInterface
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
        cyclicDIVGAMGInterface
        (
            const label index,
            const lduInterfacePtrsList& coarseInterfaces,
            Istream& is
        );

        //- Disallow default bitwise copy construction
        cyclicDIVGAMGInterface(const cyclicDIVGAMGInterface&) = delete;


    //- Destructor
    virtual ~cyclicDIVGAMGInterface();


    // Member Functions

        // Interface transfer functions

            //- Transfer and return internal field adjacent to the interface
            virtual tmp<labelField> internalFieldTransfer
            (
                const Pstream::commsTypes commsType,
                const labelUList& iF
            ) const;


        //- Cyclic interface functions

            //- Return neighbour processor number
            virtual label neighbPatchID() const
            {
                return neighbPatchID_;
            }

            virtual bool owner() const
            {
                return owner_;
            }

            virtual const cyclicDIVGAMGInterface& neighbPatch() const
            {
                return dynamic_cast<const cyclicDIVGAMGInterface&>
                (
                    coarseInterfaces_[neighbPatchID()]
                );
            }

            //- Return face transformation tensor
            virtual const tensorField& forwardT() const
            {
                return forwardT_;
            }

            //- Return neighbour-cell transformation tensor
            virtual const tensorField& reverseT() const
            {
                return reverseT_;
            }


        // I/O

            //- Write to stream
            virtual void write(Ostream&) const;


    // Member Operators

        //- Disallow default bitwise assignment
        void operator=(const cyclicDIVGAMGInterface&) = delete;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
