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
    Foam::processorDIVGAMGInterfaceField

Description
    GAMG agglomerated processor interface field.

SourceFiles
    processorDIVGAMGInterfaceField.C

\*---------------------------------------------------------------------------*/

#ifndef processorDIVGAMGInterfaceField_H
#define processorDIVGAMGInterfaceField_H

#include "DIVGAMGInterfaceField.H"
#include "processorDIVGAMGInterface.H"
#include "processorLduInterfaceField.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                  Class processorDIVGAMGInterfaceField Declaration
\*---------------------------------------------------------------------------*/

class processorDIVGAMGInterfaceField
:
    public DIVGAMGInterfaceField,
    public processorLduInterfaceField
{
    // Private Data

        //- Local reference cast into the processor interface
        const processorDIVGAMGInterface& procInterface_;

        //- Is the transform required
        bool doTransform_;

        //- Rank of component for transformation
        int rank_;


        // Sending and receiving

            //- Outstanding request
            mutable label outstandingSendRequest_;

            //- Outstanding request
            mutable label outstandingRecvRequest_;

            //- Scalar send buffer
            mutable Field<scalar> scalarSendBuf_;

            //- Scalar receive buffer
            mutable Field<scalar> scalarReceiveBuf_;


public:

    //- Runtime type information
    TypeName("processor");


    // Constructors

        //- Construct from GAMG interface and fine level interface field
        processorDIVGAMGInterfaceField
        (
            const DIVGAMGInterface& GAMGCp,
            const lduInterfaceField& fineInterface
        );

        //- Construct from GAMG interface and fine level interface field
        processorDIVGAMGInterfaceField
        (
            const DIVGAMGInterface& GAMGCp,
            const bool doTransform,
            const int rank
        );

        //- Disallow default bitwise copy construction
        processorDIVGAMGInterfaceField
        (
            const processorDIVGAMGInterfaceField&
        ) = delete;


    //- Destructor
    virtual ~processorDIVGAMGInterfaceField();


    // Member Functions

        // Access

            //- Return size
            label size() const
            {
                return procInterface_.size();
            }


        // Interface matrix update

            //- Initialise neighbour matrix update
            virtual void initInterfaceMatrixUpdate
            (
                scalarField& result,
                const scalarField& psiInternal,
                const scalarField& coeffs,
                const direction cmpt,
                const Pstream::commsTypes commsType
            ) const;

            //- Update result field based on interface functionality
            virtual void updateInterfaceMatrix
            (
                scalarField& result,
                const scalarField& psiInternal,
                const scalarField& coeffs,
                const direction cmpt,
                const Pstream::commsTypes commsType
            ) const;


        //- Processor interface functions

            //- Return communicator used for comms
            virtual label comm() const
            {
                return procInterface_.comm();
            }

            //- Return processor number
            virtual int myProcNo() const
            {
                return procInterface_.myProcNo();
            }

            //- Return neighbour processor number
            virtual int neighbProcNo() const
            {
                return procInterface_.neighbProcNo();
            }

            //- Does the interface field perform the transformation
            virtual bool doTransform() const
            {
                return doTransform_;
            }

            //- Return face transformation tensor
            virtual const tensorField& forwardT() const
            {
                return procInterface_.forwardT();
            }

            //- Return rank of component for transform
            virtual int rank() const
            {
                return rank_;
            }

            //- Return the processor interface
            virtual const processorDIVGAMGInterface& procInterface() const
            {
                return procInterface_;
            }


    // Member Operators

        //- Disallow default bitwise assignment
        void operator=(const processorDIVGAMGInterfaceField&) = delete;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
