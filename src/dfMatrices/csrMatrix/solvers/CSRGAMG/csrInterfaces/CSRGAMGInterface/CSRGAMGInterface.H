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
    Foam::CSRGAMGInterface

Description
     Abstract base class for GAMG agglomerated interfaces.

SourceFiles
    CSRGAMGInterface.C
    newAmgInterface.C

\*---------------------------------------------------------------------------*/

#ifndef CSRGAMGInterface_H
#define CSRGAMGInterface_H

#include "autoPtr.H"
#include "lduInterfacePtrsList.H"
#include "CSRGAMGAgglomeration.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                      Class CSRGAMGInterface Declaration
\*---------------------------------------------------------------------------*/

class CSRGAMGInterface
:
    public lduInterface
{

protected:

    // Protected data

        //- My index in coarseInterfaces
        const label index_;

        //- All interfaces
        const lduInterfacePtrsList& coarseInterfaces_;

        //- Face-cell addressing
        labelList faceCells_;

        //- Face restrict addressing
        labelList faceRestrictAddressing_;


public:

    //- Runtime type information
    TypeName("CSRGAMGInterface");


    // Declare run-time constructor selection tables

        declareRunTimeSelectionTable
        (
            autoPtr,
            CSRGAMGInterface,
            lduInterface,
            (
                const label index,
                const lduInterfacePtrsList& coarseInterfaces,
                const lduInterface& fineInterface,
                const labelField& localRestrictAddressing,
                const labelField& neighbourRestrictAddressing,
                const label fineLevelIndex,
                const label coarseComm
            ),
            (
                index,
                coarseInterfaces,
                fineInterface,
                localRestrictAddressing,
                neighbourRestrictAddressing,
                fineLevelIndex,
                coarseComm
            )
        );

        declareRunTimeSelectionTable
        (
            autoPtr,
            CSRGAMGInterface,
            Istream,
            (
                const label index,
                const lduInterfacePtrsList& coarseInterfaces,
                Istream& is
            ),
            (
                index,
                coarseInterfaces,
                is
            )
        );


    // Selectors

        //- Return a pointer to a new interface created on freestore given
        //  the fine interface
        static autoPtr<CSRGAMGInterface> New
        (
            const label index,
            const lduInterfacePtrsList& coarseInterfaces,
            const lduInterface& fineInterface,
            const labelField& localRestrictAddressing,
            const labelField& neighbourRestrictAddressing,
            const label fineLevelIndex,
            const label coarseComm
        );

        //- Return a pointer to a new interface created on freestore given
        //  the fine interface
        static autoPtr<CSRGAMGInterface> New
        (
            const word& coupleType,
            const label index,
            const lduInterfacePtrsList& coarseInterfaces,
            Istream& is
        );


    // Constructors

        //- Construct from interfaces, restrict addressing set later on
        CSRGAMGInterface
        (
            const label index,
            const lduInterfacePtrsList& coarseInterfaces
        )
        :
            index_(index),
            coarseInterfaces_(coarseInterfaces)
        {}


        //- Construct from interfaces and restrict addressing
        CSRGAMGInterface
        (
            const label index,
            const lduInterfacePtrsList& coarseInterfaces,
            const labelUList& faceCells,
            const labelUList& faceRestrictAddressing
        )
        :
            index_(index),
            coarseInterfaces_(coarseInterfaces),
            faceCells_(faceCells),
            faceRestrictAddressing_(faceRestrictAddressing)
        {}


        //- Construct from Istream
        CSRGAMGInterface
        (
            const label index,
            const lduInterfacePtrsList& coarseInterfaces,
            Istream& is
        );

        //- Disallow default bitwise copy construction
        CSRGAMGInterface(const CSRGAMGInterface&) = delete;


    // Member Functions

        // Access

            //- Return size
            virtual label size() const
            {
                return faceCells_.size();
            }

            virtual label index() const
            {
                return index_;
            }

            virtual const lduInterfacePtrsList& coarseInterfaces() const
            {
                return coarseInterfaces_;
            }

            //- Return faceCell addressing
            virtual const labelUList& faceCells() const
            {
                return faceCells_;
            }

            //- Return (local)face restrict addressing
            virtual const labelList& faceRestrictAddressing() const
            {
                return faceRestrictAddressing_;
            }

            //- Return non-const access to face restrict addressing
            virtual labelList& faceRestrictAddressing()
            {
                return faceRestrictAddressing_;
            }

            //- Return the interface internal field of the given field
            template<class Type>
            tmp<Field<Type>> interfaceInternalField
            (
                const UList<Type>& internalData
            ) const;

            //- Get the interface internal field of the given field
            template<class Type>
            void interfaceInternalField
            (
                const UList<Type>& internalData,
                List<Type>&
            ) const;

            //- Return the values of the given internal data adjacent to
            //  the interface as a field
            virtual tmp<labelField> interfaceInternalField
            (
                const labelUList& internalData
            ) const;


        // Agglomeration

            //- Merge the next level with this level
            //  combining the face-restrict addressing
            //  and copying the face-cell addressing
            void combine(const CSRGAMGInterface&);

            //- Agglomerating the given fine-level coefficients and return
            virtual tmp<scalarField> agglomerateCoeffs
            (
                const scalarField& fineCoeffs
            ) const;


        // I/O

            //- Write to stream
            virtual void write(Ostream&) const = 0;


    // Member Operators

        //- Disallow default bitwise assignment
        void operator=(const CSRGAMGInterface&) = delete;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
    #include "CSRGAMGInterfaceTemplates.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
