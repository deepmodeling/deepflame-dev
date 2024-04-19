/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2013-2019 OpenFOAM Foundation
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
    Foam::CSRGAMGProcAgglomeration

Description
    Processor agglomeration of CSRGAMGAgglomerations.

SourceFiles
    CSRGAMGProcAgglomeration.C

\*---------------------------------------------------------------------------*/

#ifndef CSRGAMGProcAgglomeration_H
#define CSRGAMGProcAgglomeration_H

#include "runTimeSelectionTables.H"
#include "labelList.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

class CSRGAMGAgglomeration;
class lduMesh;

/*---------------------------------------------------------------------------*\
                    Class CSRGAMGProcAgglomeration Declaration
\*---------------------------------------------------------------------------*/

class CSRGAMGProcAgglomeration
{

protected:

    // Protected data

        //- Reference to agglomeration
        CSRGAMGAgglomeration& agglom_;

    // Protected Member Functions

        //- Debug: write agglomeration info
        void printStats(Ostream& os, CSRGAMGAgglomeration& agglom) const;

        //- Agglomerate a level. Return true if anything has changed
        bool agglomerate
        (
            const label fineLevelIndex,
            const labelList& procAgglomMap,
            const labelList& masterProcs,
            const List<label>& agglomProcIDs,
            const label procAgglomComm
        );

        //- Debug: calculate global cell-cells
        static labelListList globalCellCells(const lduMesh&);


public:

    //- Runtime type information
    TypeName("CSRGAMGProcAgglomeration");


    // Declare run-time constructor selection tables

        //- Runtime selection table for pure geometric agglomerators
        declareRunTimeSelectionTable
        (
            autoPtr,
            CSRGAMGProcAgglomeration,
            CSRGAMGAgglomeration,
            (
                CSRGAMGAgglomeration& agglom,
                const dictionary& controlDict
            ),
            (
                agglom,
                controlDict
            )
        );


    // Constructors

        //- Construct given agglomerator and controls
        CSRGAMGProcAgglomeration
        (
            CSRGAMGAgglomeration& agglom,
            const dictionary& controlDict
        );

        //- Disallow default bitwise copy construction
        CSRGAMGProcAgglomeration(const CSRGAMGProcAgglomeration&) = delete;


    // Selectors

        //- Return the selected agglomerator
        static autoPtr<CSRGAMGProcAgglomeration> New
        (
            const word& type,
            CSRGAMGAgglomeration& agglom,
            const dictionary& controlDict
        );


    //- Destructor
    virtual ~CSRGAMGProcAgglomeration();


    // Member Functions

        //- Modify agglomeration. Return true if modified
        virtual bool agglomerate() = 0;


    // Member Operators

        //- Disallow default bitwise assignment
        void operator=(const CSRGAMGProcAgglomeration&) = delete;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
