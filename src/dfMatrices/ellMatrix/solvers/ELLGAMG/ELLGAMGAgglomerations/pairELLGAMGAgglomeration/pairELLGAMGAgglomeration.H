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
    Foam::pairELLGAMGAgglomeration

Description
    Agglomerate using the pair algorithm.

SourceFiles
    pairELLGAMGAgglomeration.C
    pairGAMGAgglomerate.C

\*---------------------------------------------------------------------------*/

#ifndef pairELLGAMGAgglomeration_H
#define pairELLGAMGAgglomeration_H

#include "ELLGAMGAgglomeration.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                           Class pairELLGAMGAgglomeration Declaration
\*---------------------------------------------------------------------------*/

class pairELLGAMGAgglomeration
:
    public ELLGAMGAgglomeration
{
    // Private Data

        //- Number of levels to merge, 1 = don't merge, 2 = merge pairs etc.
        label mergeLevels_;

        //- Direction of cell loop for the current level
        static bool forward_;


protected:

    // Protected Member Functions

        //- Agglomerate all levels starting from the given face weights
        void agglomerate
        (
            const lduMesh& mesh,
            const scalarField& faceWeights
        );


public:

    //- Runtime type information
    TypeName("pair");


    // Constructors

        //- Construct given mesh and controls
        pairELLGAMGAgglomeration
        (
            const lduMesh& mesh,
            const dictionary& controlDict
        );

        //- Disallow default bitwise copy construction
        pairELLGAMGAgglomeration(const pairELLGAMGAgglomeration&) = delete;

        //- Calculate and return agglomeration
        static tmp<labelField> agglomerate
        (
            label& nCoarseCells,
            const lduAddressing& fineMatrixAddressing,
            const scalarField& faceWeights
        );


    // Member Operators

        //- Disallow default bitwise assignment
        void operator=(const pairELLGAMGAgglomeration&) = delete;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
