/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2018 OpenFOAM Foundation
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
    Foam::faceAreaPairLDUGAMGAgglomeration

Description
    Agglomerate using the pair algorithm.

SourceFiles
    faceAreaPairLDUGAMGAgglomeration.C

\*---------------------------------------------------------------------------*/

#ifndef faceAreaPairLDUGAMGAgglomeration_H
#define faceAreaPairLDUGAMGAgglomeration_H

#include "pairLDUGAMGAgglomeration.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                 Class faceAreaPairLDUGAMGAgglomeration Declaration
\*---------------------------------------------------------------------------*/

class faceAreaPairLDUGAMGAgglomeration
:
    public pairLDUGAMGAgglomeration
{

public:

    //- Runtime type information
    TypeName("faceAreaPair");


    // Constructors

        //- Construct given mesh and controls
        faceAreaPairLDUGAMGAgglomeration
        (
            const lduMesh& mesh,
            const dictionary& controlDict
        );

        //- Construct given mesh and controls
        faceAreaPairLDUGAMGAgglomeration
        (
            const lduMesh& mesh,
            const scalarField& cellVolumes,
            const vectorField& faceAreas,
            const dictionary& controlDict
        );
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
