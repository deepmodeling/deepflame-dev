/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2015-2016 OpenFOAM Foundation
     \\/     M anipulation  | Copyright (C) 2018 OpenCFD Ltd.
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
    Foam::decompositionConstraints::dfRefinementHistoryConstraint

Description
    Constraint to keep all cells originating from refining the same cell
    onto the same processor. Reads polyMesh/dfRefinementHistory.

    \heading Dictionary parameters
    \table
        Property    | Description                       | Required  | Default
        type        | dfRefinementHistory                 | yes   |
    \endtable

SourceFiles
    dfRefinementHistoryConstraint.C

Note
    original code:
    HenningScheufler, URL:https://github.com/HenningScheufler/multiDimAMR, 2020
    Changes:
        + 2022-Oct: Modify names to apply AMR in DeepFlame
        + 2022-Oct: Adjust details in functions to make the code available for OpenFOAM v7

\*---------------------------------------------------------------------------*/

#ifndef dfRefinementHistoryConstraint_H
#define dfRefinementHistoryConstraint_H

#include "decompositionConstraint.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace decompositionConstraints
{

/*---------------------------------------------------------------------------*\
                      Class dfRefinementHistoryConstraint Declaration
\*---------------------------------------------------------------------------*/

class dfRefinementHistoryConstraint
:
    public decompositionConstraint
{
public:

    //- Runtime type information
    TypeName("dfRefinementHistory");


    // Constructors

        //- Construct with constraint dictionary
        explicit dfRefinementHistoryConstraint(const dictionary& dict,const word& type);

        //- Construct from components
        dfRefinementHistoryConstraint();


    //- Destructor
    virtual ~dfRefinementHistoryConstraint() = default;


    // Member Functions

        //- Add this constraint to list of constraints
        virtual void add
        (
            const polyMesh& mesh,
            boolList& blockedFace,
            PtrList<labelList>& specifiedProcessorFaces,
            labelList& specifiedProcessor,
            List<labelPair>& explicitConnections
        ) const;

        //- Apply this constraint post-decomposition
        virtual void apply
        (
            const polyMesh& mesh,
            const boolList& blockedFace,
            const PtrList<labelList>& specifiedProcessorFaces,
            const labelList& specifiedProcessor,
            const List<labelPair>& explicitConnections,
            labelList& decomposition
        ) const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace decompositionConstraints
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //