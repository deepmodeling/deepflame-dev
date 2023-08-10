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

\*---------------------------------------------------------------------------*/

#include "dfRefinementHistoryConstraint.H"
#include "addToRunTimeSelectionTable.H"
#include "syncTools.H"
#include "dfRefinementHistory.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace decompositionConstraints
{
    defineTypeName(dfRefinementHistoryConstraint);

    addToRunTimeSelectionTable
    (
        decompositionConstraint,
        dfRefinementHistoryConstraint,
        dictionary
    );
}
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::decompositionConstraints::dfRefinementHistoryConstraint::dfRefinementHistoryConstraint
(
    const dictionary& dict//,
    //const word& modelType
)
:
    decompositionConstraint(dict, typeName)
{
    if (decompositionConstraint::debug)
    {
        Info<< type()
            << " : setting constraints to refinement history New" << endl;
    }
}


Foam::decompositionConstraints::dfRefinementHistoryConstraint::dfRefinementHistoryConstraint()
:
    decompositionConstraint(dictionary(), typeName)
{
    if (decompositionConstraint::debug)
    {
        Info<< type()
            << " : setting constraints to refinement history New" << endl;
    }
}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

void Foam::decompositionConstraints::dfRefinementHistoryConstraint::add
(
    const polyMesh& mesh,
    boolList& blockedFace,
    PtrList<labelList>& specifiedProcessorFaces,
    labelList& specifiedProcessor,
    List<labelPair>& explicitConnections
) const
{
    // The refinement history type
    typedef ::Foam::dfRefinementHistory HistoryType;
    // typedef Foam::decompositionConstraints::dfRefinementHistory HistoryType;

    // Local storage if read from file
    autoPtr<const HistoryType> readFromFile;

    const HistoryType* historyPtr = nullptr;

    if (mesh.foundObject<HistoryType>("dfRefinementHistory"))
    {
        if (decompositionConstraint::debug)
        {
            Info<< type() << " : found dfRefinementHistory" << endl;
        }
        historyPtr = &mesh.lookupObject<HistoryType>("dfRefinementHistory");
    }
    else
    {
        if (decompositionConstraint::debug)
        {
            Info<< type() << " : reading dfRefinementHistory from time "
                << mesh.facesInstance() << endl;
        }

        readFromFile.reset
        (
            new HistoryType
            (
                IOobject
                (
                    "dfRefinementHistory",
                    mesh.facesInstance(),
                    polyMesh::meshSubDir,
                    mesh,
                    IOobject::READ_IF_PRESENT,
                    IOobject::NO_WRITE
                ),
                mesh.nCells()
            )
        );

        // historyPtr = readFromFile.get();  // get(), not release()
    }

    // const auto& history = *historyPtr;
    const auto& history =
    (
        readFromFile.valid()
       ? readFromFile()
       : *historyPtr
    );

    if (history.active())
    {
        // dfRefinementHistory itself implements decompositionConstraint
        history.add
        (
            blockedFace,
            specifiedProcessorFaces,
            specifiedProcessor,
            explicitConnections
        );
    }
}


void Foam::decompositionConstraints::dfRefinementHistoryConstraint::apply
(
    const polyMesh& mesh,
    const boolList& blockedFace,
    const PtrList<labelList>& specifiedProcessorFaces,
    const labelList& specifiedProcessor,
    const List<labelPair>& explicitConnections,
    labelList& decomposition
) const
{
    // The refinement history type
    typedef ::Foam::dfRefinementHistory HistoryType;

    // Local storage if read from file
    autoPtr<const HistoryType> readFromFile;

    const HistoryType* historyPtr = nullptr;

    if(mesh.foundObject<HistoryType>("dfRefinementHistory"))
    {
        historyPtr = &mesh.lookupObject<HistoryType>("dfRefinementHistory");
    }
    else
    {
        readFromFile.reset
        (
            new HistoryType
            (
                IOobject
                (
                    "dfRefinementHistory",
                    mesh.facesInstance(),
                    polyMesh::meshSubDir,
                    mesh,
                    IOobject::READ_IF_PRESENT,
                    IOobject::NO_WRITE
                ),
                mesh.nCells()
            )
        );

        // historyPtr = readFromFile.get();  // get(), not release()
    }

    // const auto& history = *historyPtr;
    const auto& history =
    (
        readFromFile.valid()
       ? readFromFile()
       : *historyPtr
    );

    if (history.active())
    {
        // dfRefinementHistory itself implements decompositionConstraint
        history.apply
        (
            blockedFace,
            specifiedProcessorFaces,
            specifiedProcessor,
            explicitConnections,
            decomposition
        );
    }
}


// ************************************************************************* //