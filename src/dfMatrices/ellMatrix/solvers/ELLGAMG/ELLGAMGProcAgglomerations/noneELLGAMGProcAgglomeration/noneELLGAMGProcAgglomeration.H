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
    Foam::noneELLGAMGProcAgglomeration

Description
    Processor agglomeration of ELLGAMGAgglomerations.

SourceFiles
    noneELLGAMGProcAgglomeration.C

\*---------------------------------------------------------------------------*/

#ifndef noneELLGAMGProcAgglomeration_H
#define noneELLGAMGProcAgglomeration_H

#include "ELLGAMGProcAgglomeration.H"
#include "DynamicList.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

class ELLGAMGAgglomeration;

/*---------------------------------------------------------------------------*\
              Class noneELLGAMGProcAgglomeration Declaration
\*---------------------------------------------------------------------------*/

class noneELLGAMGProcAgglomeration
:
    public ELLGAMGProcAgglomeration
{

public:

    //- Runtime type information
    TypeName("noneELLGAMGProcAgglomeration");


    // Constructors

        //- Construct given agglomerator and controls
        noneELLGAMGProcAgglomeration
        (
            ELLGAMGAgglomeration& agglom,
            const dictionary& controlDict
        );

        //- Disallow default bitwise copy construction
        noneELLGAMGProcAgglomeration
        (
            const noneELLGAMGProcAgglomeration&
        );


    //- Destructor
    virtual ~noneELLGAMGProcAgglomeration();


    // Member Functions

       //- Modify agglomeration. Return true if modified
        virtual bool agglomerate()
        {
            return false;
        }


    // Member Operators

        //- Disallow default bitwise assignment
        void operator=(const noneELLGAMGProcAgglomeration&) = delete;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
