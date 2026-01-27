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
    Foam::nonedfGAMGProcAgglomeration

Description
    Processor agglomeration of dfGAMGAgglomerations.

SourceFiles
    nonedfGAMGProcAgglomeration.C

\*---------------------------------------------------------------------------*/

#ifndef nonedfGAMGProcAgglomeration_H
#define nonedfGAMGProcAgglomeration_H

#include "dfGAMGProcAgglomeration.H"
#include "DynamicList.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

class dfGAMGAgglomeration;

/*---------------------------------------------------------------------------*\
              Class nonedfGAMGProcAgglomeration Declaration
\*---------------------------------------------------------------------------*/

class nonedfGAMGProcAgglomeration
:
    public dfGAMGProcAgglomeration
{

public:

    //- Runtime type information
    TypeName("nonedfGAMGProcAgglomeration");


    // Constructors

        //- Construct given agglomerator and controls
        nonedfGAMGProcAgglomeration
        (
            dfGAMGAgglomeration& agglom,
            const dictionary& controlDict
        );

        //- Disallow default bitwise copy construction
        nonedfGAMGProcAgglomeration
        (
            const nonedfGAMGProcAgglomeration&
        );


    //- Destructor
    virtual ~nonedfGAMGProcAgglomeration();


    // Member Functions

       //- Modify agglomeration. Return true if modified
        virtual bool agglomerate()
        {
            return false;
        }


    // Member Operators

        //- Disallow default bitwise assignment
        void operator=(const nonedfGAMGProcAgglomeration&) = delete;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
