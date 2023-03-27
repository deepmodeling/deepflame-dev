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
    Foam::combustionModels::PaSR

Description
    Partially stirred reactor turbulent combustion model.

    This model calculates a finite rate, based on both turbulence and chemistry
    time scales.  Depending on mesh resolution, the Cmix parameter can be used
    to scale the turbulence mixing time scale.

SourceFiles
    PaSR.C

\*---------------------------------------------------------------------------*/

#ifndef PaSR_H
#define PaSR_H

#include "../laminar/laminar.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace combustionModels
{

/*---------------------------------------------------------------------------*\
                            Class PaSR Declaration
\*---------------------------------------------------------------------------*/

template<class ReactionThermo>
class PaSR
:
    public laminar<ReactionThermo>
{
    // Private Data
        
        //- thermo mu
        volScalarField& mu_;
        
        //- thermo T and p
        const volScalarField& p_;
        const volScalarField& T_;

        //- mixing scale dictionary
        dictionary mixingScaleDict_;

        //- chemistry scale dictionary
        dictionary chemistryScaleDict_;

        //- mixing scale type
        word mixingScaleType_;

        //- chemistry scale type
        word chemistryScaleType_;

        //- dictionary 
        dictionary mixingScaleCoeffs_;
        dictionary chemistryScaleCoeffs_;

        //- fuel 
        word fuel_;

        //- oxidizer
        word oxidizer_;

        //- global mixing constant
        scalar Cmix_;    

        //- stochastic mixture fraction
        scalar Zst_;

        //- scalar mixing constant
        scalar Cd1_;  
        scalar Cd2_;  
        scalar Cp1_; 
        scalar Cp2_;   
        scalar maxChi_;   

        //- laminar flame speed constant
        scalar W_;          
        scalar eta_; 
        scalar xi_;
        scalar alpha_; 
        scalar beta_;                 


        //- mixing scale time
        volScalarField tmix_;         

        //- chemistry scale time
        volScalarField tc_; 

        //- Da number
        volScalarField Da_;

        //- laminar flame speed  

        //- mixture facraction variance 
        volScalarField Z_;

        //- mixture facraction variance 
        volScalarField Zvar_;

        //- scalar dissipation rate 
        volScalarField Chi_;

        //- equivalence ratio
        volScalarField eqR_;

        //- laminar flame speed
        volScalarField Su_;

        //- Mixing parameter
        volScalarField kappa_;
  
        //-switch for solving transport chi
       // Switch  transportChi_;
        word ChiType_; 

        multivariateSurfaceInterpolationScheme<scalar>::fieldTable fields_;

public:

    //- Runtime type information
    TypeName("PaSR");


    // Constructors

        //- Construct from components
        PaSR
        (
            const word& modelType,
            ReactionThermo& thermo,
            const compressibleTurbulenceModel& turb,
            const word& combustionProperties
        );

        //- Disallow default bitwise copy construction
        PaSR(const PaSR&);


    //- Destructor
    virtual ~PaSR();


    // Member Functions

        //- Correct combustion rate
        virtual void correct();

        //- Fuel consumption rate matrix.
        virtual tmp<fvScalarMatrix> R(volScalarField& Y) const;

        //- Heat release rate [kg/m/s^3]
        virtual tmp<volScalarField> Qdot() const;

        //- Update properties from given dictionary
        virtual bool read();

        void transport();       


    // Member Operators

        //- Disallow default bitwise assignment
        void operator=(const PaSR&) = delete;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace combustionModels
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
    #include "PaSR.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
