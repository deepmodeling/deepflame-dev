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
    Foam::combustionModels::baseFGM

Description
    Partially stirred reactor turbulent combustion model.

    This model calculates a finite rate, based on both turbulence and chemistry
    time scales.  Depending on mesh resolution, the Cmix parameter can be used
    to scale the turbulence mixing time scale.

SourceFiles
    baseFGM.C

\*---------------------------------------------------------------------------*/

#ifndef baseFGM_H
#define baseFGM_H

#include "../laminar/laminar.H"
#include "uniformDimensionedFields.H"
#include "CanteraMixture.H"
// #include "fvOptions.H"
#include "LESfilter.H"
//#include "basicThermo.H"
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace combustionModels
{

/*---------------------------------------------------------------------------*\
                            Class baseFGM Declaration
\*---------------------------------------------------------------------------*/

template<class ReactionThermo>
class baseFGM
:
    public laminar<ReactionThermo>
{
    // Protected Data
 protected:
    // fv::options& fvOptions; 
    //- buffer switch
    Switch buffer_;

    //- scaled PV switch
    Switch scaledPV_;

    //- incompressible pressure reference
    scalar incompPref_;

    //- ignition switch
    Switch ignition_;

    //- combustion switch
    Switch combustion_;

    //- enthalpy solve witch
    Switch solveEnthalpy_;
    
    //- T-lookup switch
    Switch flameletT_;

    word tablePath_;

    //- psi=\rho/p
    volScalarField& psi_;

    //- air mass/mole(28.96kg/mol)
    volScalarField Wt_;
    
    //- heat capacity(m^2*s^{-2}*K^{-1}) 
    volScalarField Cp_;

    //- mixture fraction Z
    volScalarField Z_;

    //- mixture fraction variance Zvar
    volScalarField Zvar_;    
    
    //- enthalpy 
    volScalarField He_;

    //- formation enthalpy(m^2*s^{-2})
    volScalarField Hf_;

    //- progress variable c
    volScalarField c_;

    //- progress variable c variance
    volScalarField cvar_;

    //- co-variance Z, c
    volScalarField Zcvar_;

    //- chemistry source reaction rate(kg*m^{-3}*s^{-1})
    volScalarField omega_c_;

    //- progress variable dissipation rate (s^{-1})
    volScalarField chi_c_;

    //- mixture fraction dissipation rate (s^{-1})
    volScalarField chi_Z_; 

    //-covariance Z-c dissipation rate (s^{-1})
    volScalarField chi_Zc_;

    //- filtered SDR of mixture fraction
    volScalarField chi_Zfltd_; 

    //- betac for LES
    volScalarField betac_;

    volScalarField He_s_;
    volScalarField Zvar_s_; 
    volScalarField c_s_;
    volScalarField cvar_s_;
    volScalarField Zcvar_s_;

    //- copy of omega_c_
    volScalarField cOmega_c_;
    volScalarField ZOmega_c_;


    //- internal fields
    scalarField& WtCells_; 
    scalarField& CpCells_;
    scalarField& ZCells_;
    scalarField& ZvarCells_;
    scalarField& HCells_;
    scalarField& HfCells_;
    scalarField& cCells_;
    scalarField& cvarCells_;
    scalarField& ZcvarCells_;
    scalarField& omega_cCells_;
    scalarField& cOmega_cCells_;
    scalarField& ZOmega_cCells_;
    scalarField& chi_cCells_;
    scalarField& chi_ZCells_;
    scalarField& chi_ZcCells_;

    //- filter size
    scalarField deltaCells_; 

    scalarField& chi_ZfltdCells_;

    scalarField& betacCells_; 

    //- max and min Z
    scalar ZMax_;
    scalar ZMin_;

    //- max and min Z variance
    scalar ZvarMax_;
    scalar ZvarMin_;

    //- max and min c
    scalar cMax_;
    scalar cMin_;
    scalar Ycmaxall_;

    //- max and min c variance
    scalar cvarMax_;
    scalar cvarMin_;

    //- max and min Zc variance 
    scalar ZcvarMax_;
    scalar ZcvarMin_;   


    //- fluid density
     volScalarField& rho_;
    //- fluid density
     volScalarField& rho_inRhoThermo_;

    //- fluid pressure
    const volScalarField& p_;

    //- fluid temperature
    volScalarField& T_;

    PtrList<volScalarField>& Y_;

    //- fluid velocity
    const volVectorField& U_;     

    // pressure derivate
    const volScalarField& dpdt_;

    //- fluid flux
    const surfaceScalarField& phi_;

    scalarField& TCells_;    

    //- ignition begin time
    scalar ignBeginTime_;

    //- ignition duration time
    scalar ignDurationTime_;    
 
    //- reactFlowTime_=currentTime - ignBeginTime_
    scalar reactFlowTime_;

    //- igntion origin point
    scalar x0_;
    scalar y0_;
    scalar z0_;

    //- ignition radius
    scalar R0_;

    //- turbulent schmidt number
    scalar Sct_;

    //- laminar schmidt number
    scalar Sc_;

    scalar bufferTime_;
 
    //- relaxation switch
    Switch relaxation_;

    //- Dpdt switch 
    Switch DpDt_;    

    multivariateSurfaceInterpolationScheme<scalar>::fieldTable fields_;

    //- LES
    //- if in LES mode
    Switch isLES_;

    //SGS
    volScalarField magUPrime_;
    scalarField& magUPrimeCells_; 

 public:
    //- Runtime type information
    TypeName("baseFGM");

    // Constructors

        //- Construct from components
        baseFGM
        (
            const word& modelType,
            ReactionThermo& thermo,
            const compressibleTurbulenceModel& turb,
            const word& combustionProperties
        );


    //- Destructor
    virtual ~baseFGM();


    // Member Functions

        //- transport equation 
        virtual void transport();

        //- initialize flame kernel 
        virtual void initialiseFalmeKernel();        

        //- return mag SGS
        virtual void magUPrime();

        //- reaturn buffer_
        virtual bool buffer();

};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace combustionModels
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
    #include "baseFGM.C"
#endif


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
