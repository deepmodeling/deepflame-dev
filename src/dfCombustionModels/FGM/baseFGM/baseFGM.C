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

\*---------------------------------------------------------------------------*/

#include "baseFGM.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class ReactionThermo>
Foam::combustionModels::baseFGM<ReactionThermo>::baseFGM
(
    const word& modelType,
    ReactionThermo& thermo,
    const compressibleTurbulenceModel& turb,
    const word& combustionProperties
)
:
    laminar<ReactionThermo>(modelType, thermo, turb, combustionProperties),
    buffer_(this->coeffs().lookupOrDefault("buffer", false)),
    scaledPV_(this->coeffs().lookupOrDefault("scaledPV", false)),
    incompPref_(this->coeffs().lookupOrDefault("incompPref", -10.0)),
    ignition_(this->coeffs().lookupOrDefault("ignition", false)),
    combustion_(this->coeffs().lookupOrDefault("combustion", false)),
    solveEnthalpy_(this->coeffs().lookupOrDefault("solveEnthalpy", false)),
    flameletT_(this->coeffs().lookupOrDefault("flameletT", false)),
    psi_(const_cast<volScalarField&>(dynamic_cast<psiThermo&>(thermo).psi())),
    Wt_ 
    (
        IOobject
        (
            "Wt",
            this->mesh().time().timeName(),
            this->mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh(),
        dimensionedScalar("Wt",dimensionSet(1,0,0,0,-1,0,0),28.96)  
    ),
    Cp_ 
    (
        IOobject
        (
            "Cp",
            this->mesh().time().timeName(),
            this->mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh(),
        dimensionedScalar("Cp",dimensionSet(0,2,-2,-1,0,0,0),1010.1)
    ),
    Z_
    (
        IOobject
        (
            "Z",
            this->mesh().time().timeName(),
            this->mesh(),
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh()
    ),
    Zvar_
    (
        IOobject
        (
            "Zvar",
            this->mesh().time().timeName(),
            this->mesh(),
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh()
    ),
    He_
    (
        IOobject
        (
            "He",
            this->mesh().time().timeName(),
            this->mesh(),
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        this->mesh(),
        dimensionedScalar("He",dimensionSet(0,2,-2,0,0,0,0),0.0)
    ),
    Hf_
    (
        IOobject
        (
            "Hf",
            this->mesh().time().timeName(),
            this->mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh(),
        dimensionedScalar("Hf",dimensionSet(0,2,-2,0,0,0,0),1907.0)
    ),
    c_
    (
        IOobject
        (
            "c",
            this->mesh().time().timeName(),
            this->mesh(),
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh()
    ), 
    cvar_
    (
        IOobject
        (
            "cvar",
            this->mesh().time().timeName(),
            this->mesh(),
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh()
    ),    
    Zcvar_
    (
        IOobject
        (
            "Zcvar",
            this->mesh().time().timeName(),
            this->mesh(),
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh()
    ),
    omega_c_ 
    (
        IOobject
        (
            "omega_c",
            this->mesh().time().timeName(),
            this->mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh(),
        dimensionedScalar("omega_c",dimensionSet(1,-3,-1,0,0,0,0),0.0)
    ),
    chi_c_
    (
        IOobject
        (
            "chi_c",
            this->mesh().time().timeName(),
            this->mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh(),  
        dimensionedScalar("chi_c",dimensionSet(0,0,-1,0,0,0,0),0.0)
    ),    
    chi_Z_
    (
        IOobject
        (
            "chi_Z",
            this->mesh().time().timeName(),
            this->mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh(),  
        dimensionedScalar("chi_Z",dimensionSet(0,0,-1,0,0,0,0),0.0) 
    ),    
    chi_Zc_
    (
        IOobject
        (
            "chi_Zc",
            this->mesh().time().timeName(),
            this->mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh(),  
         dimensionedScalar("chi_Zc",dimensionSet(0,0,-1,0,0,0,0),0.0)
    ),
    YCO2_  
    (
        IOobject
        (
            "YCO2",
            this->mesh().time().timeName(),
            this->mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh(),  
         dimensionedScalar("YCO2",dimless,0.0)  
    ),
    YCO_  
    (
        IOobject
        (
            "YCO",
            this->mesh().time().timeName(),
            this->mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh(),  
        dimensionedScalar("YCO",dimless,0.0)  
    ),
    YH2O_  
    (
        IOobject
        (
            "YH2O",
            this->mesh().time().timeName(),
            this->mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh(),  
        dimensionedScalar("YH2O",dimless,0.0) 
    ),
    speciesNames_(this->coeffs().lookup("speciesName")),
    Y_(speciesNames_.size()),
    cOmega_c_(omega_c_),
    ZOmega_c_(omega_c_), 
    WtCells_ (Wt_.primitiveFieldRef()),
    CpCells_ (Cp_.primitiveFieldRef()),
    ZCells_(Z_.primitiveFieldRef()),
    ZvarCells_(Zvar_.primitiveFieldRef()), 
    HCells_(He_.primitiveFieldRef()),
    HfCells_(Hf_.primitiveFieldRef()),
    cCells_(c_.primitiveFieldRef()),
    cvarCells_(cvar_.primitiveFieldRef()),
    ZcvarCells_(Zcvar_.primitiveFieldRef()), 
    omega_cCells_(omega_c_.primitiveFieldRef()), 
    cOmega_cCells_(cOmega_c_.primitiveFieldRef()),
    ZOmega_cCells_(ZOmega_c_.primitiveFieldRef()),
    chi_cCells_(chi_c_.primitiveFieldRef()),
    chi_ZCells_(chi_Z_.primitiveFieldRef()),
    chi_ZcCells_(chi_Zc_.primitiveFieldRef()), 
    YCO2Cells_(YCO2_.primitiveFieldRef()),     
    YCOCells_(YCO_.primitiveFieldRef()),   
    YH2OCells_(YH2O_.primitiveFieldRef()),        
    ZMax_(1.0),
    ZMin_(0.0),
    ZvarMax_(0.25),
    ZvarMin_(0.0),
    cMax_(1.0),
    cMin_(0.0),
    Ycmaxall_(1.0),
    cvarMax_(0.25),
    cvarMin_(0.25),
    ZcvarMax_(0.25),
    ZcvarMin_(-0.25),
    rho_(const_cast<volScalarField&>(this->mesh().objectRegistry::lookupObject<volScalarField>("rho"))),
    p_(this->thermo().p()),
    T_(this->thermo().T()),
    U_(this->mesh().objectRegistry::lookupObject<volVectorField>("U")),
    dpdt_(this->mesh().objectRegistry::lookupObject<volScalarField>("dpdt")),        
    phi_(this->mesh().objectRegistry::lookupObject<surfaceScalarField>("phi")),
    TCells_(T_.primitiveFieldRef()),
    ignBeginTime_(this->coeffs().lookupOrDefault("ignBeginTime", GREAT)),  
    ignDurationTime_(this->coeffs().lookupOrDefault("ignBeginTime", 0.0)),
    reactFlowTime_(0.0),
    x0_(this->coeffs().lookupOrDefault("x0", 0.0)),   
    y0_(this->coeffs().lookupOrDefault("y0", 0.0)), 
    z0_(this->coeffs().lookupOrDefault("z0", 0.05)),  
    R0_(this->coeffs().lookupOrDefault("R0", 0.04)),
    Sct_(this->coeffs().lookupOrDefault("Sct", 0.7)),
    Sc_(this->coeffs().lookupOrDefault("Sc", 1.0)),   
    bufferTime_(this->coeffs().lookupOrDefault("bufferTime", 0.0)),
    relaxation_(this->coeffs().lookupOrDefault("relaxation", false)),
    DpDt_(this->coeffs().lookupOrDefault("DpDt", false))
{
    if(incompPref_ < 0.0)  //local pressure used to calculate EOS
    {
      Info<< "Equation of State: local pressure used" << nl << endl;
    }
    else //constant pressure used to calculate EOS
    {
      Info<< "Equation of State: constant pressure used: "
        << incompPref_ << " Pa" << nl << endl;
    }

    //initialize species fields
    forAll(Y_, speciesI)
    {

     Y_.set
     (
      speciesI,
        new volScalarField
        (
            IOobject
            (
                "Y_"  + speciesNames_[speciesI],
                this->mesh().time().timeName(),
                this->mesh(),
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            this->mesh(),
            dimensionedScalar(dimless, Zero)
        )
     );

   }

   // add fields
     fields_.add(Z_);
     fields_.add(Zvar_);
     fields_.add(c_);     
     fields_.add(cvar_);  
     fields_.add(Zcvar_);  
     fields_.add(He_);  

}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

template<class ReactionThermo>
Foam::combustionModels::baseFGM<ReactionThermo>::~baseFGM()
{}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

template<class ReactionThermo>
void Foam::combustionModels::baseFGM<ReactionThermo>::transport()
{
    buffer();

    tmp<volScalarField> tmut(this->turbulence().mut());
    const volScalarField& mut = tmut();

    tmp<volScalarField> tmu(this->turbulence().mu());
    const volScalarField& mu = tmu();

    //scalarUW used for cEqn, cvarEqn, ZEqn, ZvarEqn,ZcvarEqn to ensure the convergence when buffer_ is true
    tmp<fv::convectionScheme<scalar> > scalarUWConvection   
    (
        fv::convectionScheme<scalar>::New
        (
            this->mesh(),
            fields_,
            phi_,
            this->mesh().divScheme("div(phi,scalarUW)") 
        )
    );


    // Solve the mixture fraction transport equation
    if(buffer_) 
    {
        Info<< "UW schemes used for scalars" << endl;

        fvScalarMatrix ZEqn
        (
            fvm::ddt(rho_,Z_)
            +scalarUWConvection->fvmDiv(phi_, Z_)  
            -fvm::laplacian( mut/Sct_ + mu/Sc_ , Z_)     
        );
        if(relaxation_)
        {
            ZEqn.relax();
        }
        ZEqn.solve();
    }

    else
    {
        Info<< "TVD schemes used for scalars" << endl;

        fvScalarMatrix ZEqn
        (
            fvm::ddt(rho_,Z_)
            +fvm::div(phi_, Z_)
            -fvm::laplacian(mut/Sct_ + mu/Sc_ , Z_) 
        );

        if(relaxation_)
        {
            ZEqn.relax();
        }
        ZEqn.solve();
    }
    Z_.min(ZMax_);  
    Z_.max(ZMin_);

   // solve the total enthalpy transport equation
    if(solveEnthalpy_)
    {
        fvScalarMatrix HEqn
        (
            fvm::ddt(rho_,He_)
            +(
                buffer_
                ? scalarUWConvection->fvmDiv(phi_, He_)
                : fvm::div(phi_, He_)
            )
            +(
                DpDt_
                ? - dpdt_ - ( U_ & fvc::grad(p_) ) - fvm::laplacian( mut/Sct_ + mu/Sc_, He_)
                : - fvm::laplacian( mut/Sct_ + mu/Sc_, He_) 
            )
        ); 

        if(relaxation_)
        {
            HEqn.relax();
        }
        HEqn.solve();
    }


    // Solve the mixture fraction variance transport equation

    fvScalarMatrix ZvarEqn
    (
        fvm::ddt(rho_,Zvar_)
        +(
            buffer_
            ?  scalarUWConvection->fvmDiv(phi_, Zvar_)
            :  fvm::div(phi_, Zvar_)
        )
        -fvm::laplacian( mut/Sct_+mu/Sc_, Zvar_)
        -(2.0*mut/Sct_*(fvc::grad(Z_) & fvc::grad(Z_)))
        +(2.0*rho_*chi_Z_)  
    );

    if(relaxation_)
    {
        ZvarEqn.relax();
    }

    ZvarEqn.solve();

    Zvar_.min(ZvarMax_);
    Zvar_.max(ZvarMin_);    


    if(combustion_ && reactFlowTime_ > 0.0)
    {
        // At initial time, cMax is set as Ycmaxall when unscaled PV employed
        if(this->mesh().time().timeIndex() == 1 && !scaledPV_) cMax_ = Ycmaxall_;  

        // Solve the progress variable transport equation
        fvScalarMatrix cEqn
        (
            fvm::ddt(rho_, c_)
            +(
                buffer_
                ? scalarUWConvection->fvmDiv(phi_, c_)
                : fvm::div(phi_, c_)
            )
            -fvm::laplacian( mut/Sct_ + mu/Sc_, c_)
            -omega_c_
        );  
        if(relaxation_)
        {
            cEqn.relax();
        }
        cEqn.solve();
        c_.min(cMax_);
        c_.max(cMin_); 

        // Solve the progress variable variance transport equation
        fvScalarMatrix cvarEqn
        (
            fvm::ddt(rho_,cvar_)
            +(
                buffer_
                ? scalarUWConvection->fvmDiv(phi_, cvar_)
                : fvm::div(phi_, cvar_)
            )
            -fvm::laplacian( mut/Sct_ + mu/Sc_, cvar_)
            -(2.0*mut/Sct_*(fvc::grad(c_) & fvc::grad(c_)))
            +2.0*(rho_*chi_c_)
            -2.0*(cOmega_c_-omega_c_*c_)  
        ); 

        if(relaxation_)
        {
            cvarEqn.relax();
        }
        cvarEqn.solve();
        cvar_.min(cvarMax_);
        cvar_.max(cvarMin_);      


        // Solve the covariance transport equation
        fvScalarMatrix ZcvarEqn
        (
            fvm::ddt(rho_,Zcvar_)
            +(
                buffer_
                ?  scalarUWConvection->fvmDiv(phi_, Zcvar_)
                :  fvm::div(phi_, Zcvar_)
            )
            -fvm::laplacian( mut/Sct_+mu/Sc_, Zcvar_)
            -(1.0*mut/Sct_*(fvc::grad(c_) & fvc::grad(Z_)))
            -(1.0*mut/Sct_*(fvc::grad(Z_) & fvc::grad(c_)))
            +(2.0*rho_*chi_Zc_)  
            -1.0*(ZOmega_c_-omega_c_*Z_)  
        );

        if(relaxation_)
        {
            ZcvarEqn.relax();
        }

        ZcvarEqn.solve();

        Zcvar_.min(ZcvarMax_);
        Zcvar_.max(ZcvarMin_);
        
    } 


}

template<class ReactionThermo>
void Foam::combustionModels::baseFGM<ReactionThermo>::initialiseFalmeKernel()
{

    reactFlowTime_= this->mesh().time().value()-ignBeginTime_; 

    Info<< "Time = " << this->mesh().time().timeName()
            << "   reactFlowTime = " << reactFlowTime_ << nl << endl;

    if(ignition_ && reactFlowTime_ > 0.0 && reactFlowTime_ < ignDurationTime_)
    {
        const vectorField& centre = this->mesh().cellCentres();    
        const scalarField x = centre.component(vector::X);  
        const scalarField y = centre.component(vector::Y); 
        const scalarField z = centre.component(vector::Z); 

        forAll(cCells_,celli) 
        {
            scalar R = Foam::sqrt(magSqr(x[celli]-x0_) + magSqr(y[celli]-y0_)
                                + magSqr(z[celli]-z0_));
            if(R <= R0_) {cCells_[celli] = 1.0;}
        }

        Info<< "Flame initialisation done"<< endl;
    }    


}

template<class ReactionThermo>
Foam::tmp<Foam::fvScalarMatrix>
Foam::combustionModels::baseFGM<ReactionThermo>::R(volScalarField& Y) const
{
    return laminar<ReactionThermo>::R(Y);
}

template<class ReactionThermo>
Foam::tmp<Foam::volScalarField>
Foam::combustionModels::baseFGM<ReactionThermo>::Qdot() const
{
    return volScalarField::New
    (
        this->thermo().phasePropertyName("Qdot"),
        laminar<ReactionThermo>::Qdot()
    );
}


template<class ReactionThermo>
bool Foam::combustionModels::baseFGM<ReactionThermo>::buffer()
{
    if(this->mesh().time().value() - this->mesh().time().startTime().value() < bufferTime_) buffer_ = true;  
    else buffer_ = false;

    return buffer_;
}


// ************************************************************************* //
