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

#include "PaSR.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class ReactionThermo>
Foam::combustionModels::PaSR<ReactionThermo>::PaSR
(
    const word& modelType,
    ReactionThermo& thermo,
    const compressibleTurbulenceModel& turb,
    const word& combustionProperties
)
:
    laminar<ReactionThermo>(modelType, thermo, turb, combustionProperties),
    mu_(const_cast<volScalarField&>(dynamic_cast<psiThermo&>(thermo).mu()())),
    p_(this->thermo().p()),
    T_(this->thermo().T()),
    mixingScaleDict_(this->coeffs().subDict("mixingScale")),
    chemistryScaleDict_(this->coeffs().subDict("chemistryScale")),
    mixingScaleType_(mixingScaleDict_.lookup("type")),
    chemistryScaleType_(chemistryScaleDict_.lookup("type")),
    mixingScaleCoeffs_(mixingScaleDict_.optionalSubDict(mixingScaleType_ + "Coeffs")),
    chemistryScaleCoeffs_(chemistryScaleDict_.optionalSubDict(chemistryScaleType_ + "Coeffs")),
    fuel_(chemistryScaleCoeffs_.lookupOrDefault("fuel", word(""))),
    oxidizer_(chemistryScaleCoeffs_.lookupOrDefault("oxidizer",word(""))),
    tmix_
    (
        IOobject
        (
            "tmix",
            this->mesh().time().timeName(),
            this->mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh(),
        dimensionedScalar(dimTime, 0)
    ),
    tc_
    (
        IOobject
        (
            "tc",
            this->mesh().time().timeName(),
            this->mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh(),
        dimensionedScalar(dimTime, 0.1)
    ),
    Da_
    (
        IOobject
        (
            "Da",
            this->mesh().time().timeName(),
            this->mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh(),
        dimensionedScalar(dimless, 0)
    ),
    Z_
    (
        IOobject
        (
            "Z",
            this->mesh().time().timeName(),
            this->mesh(),
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        this->mesh(),
        dimensionedScalar(dimless, 0)
    ),
    Zvar_
    (
        IOobject
        (
            "Zvar",
            this->mesh().time().timeName(),
            this->mesh(),
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        this->mesh(),
        dimensionedScalar(dimless, 0)
    ),
    Chi_
    (
        IOobject
        (
            "Chi",
            this->mesh().time().timeName(),
            this->mesh(),
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        this->mesh(),
        dimensionedScalar(dimless/dimTime, SMALL)
    ),
    eqR_
    (
        IOobject
        (
            "eqR",
            this->mesh().time().timeName(),
            this->mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh(),
        dimensionedScalar(dimless, 0.0)
    ),
    Su_
    (
        IOobject
        (
            "Su",
            this->mesh().time().timeName(),
            this->mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh(),
        dimensionedScalar(dimVelocity, 0.0)
    ),        
    kappa_
    (
        IOobject
        (
            thermo.phasePropertyName(typeName + ":kappa"),
            this->mesh().time().timeName(),
            this->mesh(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh(),
        dimensionedScalar(dimless, 1.0)
    ),
    ChiType_(mixingScaleCoeffs_.lookupOrDefault("ChiType", word("")))
{
     Cmix_=mixingScaleCoeffs_.lookupOrDefault("Cmix",0.1);
     Zst_=mixingScaleCoeffs_.lookupOrDefault("Zst",0.054);

     //- adopted from Ferrarotti et al. 2019 PCI
     Cd1_=mixingScaleCoeffs_.lookupOrDefault("Cd1",1.5604);
     Cd2_=mixingScaleCoeffs_.lookupOrDefault("Cd2",1.1854);
     Cp1_=mixingScaleCoeffs_.lookupOrDefault("Cp1",1.6053);
     Cp2_=mixingScaleCoeffs_.lookupOrDefault("Cp2",1.1978);
     maxChi_=mixingScaleCoeffs_.lookupOrDefault("maxChi",5000);

     //- Gulders laminar flame speed model constants
     W_=mixingScaleCoeffs_.lookupOrDefault("W",0.422);
     eta_=mixingScaleCoeffs_.lookupOrDefault("eta",0.15);
     xi_=mixingScaleCoeffs_.lookupOrDefault("xi",5.18);
     alpha_=mixingScaleCoeffs_.lookupOrDefault("alpha",2.0);
     beta_=mixingScaleCoeffs_.lookupOrDefault("beta",-0.5);


     fields_.add(Z_);
     fields_.add(Zvar_);
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

template<class ReactionThermo>
Foam::combustionModels::PaSR<ReactionThermo>::~PaSR()
{}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

template<class ReactionThermo>
void Foam::combustionModels::PaSR<ReactionThermo>::correct()
{
    laminar<ReactionThermo>::correct();
 
    tmp<volScalarField> tk(this->turbulence().k());
    const volScalarField& k = tk();

    tmp<volScalarField> tepsilon(this->turbulence().epsilon());
    const volScalarField& epsilon = tepsilon();


    tmp<volScalarField> trho(this->rho());
    const volScalarField& rho = trho();

    dimensionedScalar smallEpsilon("smallEpsilon",dimensionSet(0, 2, -3, 0, 0, 0, 0), SMALL);	
    dimensionedScalar chismall_("chismall", dimensionSet(0,0,-1,0,0,0,0), SMALL );
    dimensionedScalar tauMixsmall_("tauMixsmall", dimensionSet(0,0,1,0,0,0,0), this->mesh().time().deltaTValue() );
    dimensionedScalar tauMixlarge_("tauMixlarge", dimensionSet(0,0,1,0,0,0,0), 0.1 );

    //-mixing time scale 
    if(mixingScaleType_=="globalScale")
    {
       tmix_=Cmix_*k/(epsilon+smallEpsilon);
    }

    else if(mixingScaleType_=="kolmogorovScale")
    {
       tmix_=sqrt(mag(mu_/rho/(epsilon+smallEpsilon)));  
    }

    else if(mixingScaleType_=="geometriMeanScale")
    {
       tmix_=sqrt((mag(k/(epsilon+smallEpsilon)))*sqrt(mag(mu_/rho/(epsilon+smallEpsilon))));
    }
    
    else if(mixingScaleType_=="dynamicScale")
    {
        transport();
        tmix_ = min(max(Zvar_/(Chi_+chismall_),tauMixsmall_),tauMixlarge_);
    }
    else
    {
            FatalErrorInFunction
            << "Unknown mixingScaleType type "
            << mixingScaleType_
            << ", mixingScaleType not in table" << nl << nl
            << exit(FatalError);
    }


    //- chemitry time scale
    if(chemistryScaleType_=="globalConvertion")
    {

       if(fuel_==word(""))
       {
            FatalErrorInFunction
            << "fuel is not specified " << nl << nl
            << exit(FatalError);
       }

       const label specieFuel= this->chemistryPtr_->species()[fuel_];
	   const label specieOxidizer = this->chemistryPtr_->species()[oxidizer_];
       const label specieCO2 = this->chemistryPtr_->species()["CO2"];
       const label specieH2 = this->chemistryPtr_->species()["H2"];



       const volScalarField& Yfuel = this->chemistryPtr_->Y()[specieFuel];
       const volScalarField& Yoxidizer = this->chemistryPtr_->Y()[specieOxidizer];
       const volScalarField& YCO2 = this->chemistryPtr_->Y()[specieCO2];
       const volScalarField& YH2 = this->chemistryPtr_->Y()[specieH2];

     //- initialize fuel and oxidizer chemitry time scale
        volScalarField t_fuel=tc_;
		volScalarField t_oxidizer=tc_;
		volScalarField t_CO2=tc_;        
		volScalarField t_H2=tc_;   

	   forAll(rho,cellI)
	   {

		scalar RR_fuel = this->chemistryPtr_->RR(specieFuel)[cellI];
		scalar RR_oxidizer = this->chemistryPtr_->RR(specieOxidizer)[cellI];
		scalar RR_CO2 = this->chemistryPtr_->RR(specieCO2)[cellI];
		scalar RR_H2 = this->chemistryPtr_->RR(specieH2)[cellI];

		if( (RR_oxidizer < 0.0)  &&  (Yoxidizer[cellI] > 1e-10) )							
		{			
			t_oxidizer[cellI] =  -rho[cellI] * Yoxidizer[cellI]/(RR_oxidizer);   
		}
	 
		if	( (RR_fuel < 0.0) && (Yfuel[cellI] > 1e-10))
		{								
			t_fuel[cellI] =  -rho[cellI] * Yfuel[cellI]/(RR_fuel);   
		}

		if	( (RR_CO2 > 0.0) && (YCO2[cellI] > 1e-10))
		{								
			t_CO2[cellI] =  rho[cellI] * YCO2[cellI]/(RR_CO2);   
		}        

        if	( (RR_H2 < 0.0) && (YH2[cellI] > 1e-10))
		{								
			t_H2[cellI] =  -rho[cellI] * YH2[cellI]/(RR_H2);   
		}        

		tc_[cellI] = max(t_oxidizer[cellI],t_fuel[cellI]);     

        tc_[cellI] = max(t_CO2[cellI],tc_[cellI]);     
	 	
        tc_[cellI] = max(t_H2[cellI],tc_[cellI]);     
	  }

    }

    if(chemistryScaleType_=="formationRate")
    {
       tc_ = this->tc();
    }


    forAll(kappa_, cellI)
    {
		kappa_[cellI] = (tmix_[cellI] > SMALL && tc_[cellI] > SMALL) ?  tc_[cellI]/(tc_[cellI] + tmix_[cellI]) : 1.0;
        
    }
}


template<class ReactionThermo>
Foam::tmp<Foam::fvScalarMatrix>
Foam::combustionModels::PaSR<ReactionThermo>::R(volScalarField& Y) const
{
    return kappa_*laminar<ReactionThermo>::R(Y);
}

template<class ReactionThermo>
bool Foam::combustionModels::PaSR<ReactionThermo>::read()
{
    if (laminar<ReactionThermo>::read())
    {
        this->coeffs().lookup("Cmix") >> Cmix_;
        return true;
    }
    else
    {
        return false;
    }
}


template<class ReactionThermo>
Foam::tmp<Foam::volScalarField>
Foam::combustionModels::PaSR<ReactionThermo>::Qdot() const
{
    return volScalarField::New
    (
        this->thermo().phasePropertyName(typeName + ":Qdot"),
        kappa_*laminar<ReactionThermo>::Qdot()
    );
}

template<class ReactionThermo>
void Foam::combustionModels::PaSR<ReactionThermo>::transport() 
{

    tmp<volScalarField> tmuEff(this->turbulence().muEff());
    const volScalarField& muEff = tmuEff();

    tmp<volScalarField> tmut(this->turbulence().mut());
    const volScalarField& mut = tmut();

    const surfaceScalarField& phi_ = this->mesh().objectRegistry::lookupObject<surfaceScalarField>("phi");
    const volVectorField& U_=this->mesh().objectRegistry::lookupObject<volVectorField>("U");

    tmp<fv::convectionScheme<scalar>> mvConvection
       (
          fv::convectionScheme<scalar>::New
           (
              this->mesh(),
              fields_,
              phi_,
              this->mesh().divScheme("div(phi,Z)")
           )
       );


  dimensionedScalar smallK_("smallK",dimVelocity*dimVelocity,SMALL);


  //- mixture fraction  equation 
  fvScalarMatrix ZEqn
  (
     fvm::ddt(this->rho(), this->Z_)
    //+ fvm::div(this->phi(), this->Z_)
    + mvConvection->fvmDiv(phi_, this->Z_)
    - fvm::laplacian(muEff, this->Z_)
  );
  ZEqn.relax();
  ZEqn.solve("Z");
  this->Z_.max(0.0);

   Info<< "min/max(Z_) = " << min(Z_).value() << ", " << max(Z_).value() << endl;

  //- mixtrue fraction variance equation
  
    fvScalarMatrix ZvarEqn
    (
        fvm::ddt(this->rho(), this->Zvar_)
      + mvConvection->fvmDiv(phi_, this->Zvar_)
      - fvm::laplacian(muEff, this->Zvar_)
      ==
      + 2*mut*magSqr(fvc::grad(this->Z_))
      - this->rho()*Chi_
    );

    ZvarEqn.relax();
    ZvarEqn.solve("Zvar");
    this->Zvar_.max(0.0);
    this->Zvar_.min(0.25);

     Info<< "min/max(Zvar_) = " << min(Zvar_).value() << ", " << max(Zvar_).value() << endl;


  //- scalar dissipation rate equation 
  if(ChiType_=="constAlgebraic")
  {
       Chi_ = 1*this->turbulence().epsilon()/(this->turbulence().k()+smallK_)*Zvar_; //-default is 2

       Info<< "min/max(Chi_) = " << min(Chi_).value() << ", " << max(Chi_).value() << endl; 
  }

  if(ChiType_=="dynAlgebraic")  
  {
      forAll(eqR_, cellI)
      {
        eqR_[cellI] = (Z_[cellI]/((1.0-Z_[cellI])+SMALL))*((1-Zst_)/Zst_);
      }
      eqR_.max(0);  

      //laminar speed calculation 
      volScalarField SuRef=0*U_.component(0);
      
      static const scalar Tref = 300.0;
      static const scalar pRef = 1.013e5;    

      forAll(SuRef, cellI)
      {
        SuRef[cellI]=W_*pow(eqR_[cellI], eta_)*exp(-xi_*sqr(eqR_[cellI] - 1.075));

        Su_[cellI]=SuRef[cellI]*pow((T_[cellI]/Tref), alpha_)*pow((p_[cellI]/pRef), beta_);
      }
  
      Chi_ = 0.21*this->turbulence().epsilon()/(this->turbulence().k()+smallK_)*Zvar_ 
             +2/3*(0.1*Su_/(sqrt(this->turbulence().k())))*0.21*this->turbulence().epsilon()/(this->turbulence().k()+smallK_)*Zvar_;


      Chi_.min(maxChi_);

       Info<< "min/max(Chi_) = " << min(Chi_).value() << ", " << max(Chi_).value() << endl; 

  }
  if(ChiType_=="transport") 
  {
        scalar Sct=0.7;
    	volScalarField D1 = Cd1_*this->rho()*sqr(Chi_)/(Zvar_+SMALL);
        volScalarField D2 = Cd2_*this->rho()*this->turbulence().epsilon()/(this->turbulence().k()+smallK_)*Chi_;
        volScalarField P1 = 2.00*Cp1_*this->turbulence().epsilon()/(this->turbulence().k()+smallK_)*this->turbulence().mut()/Sct*magSqr(fvc::grad(Z_));
        volScalarField P2 = Cp2_*this->turbulence().mut()*Chi_/(this->turbulence().k()+smallK_)*(fvc::grad(U_) && dev(twoSymm(fvc::grad(U_))));

        volScalarField S_chi = P1 + P2 - D1 - D2;	

        fvScalarMatrix ChiEqn
        (
            fvm::ddt(this->rho(), Chi_)
            + mvConvection->fvmDiv(phi_, Chi_)
            // - fvm::laplacian(this->thermo().alpha()+this->turbulence().mut()/Sct, Chi_)
            - fvm::laplacian(muEff/Sct, Chi_)
            ==
            S_chi
        );


        ChiEqn.relax();
        ChiEqn.solve();
        Chi_.max(0.00000001);
        Chi_.min(maxChi_);

         Info<< "min/max(Chi_) = " << min(Chi_).value() << ", " << max(Chi_).value() << endl; 

  }


}
// ************************************************************************* //
