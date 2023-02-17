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

#include "flareFGM.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class ReactionThermo>
Foam::combustionModels::flareFGM<ReactionThermo>::flareFGM
(
    const word& modelType,
    ReactionThermo& thermo,
    const compressibleTurbulenceModel& turb,
    const word& combustionProperties
)
:
    baseFGM<ReactionThermo>(modelType, thermo, turb, combustionProperties),
    tableSolver(
                 baseFGM<ReactionThermo>::speciesNames_,
                 baseFGM<ReactionThermo>::scaledPV_,
                 baseFGM<ReactionThermo>::flameletT_,
                 baseFGM<ReactionThermo>::Ycmaxall_
               )
{
    //- retrieval data from table
    retrieval();
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

template<class ReactionThermo>
Foam::combustionModels::flareFGM<ReactionThermo>::~flareFGM()
{}

template<class ReactionThermo>
void Foam::combustionModels::flareFGM<ReactionThermo>::correct()
{
    //- initialize flame kernel
    baseFGM<ReactionThermo>::initialiseFalmeKernel();

    //- solve transport equation
    baseFGM<ReactionThermo>::transport();

    //update enthalpy using lookup data
    if(!(this->solveEnthalpy_))
    {
        this->He_ = this->Z_*(H_fuel-H_ox) + H_ox;
    }
  
    //- retrieval data from table
    retrieval();
    
}

template<class ReactionThermo>
void Foam::combustionModels::flareFGM<ReactionThermo>::retrieval()
{

    tmp<volScalarField> tk(this->turbulence().k());
    volScalarField& k = const_cast<volScalarField&>(tk());
    scalarField& kCells =k.primitiveFieldRef();

    tmp<volScalarField> tepsilon(this->turbulence().epsilon());
     volScalarField& epsilon = const_cast<volScalarField&>(tepsilon());
    const scalarField& epsilonCells =epsilon.primitiveFieldRef();

    tmp<volScalarField> tmu = this->turbulence().mu();  
    volScalarField& mu = const_cast<volScalarField&>(tmu());
    scalarField& muCells = mu.primitiveFieldRef();     


    //- calculate reacting flow solution
    const scalar Zl{z_Tb5[0]};  
    const scalar Zr{z_Tb5[NZL-1]};  

    dimensionedScalar TMin("TMin",dimensionSet(0,0,0,1,0,0,0),200.0);    
    dimensionedScalar TMax("TMax",dimensionSet(0,0,0,1,0,0,0),3000.0);  

    forAll(this->rho_, celli)  
    {

        this->chi_ZCells_[celli] = 1.0*epsilonCells[celli]/kCells[celli] *this->ZvarCells_[celli]; 

        this->chi_ZcCells_[celli] = 1.0*epsilonCells[celli]/kCells[celli] *this->ZcvarCells_[celli]; 

        if(this->ZCells_[celli] >= Zl && this->ZCells_[celli] <= Zr     
        && this->combustion_ && this->cCells_[celli] > this->small) 
        {

            double kc_s = this->lookup1d(NZL,z_Tb5,this->ZCells_[celli],kctau_Tb5);      
            double tau = this->lookup1d(NZL,z_Tb5,this->ZCells_[celli],tau_Tb5);    
            double sl = this->lookup1d(NZL,z_Tb5,this->ZCells_[celli],sl_Tb5);     
            double dl = this->lookup1d(NZL,z_Tb5,this->ZCells_[celli],th_Tb5);   

            this->chi_cCells_[celli] =
                this->RANSsdrFLRmodel(this->cvarCells_[celli],epsilonCells[celli],
                    kCells[celli],muCells[celli]/this->rho_[celli],
                    sl,dl,tau,kc_s,this->rho_[celli]);  

        }
        else
        {
            this->chi_cCells_[celli] = 1.0*epsilonCells[celli]/kCells[celli]*this->cvarCells_[celli]; 
        }

    // = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    
    double gz{cal_gvar(this->ZCells_[celli],this->ZvarCells_[celli])};   
    double gcz{cal_gcor(this->ZCells_[celli],this->cCells_[celli],this->ZvarCells_[celli],this->cvarCells_[celli],this->ZcvarCells_[celli])},
            Ycmax{-1.0},cNorm{},gc{};    

    if(scaledPV_)    
    {
        cNorm = this->cCells_[celli];  
    }
    else
    {
        Ycmax = this->lookup5d(NZ,z_Tb3,this->ZCells_[celli],
                              NC,c_Tb3,0.0,
                              NGZ,gz_Tb3,gz,
                              NGC,gc_Tb3,0.0,
                              NZC,gzc_Tb3,0.0,
                              Ycmax_Tb3);    
        Ycmax = max(this->smaller,Ycmax);    
        cNorm = this->cCells_[celli]/Ycmax; 
    }

    gc = cal_gvar(this->cCells_[celli],this->cvarCells_[celli],Ycmax);  

    this->WtCells_[celli] = lookup5d(NZ,z_Tb3,this->ZCells_[celli],    
                                   NC,c_Tb3,cNorm,
                                   NGZ,gz_Tb3,gz,
                                   NGC,gc_Tb3,gc,
                                   NZC,gzc_Tb3,gcz,
                                   mwt_Tb3);      

    muCells[celli] = lookup5d(NZ,z_Tb3,this->ZCells_[celli],
                                   NC,c_Tb3,cNorm,
                                   NGZ,gz_Tb3,gz,
                                   NGC,gc_Tb3,gc,
                                   NZC,gzc_Tb3,gcz,
                                   nu_Tb3)*this->rho_[celli];   

   // -------------------- Yis begin ------------------------------
    if(NY > 0)
    {
        this->YH2OCells_[celli] = lookup5d(NZ,z_Tb3,this->ZCells_[celli],
                                        NC,c_Tb3,cNorm,
                                        NGZ,gz_Tb3,gz,
                                        NGC,gc_Tb3,gc,
                                        NZC,gzc_Tb3,gcz,
                                        Yi01_Tb3);   
        this->YCOCells_[celli] = lookup5d(NZ,z_Tb3,this->ZCells_[celli],
                                       NC,c_Tb3,cNorm,
                                       NGZ,gz_Tb3,gz,
                                       NGC,gc_Tb3,gc,
                                       NZC,gzc_Tb3,gcz,
                                       Yi02_Tb3);   
        this->YCO2Cells_[celli] = lookup5d(NZ,z_Tb3,this->ZCells_[celli],
                                       NC,c_Tb3,cNorm,
                                       NGZ,gz_Tb3,gz,
                                       NGC,gc_Tb3,gc,
                                       NZC,gzc_Tb3,gcz,
                                       Yi03_Tb3);   
    }

    // -------------------- Yis end ------------------------------

    if(this->ZCells_[celli] >= Zl && this->ZCells_[celli] <= Zr
       && this->combustion_ && this->cCells_[celli] > this->small)  
    {
        this->omega_cCells_[celli] =
            lookup5d(NZ,z_Tb3,this->ZCells_[celli],
                          NC,c_Tb3,cNorm,
                          NGZ,gz_Tb3,gz,
                          NGC,gc_Tb3,gc,
                          NZC,gzc_Tb3,gcz,
                          omgc_Tb3)
            + (
                  scaledPV_
                  ? this->chi_ZCells_[celli]*this->cCells_[celli]
                    *lookup2d(NZ,z_Tb3,this->ZCells_[celli],
                                   NGZ,gz_Tb3,gz,d2Yeq_Tb2)
                  : 0.0
              );   

         this->cOmega_cCells_[celli] = lookup5d(NZ,z_Tb3,this->ZCells_[celli],
                                             NC,c_Tb3,cNorm,
                                             NGZ,gz_Tb3,gz,
                                             NGC,gc_Tb3,gc,
                                             NZC,gzc_Tb3,gcz,
                                             cOc_Tb3);    

         this->ZOmega_cCells_[celli] = lookup5d(NZ,z_Tb3,this->ZCells_[celli],
                                             NC,c_Tb3,cNorm,
                                             NGZ,gz_Tb3,gz,
                                             NGC,gc_Tb3,gc,
                                             NZC,gzc_Tb3,gcz,
                                             ZOc_Tb3);    

    }
    else   
    {
        this->omega_cCells_[celli] = 0.0;
        this->cOmega_cCells_[celli] = 0.0;  
        this->ZOmega_cCells_[celli] = 0.0; 
    }

    if(flameletT_)   
    {
        this->TCells_[celli] = lookup5d(NZ,z_Tb3,this->ZCells_[celli],
                                      NC,c_Tb3,cNorm,
                                      NGZ,gz_Tb3,gz,
                                      NGC,gc_Tb3,gc,
                                      NZC,gzc_Tb3,gcz,
                                      Tf_Tb3);   
    }
    else
    {
        this->CpCells_[celli] = lookup5d(NZ,z_Tb3,this->ZCells_[celli],
                                       NC,c_Tb3,cNorm,
                                       NGZ,gz_Tb3,gz,
                                       NGC,gc_Tb3,gc,
                                       NZC,gzc_Tb3,gcz,
                                       cp_Tb3);   

         this->HfCells_[celli] = lookup5d(NZ,z_Tb3,this->ZCells_[celli],
                                       NC,c_Tb3,cNorm,
                                       NGZ,gz_Tb3,gz,
                                       NGC,gc_Tb3,gc,
                                       NZC,gzc_Tb3,gcz,
                                       hiyi_Tb3);   

        this->TCells_[celli] = (this->HCells_[celli]-this->HfCells_[celli])/this->CpCells_[celli]
                        + this->T0;   
    }

    this->omega_cCells_[celli] = this->omega_cCells_[celli]*this->rho_[celli];   
    this->cOmega_cCells_[celli] = this->cOmega_cCells_[celli]*this->rho_[celli];
    this->ZOmega_cCells_[celli] = this->ZOmega_cCells_[celli]*this->rho_[celli];

    }


    //-----------update boundary---------------------------------

    forAll(this->rho_.boundaryFieldRef(), patchi)   
    {
        fvPatchScalarField& pZ = this->Z_.boundaryFieldRef()[patchi];     
        fvPatchScalarField& pZvar = this->Zvar_.boundaryFieldRef()[patchi];  
        fvPatchScalarField& pH = this->He_.boundaryFieldRef()[patchi];     
        fvPatchScalarField& pWt = this->Wt_.boundaryFieldRef()[patchi];   
        fvPatchScalarField& pCp = this->Cp_.boundaryFieldRef()[patchi];   
        fvPatchScalarField& pHf = this->Hf_.boundaryFieldRef()[patchi];   
        fvPatchScalarField& pT = this->T_.boundaryFieldRef()[patchi];     
        fvPatchScalarField& prho_ = this->rho_.boundaryFieldRef()[patchi];   
        fvPatchScalarField& pomega_c = this->omega_c_.boundaryFieldRef()[patchi];   
        fvPatchScalarField& pcOmega_c = this->cOmega_c_.boundaryFieldRef()[patchi];  
        fvPatchScalarField& pZOmega_c = this->ZOmega_c_.boundaryFieldRef()[patchi];  
        fvPatchScalarField& pc = this->c_.boundaryFieldRef()[patchi];          
        fvPatchScalarField& pcvar = this->cvar_.boundaryFieldRef()[patchi];    
        fvPatchScalarField& pZcvar = this->Zcvar_.boundaryFieldRef()[patchi];    
        fvPatchScalarField& pchi_Z = this->chi_Z_.boundaryFieldRef()[patchi];  
        fvPatchScalarField& pchi_c = this->chi_c_.boundaryFieldRef()[patchi];   
        fvPatchScalarField& pchi_Zc = this->chi_Zc_.boundaryFieldRef()[patchi];   

        tmp<scalarField> tmuw = this->turbulence().mu(patchi);
        scalarField& pmu = const_cast<scalarField&>(tmuw());    

        fvPatchScalarField& pk = k.boundaryFieldRef()[patchi];
        fvPatchScalarField& pepsilon = epsilon.boundaryFieldRef()[patchi];


        forAll(prho_, facei)   
        {

            pchi_Z[facei] = 1.0*pepsilon[facei]/pk[facei] *pZvar[facei]; 

            pchi_Zc[facei]= 1.0*pepsilon[facei]/pk[facei] *pZcvar[facei]; 

            if(pZ[facei] >= Zl && pZ[facei] <= Zr
            && this->combustion_ && pc[facei] > this->small) 
            {

                double kc_s = lookup1d(NZL,z_Tb5,pZ[facei],kctau_Tb5);     
                double tau = lookup1d(NZL,z_Tb5,pZ[facei],tau_Tb5);    
                double sl = lookup1d(NZL,z_Tb5,pZ[facei],sl_Tb5);      
                double dl = lookup1d(NZL,z_Tb5,pZ[facei],th_Tb5);      

                pchi_c[facei] =
                    RANSsdrFLRmodel(pcvar[facei],pepsilon[facei],
                        pk[facei],pmu[facei]/prho_[facei],
                        sl,dl,tau,kc_s,prho_[facei]);
            }
            else
            {      
                pchi_c[facei] = 1.0*pepsilon[facei]/pk[facei] *pcvar[facei]; 

            }

         // = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

            double gz{cal_gvar(pZ[facei],pZvar[facei])};  
            double gcz{cal_gcor(pZ[facei],pc[facei],pZvar[facei],pcvar[facei],pZcvar[facei])},
                    Ycmax{-1.0},cNorm{},gc{};     

            if(scaledPV_)
            {
                cNorm = pc[facei];   
            }
            else
            {
                Ycmax = lookup5d(NZ,z_Tb3,pZ[facei],
                                        NC,c_Tb3,0.0,
                                        NGZ,gz_Tb3,gz,
                                        NGC,gc_Tb3,0.0,
                                        NZC,gzc_Tb3,0.0,
                                        Ycmax_Tb3);    
                Ycmax = max(this->smaller,Ycmax);  
                cNorm = pc[facei]/Ycmax;   
            }

            gc = cal_gvar(pc[facei],pcvar[facei],Ycmax);   

            pWt[facei] = lookup5d(NZ,z_Tb3,pZ[facei],
                                        NC,c_Tb3,cNorm,
                                        NGZ,gz_Tb3,gz,
                                        NGC,gc_Tb3,gc,
                                        NZC,gzc_Tb3,gcz,
                                        mwt_Tb3);    

            pmu[facei] = lookup5d(NZ,z_Tb3,pZ[facei],
                                        NC,c_Tb3,cNorm,
                                        NGZ,gz_Tb3,gz,
                                        NGC,gc_Tb3,gc,
                                        NZC,gzc_Tb3,gcz,
                                        nu_Tb3)*prho_[facei];  

            if(pZ[facei] >= Zl && pZ[facei] <= Zr
                && this->combustion_ && pc[facei] > this->small) 
            {
                pomega_c[facei] =
                        lookup5d(NZ,z_Tb3,pZ[facei],
                                    NC,c_Tb3,cNorm,
                                    NGZ,gz_Tb3,gz,
                                    NGC,gc_Tb3,gc,
                                    NZC,gzc_Tb3,gcz,
                                    omgc_Tb3)   
                    + (
                            scaledPV_
                            ? pchi_Z[facei]*pc[facei]
                            *lookup2d(NZ,z_Tb3,pZ[facei],
                                            NGZ,gz_Tb3,gz,d2Yeq_Tb2)
                            : 0.0
                        );  

                pcOmega_c[facei] = lookup5d(NZ,z_Tb3,pZ[facei],
                                                    NC,c_Tb3,cNorm,
                                                    NGZ,gz_Tb3,gz,
                                                    NGC,gc_Tb3,gc,
                                                    NZC,gzc_Tb3,gcz,cOc_Tb3);

                pZOmega_c[facei] = lookup5d(NZ,z_Tb3,pZ[facei],
                                            NC,c_Tb3,cNorm,
                                            NGZ,gz_Tb3,gz,
                                            NGC,gc_Tb3,gc,
                                            NZC,gzc_Tb3,gcz,ZOc_Tb3);
            }    
            else
            {
                pomega_c[facei] = 0.0;
                pcOmega_c[facei] = 0.0;
                pZOmega_c[facei] = 0.0;
            }

            if(flameletT_)  
            {
                pT[facei] = lookup5d(NZ,z_Tb3,pZ[facei],
                                        NC,c_Tb3,cNorm,
                                        NGZ,gz_Tb3,gz,
                                        NGC,gc_Tb3,gc,
                                        NZC,gzc_Tb3,gcz,
                                        Tf_Tb3);   
            }
            else
            {
                pCp[facei] = lookup5d(NZ,z_Tb3,pZ[facei],
                                        NC,c_Tb3,cNorm,
                                        NGZ,gz_Tb3,gz,
                                        NGC,gc_Tb3,gc,
                                        NZC,gzc_Tb3,gcz,
                                        cp_Tb3);   

                pHf[facei] = lookup5d(NZ,z_Tb3,pZ[facei],
                                        NC,c_Tb3,cNorm,
                                        NGZ,gz_Tb3,gz,
                                        NGC,gc_Tb3,gc,
                                        NZC,gzc_Tb3,gcz,
                                        hiyi_Tb3);  

                pT[facei] = (pH[facei]-pHf[facei])/pCp[facei]
                            + this->T0;  
            }

            pomega_c[facei] = pomega_c[facei]*prho_[facei];    
            pcOmega_c[facei] = pcOmega_c[facei]*prho_[facei];
            pZOmega_c[facei] = pZOmega_c[facei]*prho_[facei];

        } 

    }

    this->T_.max(TMin);  
    this->T_.min(TMax);  

    if(this->mesh().time().timeIndex() > 0)   
    {
        dimensionedScalar p_operateDim("p_operateDim", dimensionSet(1,-1,-2,0,0,0,0),this->incompPref_);  

        if(this->incompPref_ > 0.0) 
        {  
            this->rho_ = p_operateDim*this->psi_; 
        }
        else 
        {
            this->rho_ = this->p_*this->psi_;
        }
    }

    dimensionedScalar R_uniGas("R_uniGas",dimensionSet(1,2,-2,-1,-1,0,0),8.314e3);
    this->psi_ = this->Wt_/(R_uniGas*this->T_);
}

// ************************************************************************* //
