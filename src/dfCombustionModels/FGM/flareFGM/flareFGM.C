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
    tableSolver(baseFGM<ReactionThermo>::tablePath_)
{
    //- retrieval data from table
    retrieval();

    Info<< "At flareFGM, min/max(T) = " << min(this->T_).value() << ", " << max(this->T_).value() << endl;    

    if (tableSolver::scaledPV_ != baseFGM<ReactionThermo>::scaledPV_)
    {
        Info << "Warning! -- scaledPV in FGM table: " << tableSolver::scaledPV_
            << "are not equal to that in combustionProperties: " << baseFGM<ReactionThermo>::scaledPV_ << endl;
    }
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

    if (this->isLES_)
    {
        baseFGM<ReactionThermo>::magUPrime();
    }

    //update enthalpy using lookup data
    if(!(this->solveEnthalpy_))
    {
        this->He_ = this->Z_*(this->H_fuel-this->H_ox) + this->H_ox;
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

    tmp<volScalarField> tmut = this->turbulence().mut();  
    volScalarField& mut = const_cast<volScalarField&>(tmut());
    scalarField& mutCells = mut.primitiveFieldRef();   

    int ih = 0;
    scalar Zl, Zr;

    // dimensionedScalar TMin("TMin",dimensionSet(0,0,0,1,0,0,0),270.0);    
    // dimensionedScalar TMax("TMax",dimensionSet(0,0,0,1,0,0,0),5000.0);  
    const double TMin = 270.0;
    const double TMax = 5000.0;

    forAll(this->rho_, celli)  
    {
        if(this->isLES_)
        {
             this->chi_ZCells_[celli] = this->sdrLRXmodel(2.0,mutCells[celli]
                                   /this->rho_[celli],this->deltaCells_[celli],this->ZvarCells_[celli]); 
             this->chi_ZcCells_[celli] = this->sdrLRXmodel(2.0,mutCells[celli]
                                  /this->rho_[celli],this->deltaCells_[celli],this->ZcvarCells_[celli]); 
        }     
        else
        {
            this->chi_ZCells_[celli] = 1.0*epsilonCells[celli]/kCells[celli] *this->ZvarCells_[celli]; 

            this->chi_ZcCells_[celli] = 1.0*epsilonCells[celli]/kCells[celli] *this->ZcvarCells_[celli];             
        }


        double hLoss = ( this->ZCells_[celli]*(this->Hfu-this->Hox) + this->Hox ) - this->HCells_[celli];
        hLoss = max(hLoss, this->h_Tb3[0]);
        hLoss = min(hLoss, this->h_Tb3[this->NH - 1]);

        ih = this->locate_lower(this->NH,this->h_Tb3,hLoss);

        Zl = this->z_Tb5[ih*this->NZL];  
        Zr = this->z_Tb5[(ih+1)*this->NZL-1];

        if(this->ZCells_[celli] >= Zl && this->ZCells_[celli] <= Zr     
        && this->combustion_ && this->cCells_[celli] > this->small) 
        {

            double kc_s = this->lookup2d(this->NH,this->h_Tb3,hLoss,this->NZL,this->z_Tb5,this->ZCells_[celli],this->kctau_Tb5);      
            double tau = this->lookup2d(this->NH,this->h_Tb3,hLoss,this->NZL,this->z_Tb5,this->ZCells_[celli],this->tau_Tb5);    
            double sl = this->lookup2d(this->NH,this->h_Tb3,hLoss,this->NZL,this->z_Tb5,this->ZCells_[celli],this->sl_Tb5);     
            double dl = this->lookup2d(this->NH,this->h_Tb3,hLoss,this->NZL,this->z_Tb5,this->ZCells_[celli],this->th_Tb5);   

            if(this->isLES_)
            {
                this->chi_cCells_[celli] = this->sdrFLRmodel(this->cvarCells_[celli],this->magUPrimeCells_[celli],
                                            this->deltaCells_[celli],sl,dl,tau,kc_s,this->betacCells_[celli]);
            } 
            else
            {
                this->chi_cCells_[celli] =
                    this->RANSsdrFLRmodel(this->cvarCells_[celli],epsilonCells[celli],
                        kCells[celli],muCells[celli]/this->rho_[celli],
                        sl,dl,tau,kc_s,this->rho_[celli]);  
            }

        }
        else
        {
            if(this->isLES_)
            {
                this->chi_cCells_[celli] = this->sdrLRXmodel(2.0,mutCells[celli]
                           /this->rho_[celli],this->deltaCells_[celli],this->cvarCells_[celli]);     
            }
            else
            {
                this->chi_cCells_[celli] = 1.0*epsilonCells[celli]/kCells[celli]*this->cvarCells_[celli]; 
            }
        }
  

        // = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
        
        double gz{this->cal_gvar(this->ZCells_[celli],this->ZvarCells_[celli])};   
        double gcz{this->cal_gcor(this->ZCells_[celli],this->cCells_[celli],this->ZvarCells_[celli],this->cvarCells_[celli],this->ZcvarCells_[celli])},
                Ycmax{-1.0},cNorm{},gc{};    

        if(tableSolver::scaledPV_)    
        {
            cNorm = this->cCells_[celli];  
        }
        else
        {
            Ycmax = this->lookup6d(this->NH,this->h_Tb3,hLoss,
                                this->NZ,this->z_Tb3,this->ZCells_[celli],
                                this->NC,this->c_Tb3,0.0,
                                this->NGZ,this->gz_Tb3,gz,
                                this->NGC,this->gc_Tb3,0.0,
                                this->NZC,this->gzc_Tb3,0.0,
                                this->tableValues_[this->NS-1]);    
            Ycmax = max(this->smaller,Ycmax);    
            cNorm = this->cCells_[celli]/Ycmax; 
        }

        gc = this->cal_gvar(this->cCells_[celli],this->cvarCells_[celli],Ycmax);  

        if(this->ZCells_[celli] >= Zl && this->ZCells_[celli] <= Zr
        && this->combustion_ && this->cCells_[celli] > this->small)  
        {
            if (this->isLES_)
            {
                this->omega_cCells_[celli] =
                    this->lookup6d(this->NH,this->h_Tb3,hLoss,
                                this->NZ,this->z_Tb3,this->ZCells_[celli],
                                this->NC,this->c_Tb3,cNorm,
                                this->NGZ,this->gz_Tb3,gz,
                                this->NGC,this->gc_Tb3,gc,
                                this->NZC,this->gzc_Tb3,gcz,
                                this->tableValues_[0])
                    + (
                        tableSolver::scaledPV_
                        ? (this->chi_ZCells_[celli] + this->chi_ZfltdCells_[celli])*this->cCells_[celli]
                            *this->lookup3d(this->NH,this->h_Tb3,hLoss,this->NZ,this->z_Tb3,this->ZCells_[celli],
                                        this->NGZ,this->gz_Tb3,gz,this->d2Yeq_Tb2)
                            + 2*this->chi_ZcCells_[celli]*this->lookup3d(this->NH,this->h_Tb3,hLoss,
                                                    this->NZ,this->z_Tb3,this->ZCells_[celli],
                                                    this->NGZ,this->gz_Tb3,gz,this->d1Yeq_Tb2)
                        : 0.0
                    );   
            }
            else
            {
                this->omega_cCells_[celli] =
                    this->lookup6d(this->NH,this->h_Tb3,hLoss,
                                this->NZ,this->z_Tb3,this->ZCells_[celli],
                                this->NC,this->c_Tb3,cNorm,
                                this->NGZ,this->gz_Tb3,gz,
                                this->NGC,this->gc_Tb3,gc,
                                this->NZC,this->gzc_Tb3,gcz,
                                this->tableValues_[0])
                    + (
                        tableSolver::scaledPV_
                        ? (  this->chi_ZCells_[celli]*this->cCells_[celli]
                            *this->lookup3d(this->NH,this->h_Tb3,hLoss,this->NZ,this->z_Tb3,this->ZCells_[celli],
                                        this->NGZ,this->gz_Tb3,gz,this->d2Yeq_Tb2)
                            + 2*this->chi_ZcCells_[celli]*this->lookup3d(this->NH,this->h_Tb3,hLoss,
                                                        this->NZ,this->z_Tb3,this->ZCells_[celli],
                                                        this->NGZ,this->gz_Tb3,gz,this->d1Yeq_Tb2)  )
                        : 0.0
                    );   
            }


            this->cOmega_cCells_[celli] = this->lookup6d(this->NH,this->h_Tb3,hLoss,
                                                this->NZ,this->z_Tb3,this->ZCells_[celli],
                                                this->NC,this->c_Tb3,cNorm,
                                                this->NGZ,this->gz_Tb3,gz,
                                                this->NGC,this->gc_Tb3,gc,
                                                this->NZC,this->gzc_Tb3,gcz,
                                                this->tableValues_[1]);    

            this->ZOmega_cCells_[celli] = this->lookup6d(this->NH,this->h_Tb3,hLoss,
                                                this->NZ,this->z_Tb3,this->ZCells_[celli],
                                                this->NC,this->c_Tb3,cNorm,
                                                this->NGZ,this->gz_Tb3,gz,
                                                this->NGC,this->gc_Tb3,gc,
                                                this->NZC,this->gzc_Tb3,gcz,
                                                this->tableValues_[2]);    

        }
        else   
        {
            this->omega_cCells_[celli] = 0.0;
            this->cOmega_cCells_[celli] = 0.0;  
            this->ZOmega_cCells_[celli] = 0.0; 
        }
        this->omega_cCells_[celli] = this->omega_cCells_[celli]*this->rho_[celli];   
        this->cOmega_cCells_[celli] = this->cOmega_cCells_[celli]*this->rho_[celli];
        this->ZOmega_cCells_[celli] = this->ZOmega_cCells_[celli]*this->rho_[celli];

        this->He_s_[celli] = hLoss;
        this->c_s_[celli] = cNorm;
        this->Zvar_s_[celli] = gz;
        this->cvar_s_[celli] = gc;
        this->Zcvar_s_[celli] = gcz;

        this->WtCells_[celli] = this->lookup6d(this->NH,this->h_Tb3,hLoss,
                                    this->NZ,this->z_Tb3,this->ZCells_[celli],    
                                    this->NC,this->c_Tb3,cNorm,
                                    this->NGZ,this->gz_Tb3,gz,
                                    this->NGC,this->gc_Tb3,gc,
                                    this->NZC,this->gzc_Tb3,gcz,
                                    this->tableValues_[4]);      

        muCells[celli] = this->lookup6d(this->NH,this->h_Tb3,hLoss,
                                    this->NZ,this->z_Tb3,this->ZCells_[celli],
                                    this->NC,this->c_Tb3,cNorm,
                                    this->NGZ,this->gz_Tb3,gz,
                                    this->NGC,this->gc_Tb3,gc,
                                    this->NZC,this->gzc_Tb3,gcz,
                                    this->tableValues_[7])*this->rho_[celli];   

        // // -------------------- Yis begin ------------------------------
        for (int yi=0; yi<this->NY; yi++)
        {
            word specieName2Update = this->speciesNames_table_[yi];
            const label& specieLabel2Update = this->chemistryPtr_->species()[specieName2Update];
            this->Y_[specieLabel2Update].primitiveFieldRef()[celli] = this->lookup6d(this->NH,this->h_Tb3,hLoss,
                                        this->NZ,this->z_Tb3,this->ZCells_[celli],
                                        this->NC,this->c_Tb3,cNorm,
                                        this->NGZ,this->gz_Tb3,gz,
                                        this->NGC,this->gc_Tb3,gc,
                                        this->NZC,this->gzc_Tb3,gcz,
                                        this->tableValues_[this->NS+yi]);  
        }

        // -------------------- Yis end ------------------------------

        if(baseFGM<ReactionThermo>::flameletT_)   
        {
            this->TCells_[celli] = this->lookup6d(this->NH,this->h_Tb3,hLoss,
                                        this->NZ,this->z_Tb3,this->ZCells_[celli],
                                        this->NC,this->c_Tb3,cNorm,
                                        this->NGZ,this->gz_Tb3,gz,
                                        this->NGC,this->gc_Tb3,gc,
                                        this->NZC,this->gzc_Tb3,gcz,
                                        this->tableValues_[6]);   
        }
        else
        {
            this->CpCells_[celli] = this->lookup6d(this->NH,this->h_Tb3,hLoss,
                                        this->NZ,this->z_Tb3,this->ZCells_[celli],
                                        this->NC,this->c_Tb3,cNorm,
                                        this->NGZ,this->gz_Tb3,gz,
                                        this->NGC,this->gc_Tb3,gc,
                                        this->NZC,this->gzc_Tb3,gcz,
                                        this->tableValues_[3]);   

            this->HfCells_[celli] = this->lookup6d(this->NH,this->h_Tb3,hLoss,
                                        this->NZ,this->z_Tb3,this->ZCells_[celli],
                                        this->NC,this->c_Tb3,cNorm,
                                        this->NGZ,this->gz_Tb3,gz,
                                        this->NGC,this->gc_Tb3,gc,
                                        this->NZC,this->gzc_Tb3,gcz,
                                        this->tableValues_[5]);   

            this->TCells_[celli] = (this->HCells_[celli]-this->HfCells_[celli])/this->CpCells_[celli]
                            + this->T0;   
        }

        this->TCells_[celli] = max(this->TCells_[celli], TMin);
        this->TCells_[celli] = min(this->TCells_[celli], TMax);
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

        const tmp<scalarField> tmutw = this->turbulence().mut(patchi);      
        const scalarField& pmut = tmutw();  

        fvPatchScalarField& pmagUPrime = this->magUPrime_.boundaryFieldRef()[patchi];    
        fvPatchScalarField& pchi_Zfltd = this->chi_Zfltd_.boundaryFieldRef()[patchi];    

        fvPatchScalarField& pk = k.boundaryFieldRef()[patchi];
        fvPatchScalarField& pepsilon = epsilon.boundaryFieldRef()[patchi];


        forAll(prho_, facei)   
        {
            label cellID = this->mesh().boundary()[patchi].faceCells()[facei]; 

            double hLoss = ( pZ[facei]*(this->Hfu-this->Hox) + this->Hox ) - pH[facei];
            hLoss = max(hLoss, this->h_Tb3[0]);
            hLoss = min(hLoss, this->h_Tb3[this->NH - 1]);

            ih = this->locate_lower(this->NH,this->h_Tb3,hLoss);

            Zl = this->z_Tb5[ih*this->NZL];  
            Zr = this->z_Tb5[(ih+1)*this->NZL-1];

            if(this->isLES_)
            {

                pchi_Z[facei] = sdrLRXmodel(2.0,pmut[facei]
                    /this->rho_[facei],this->deltaCells_[cellID],pZvar[facei]);
                pchi_Zc[facei]= sdrLRXmodel(2.0,pmut[facei]
                    /this->rho_[facei],this->deltaCells_[cellID],pZcvar[facei]); 
            }
            else
            {
                pchi_Z[facei] = 1.0*pepsilon[facei]/pk[facei] *pZvar[facei]; 

                pchi_Zc[facei]= 1.0*pepsilon[facei]/pk[facei] *pZcvar[facei]; 
            }


            if(pZ[facei] >= Zl && pZ[facei] <= Zr
            && this->combustion_ && pc[facei] > this->small) 
            {

                double kc_s = this->lookup2d(this->NH,this->h_Tb3,hLoss,this->NZL,this->z_Tb5,pZ[facei],this->kctau_Tb5);     
                double tau = this->lookup2d(this->NH,this->h_Tb3,hLoss,this->NZL,this->z_Tb5,pZ[facei],this->tau_Tb5);    
                double sl = this->lookup2d(this->NH,this->h_Tb3,hLoss,this->NZL,this->z_Tb5,pZ[facei],this->sl_Tb5);      
                double dl = this->lookup2d(this->NH,this->h_Tb3,hLoss,this->NZL,this->z_Tb5,pZ[facei],this->th_Tb5);      

                if(this->isLES_)
                {
                    pchi_c[facei] =  sdrFLRmodel(pcvar[facei],pmagUPrime[facei],
                                      this->deltaCells_[cellID],sl,dl,tau,kc_s,this->betacCells_[cellID]);   
                }
                else
                {
                    pchi_c[facei] =
                        this->RANSsdrFLRmodel(pcvar[facei],pepsilon[facei],
                            pk[facei],pmu[facei]/prho_[facei],
                            sl,dl,tau,kc_s,prho_[facei]);
                }
            }
            else
            {      
                if(this->isLES_)
                {
                    pchi_c[facei] = sdrLRXmodel(2.0,pmut[facei]
                        /this->rho_[facei],this->deltaCells_[cellID],pcvar[facei]);                    
                }
                else
                {
                    pchi_c[facei] = 1.0*pepsilon[facei]/(pk[facei]+SMALL)*pcvar[facei];    
                }

            }


         // = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

            double gz{this->cal_gvar(pZ[facei],pZvar[facei])};  
            double gcz{this->cal_gcor(pZ[facei],pc[facei],pZvar[facei],pcvar[facei],pZcvar[facei])},
                    Ycmax{-1.0},cNorm{},gc{};     

            if(tableSolver::scaledPV_)
            {
                cNorm = pc[facei];   
            }
            else
            {
                Ycmax = this->lookup6d(this->NH,this->h_Tb3,hLoss,
                                        this->NZ,this->z_Tb3,pZ[facei],
                                        this->NC,this->c_Tb3,0.0,
                                        this->NGZ,this->gz_Tb3,gz,
                                        this->NGC,this->gc_Tb3,0.0,
                                        this->NZC,this->gzc_Tb3,0.0,
                                        this->tableValues_[this->NS-1]);    
                Ycmax = max(this->smaller,Ycmax);  
                cNorm = pc[facei]/Ycmax;   
            }

            gc = this->cal_gvar(pc[facei],pcvar[facei],Ycmax);   

            this->He_s_.boundaryFieldRef()[patchi][facei] = hLoss;
            this->c_s_.boundaryFieldRef()[patchi][facei] = cNorm;
            this->Zvar_s_.boundaryFieldRef()[patchi][facei] = gz;
            this->cvar_s_.boundaryFieldRef()[patchi][facei] = gc;
            this->Zcvar_s_.boundaryFieldRef()[patchi][facei] = gcz;

            if(pZ[facei] >= Zl && pZ[facei] <= Zr
                && this->combustion_ && pc[facei] > this->small) 
            {
                if (this->isLES_)
                {
                    pomega_c[facei] =
                            this->lookup6d(this->NH,this->h_Tb3,hLoss,
                                        this->NZ,this->z_Tb3,pZ[facei],
                                        this->NC,this->c_Tb3,cNorm,
                                        this->NGZ,this->gz_Tb3,gz,
                                        this->NGC,this->gc_Tb3,gc,
                                        this->NZC,this->gzc_Tb3,gcz,
                                        this->tableValues_[0])    
                        + (
                                tableSolver::scaledPV_
                                ? (pchi_Z[facei]+ pchi_Zfltd[facei])*pc[facei]
                                    *this->lookup3d(this->NH,this->h_Tb3,hLoss,this->NZ,this->z_Tb3,pZ[facei],
                                                    this->NGZ,this->gz_Tb3,gz,this->d2Yeq_Tb2)
                                    + 2.0*pchi_Zc[facei]*this->lookup3d(this->NH,this->h_Tb3,hLoss,this->NZ,this->z_Tb3,pZ[facei],
                                    this->NGZ,this->gz_Tb3,gz,this->d1Yeq_Tb2)
                                : 0.0
                            );  
                }
                else
                {
                    pomega_c[facei] =
                            this->lookup6d(this->NH,this->h_Tb3,hLoss,
                                        this->NZ,this->z_Tb3,pZ[facei],
                                        this->NC,this->c_Tb3,cNorm,
                                        this->NGZ,this->gz_Tb3,gz,
                                        this->NGC,this->gc_Tb3,gc,
                                        this->NZC,this->gzc_Tb3,gcz,
                                        this->tableValues_[0])   
                        + (
                                tableSolver::scaledPV_
                                ? ( pchi_Z[facei]*pc[facei]
                                *this->lookup3d(this->NH,this->h_Tb3,hLoss,this->NZ,this->z_Tb3,pZ[facei],
                                                this->NGZ,this->gz_Tb3,gz,this->d2Yeq_Tb2)
                                + 2.0*pchi_Zc[facei]*this->lookup3d(this->NH,this->h_Tb3,hLoss,this->NZ,this->z_Tb3,pZ[facei],
                                    this->NGZ,this->gz_Tb3,gz,this->d1Yeq_Tb2)  )
                                : 0.0
                            );  
                }

                pcOmega_c[facei] = this->lookup6d(this->NH,this->h_Tb3,hLoss,
                                                    this->NZ,this->z_Tb3,pZ[facei],
                                                    this->NC,this->c_Tb3,cNorm,
                                                    this->NGZ,this->gz_Tb3,gz,
                                                    this->NGC,this->gc_Tb3,gc,
                                                    this->NZC,this->gzc_Tb3,gcz,this->tableValues_[1]);

                pZOmega_c[facei] = this->lookup6d(this->NH,this->h_Tb3,hLoss,
                                            this->NZ,this->z_Tb3,pZ[facei],
                                            this->NC,this->c_Tb3,cNorm,
                                            this->NGZ,this->gz_Tb3,gz,
                                            this->NGC,this->gc_Tb3,gc,
                                            this->NZC,this->gzc_Tb3,gcz,this->tableValues_[2]);
            }    
            else
            {
                pomega_c[facei] = 0.0;
                pcOmega_c[facei] = 0.0;
                pZOmega_c[facei] = 0.0;
            }
            pomega_c[facei] = pomega_c[facei]*prho_[facei];    
            pcOmega_c[facei] = pcOmega_c[facei]*prho_[facei];
            pZOmega_c[facei] = pZOmega_c[facei]*prho_[facei];

            pWt[facei] = this->lookup6d(this->NH,this->h_Tb3,hLoss,
                                        this->NZ,this->z_Tb3,pZ[facei],
                                        this->NC,this->c_Tb3,cNorm,
                                        this->NGZ,this->gz_Tb3,gz,
                                        this->NGC,this->gc_Tb3,gc,
                                        this->NZC,this->gzc_Tb3,gcz,
                                        this->tableValues_[4]);    

            pmu[facei] = this->lookup6d(this->NH,this->h_Tb3,hLoss,
                                        this->NZ,this->z_Tb3,pZ[facei],
                                        this->NC,this->c_Tb3,cNorm,
                                        this->NGZ,this->gz_Tb3,gz,
                                        this->NGC,this->gc_Tb3,gc,
                                        this->NZC,this->gzc_Tb3,gcz,
                                        this->tableValues_[7])*prho_[facei];  

            if(baseFGM<ReactionThermo>::flameletT_)  
            {
                pT[facei] = this->lookup6d(this->NH,this->h_Tb3,hLoss,
                                        this->NZ,this->z_Tb3,pZ[facei],
                                        this->NC,this->c_Tb3,cNorm,
                                        this->NGZ,this->gz_Tb3,gz,
                                        this->NGC,this->gc_Tb3,gc,
                                        this->NZC,this->gzc_Tb3,gcz,
                                        this->tableValues_[6]);   
            }
            else
            {
                pCp[facei] = this->lookup6d(this->NH,this->h_Tb3,hLoss,
                                        this->NZ,this->z_Tb3,pZ[facei],
                                        this->NC,this->c_Tb3,cNorm,
                                        this->NGZ,this->gz_Tb3,gz,
                                        this->NGC,this->gc_Tb3,gc,
                                        this->NZC,this->gzc_Tb3,gcz,
                                        this->tableValues_[3]);   

                pHf[facei] = this->lookup6d(this->NH,this->h_Tb3,hLoss,
                                        this->NZ,this->z_Tb3,pZ[facei],
                                        this->NC,this->c_Tb3,cNorm,
                                        this->NGZ,this->gz_Tb3,gz,
                                        this->NGC,this->gc_Tb3,gc,
                                        this->NZC,this->gzc_Tb3,gcz,
                                        this->tableValues_[5]);  

                pT[facei] = (pH[facei]-pHf[facei])/pCp[facei]
                            + this->T0;  
            }

            pT[facei] = max(pT[facei], TMin);
            pT[facei] = min(pT[facei], TMax);


            // // -------------------- Yis begin ------------------------------

            for (int yi=0; yi<this->NY; yi++)
            {
                word specieName2Update = this->speciesNames_table_[yi];
                const label& specieLabel2Update = this->chemistryPtr_->species()[specieName2Update];
                this->Y_[specieLabel2Update].boundaryFieldRef()[patchi][facei]  = this->lookup6d(this->NH,this->h_Tb3,hLoss,
                                            this->NZ,this->z_Tb3,pZ[facei],
                                            this->NC,this->c_Tb3,cNorm,
                                            this->NGZ,this->gz_Tb3,gz,
                                            this->NGC,this->gc_Tb3,gc,
                                            this->NZC,this->gzc_Tb3,gcz,
                                            this->tableValues_[this->NS+yi]);  
            }

            // -------------------- Yis end ------------------------------
        } 
    }

    // this->T_.max(TMin);  
    // this->T_.min(TMax);  

    double R_uniGas_ = 8.314e3;
    double p_operateDim_ = this->coeffs().lookupOrDefault("p_operateDim", this->incompPref_);

    forAll(this->rho_.boundaryFieldRef(), patchi)   
    {
        fvPatchScalarField& ppsi_ = this->psi_.boundaryFieldRef()[patchi];
        fvPatchScalarField& prho_ = this->rho_.boundaryFieldRef()[patchi];
        fvPatchScalarField& pWt = this->Wt_.boundaryFieldRef()[patchi];
        fvPatchScalarField& pT = this->T_.boundaryFieldRef()[patchi];
        fvPatchScalarField pp_ = this->p_.boundaryField()[patchi];

        forAll(prho_, facei)   
        {
            ppsi_[facei] = pWt[facei] / (R_uniGas_*pT[facei]);

            if(this->incompPref_ > 0.0) 
            {  
                prho_[facei] = p_operateDim_*ppsi_[facei]; 
            }
            else 
            {
                prho_[facei] = pp_[facei]*ppsi_[facei];
            }
        }
    }

    dimensionedScalar R_uniGas("R_uniGas",dimensionSet(1,2,-2,-1,-1,0,0),8.314e3);
    this->psi_ = this->Wt_/(R_uniGas*this->T_);
  
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
    this->rho_inRhoThermo_ = this->rho_;

}

// ************************************************************************* //
