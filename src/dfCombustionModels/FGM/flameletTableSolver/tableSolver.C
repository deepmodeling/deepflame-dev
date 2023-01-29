/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2021 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software; you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the
    Free Software Foundation; either version 2 of the License, or (at your
    option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM; if not, write to the Free Software Foundation,
    Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA

\*---------------------------------------------------------------------------*/


#include "tableSolver.H"
#include "dictionary.H"
#include "IFstream.H"
#include "Pstream.H"
#include "clockTime.H"


namespace Foam
{

//-

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

//tableSolver::tableSolver(const wordList& tableNames, string suffix_)
tableSolver::tableSolver(wordList speciesNames,Switch& scaledPV, Switch flameletT, scalar& cMaxAll)
:
small(1.0e-4),
smaller(1.0e-6),
smallest(1.0e-12),
T0(298.15),
table(fopen("./flare.tbl", "r")),
speciesNames_(speciesNames),
scaledPV_(scaledPV),
flameletT_(flameletT),
cMaxAll_(cMaxAll),
fHox_fu(fscanf(table, "%lf %lf",&Hfu,&Hox)),
fNZL(fscanf(table, "%d",&NZL)),
H_fuel("H_fuel",dimensionSet(0,2,-2,0,0,0,0),Hfu),
H_ox("H_ox",dimensionSet(0,2,-2,0,0,0,0),Hox)
{

  #include "readThermChemTables.H"

}

// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

tableSolver::~tableSolver()
{

}

// * * * * * * * * * * * * * *  Member Functions * * * * * * * * * * * * * * //

//- solve variance 
double Foam::tableSolver::cal_gvar
(
    double mean, 
    double var, 
    double Ycmax
)
{
    double gvar;

    if(Ycmax < 0.0)
    {
        if(mean < small || mean > (1.0-small)) gvar = 0.0;
        else gvar = var/(mean*(1.-mean)) ;
    }
    else
    {
        if(mean < small || mean > (1.0-small)) gvar = 0.0;
        else gvar = var/(mean*(Ycmax-mean)) ;
    }

    gvar = min(1.0,gvar);
    gvar = max(smaller,gvar);

    return gvar;    

}

double Foam::tableSolver::cal_gcor
(
    double Z, 
    double c, 
    double Zvar, 
    double cvar, 
    double Zcvar
)
{
    double gcor;

    if(abs(Zcvar) < 1e-12)
    {
        gcor=0.0;
    }
    else if(abs(cvar) < 1e-12 || abs(Zvar) < 1e-12)
    {
        gcor=0.0;
    }
    else
    {
        gcor = (Zcvar - Z * c) /(Foam::sqrt(Zvar * cvar));
    }

    gcor = fmin(1.0,gcor);
    gcor = fmax(-1.0,gcor);

    return gcor;    
}

int Foam::tableSolver::locate_lower
(
    int n, 
    double array[], 
    double x
)
{
        int i,loc;

        loc = 0;
        if(x <= array[loc]) return loc;

        loc = n-2;
        if(x >= array[loc+1]) return loc;

        for(i=0; i<n-1; i++)
        {
            if(x >= array[i] && x < array[i+1])
            {
                loc = i;
                return loc;
            }
        }

        return loc;    
}

double Foam::tableSolver::intfac
(
     double xx, 
     double low, 
     double high
)
{
        double fac;

        if(xx <= low) fac = 0.0;
        else if(xx >= high) fac = 1.0;
        else fac = (xx-low)/(high-low);

        return fac;    
}

double Foam::tableSolver::interp1d
(
  int n1, 
  int loc_z,
  double zfac, 
  double table_1d[]
)
{
        int i1,j1;
        double factor,result;

        result =0.0;
        for(i1=0; i1<2; i1++)
        {
            factor = (1.0-zfac+i1*(2.0*zfac-1.0));

            if(i1 == 1) j1 = loc_z+1;
            else j1 = loc_z;

            result = result + factor*table_1d[j1];
        }

        return result;    
}

double Foam::tableSolver::interp5d
(
    int nz, int nc, int ngz, int ngc, int ngcz,
    int loc_z, int loc_c, int loc_gz, int loc_gc,
    int loc_gcz,double zfac, double cfac, double gzfac, 
    double gcfac,double gczfac, double table_5d[]
)
{
        int i1,i2,i3,i4,i5,j1,j2,j3,j4,j5,loc;
        double factor, result;

        result =0.0;
        for(i1=0; i1<2; i1++)
        {
            if(i1 == 1) j1 = loc_z+1;
            else j1 = loc_z;

            for(i2=0; i2<2; i2++)
            {
                if(i2 == 1) j2 = loc_c+1;
                else j2 = loc_c;

                for(i3=0; i3<2; i3++)
                {
                    if(i3 == 1) j3 = loc_gz+1;
                    else j3 = loc_gz;

                    for(i4=0; i4<2; i4++)
                    {
                        if(i4 == 1) j4 = loc_gc+1;
                        else j4 = loc_gc;

                        for(i5=0; i5<2; i5++)
                        {
                            factor = (1.0-zfac+i1*(2.0*zfac-1.0))
                                    *(1.0-cfac+i2*(2.0*cfac-1.0))
                                    *(1.0-gzfac+i3*(2.0*gzfac-1.0))
                                    *(1.0-gcfac+i4*(2.0*gcfac-1.0))
                                    *(1.0-gczfac+i5*(2.0*gczfac-1.0));

                            if(i5 == 1) j5 = loc_gcz+1;
                            else j5 = loc_gcz;

                            loc = j1*nc*ngz*ngc*ngcz
                                 +j2*ngz*ngc*ngcz
                                 +j3*ngc*ngcz
                                 +j4*ngcz
                                 +j5;
                            result = result + factor*table_5d[loc] ;
                        }
                    }
                }
            }
        }

        return result;    
}

double Foam::tableSolver::interp2d
(
    int n1, int n2, int loc_z, int loc_gz,
    double zfac,double gzfac, double table2d[]
)
{
        int i1,i2,j1,j2,loc;
        double factor,result;

        result = 0.0;
        for(i1=0; i1<2; i1++)
        {
            if(i1 == 1) j1 = loc_z+1;
            else j1 = loc_z;

            for(i2=0; i2<2; i2++)
            {
                factor = (1.0-zfac+i1*(2.0*zfac-1.0))
                        *(1.0-gzfac+i2*(2.0*gzfac-1.0));

                if(i2 == 1) j2 = loc_gz+1;
                else j2 = loc_gz;

                loc = j1*n2+j2;
                result = result + factor*table2d[loc];
            }
        }

        return result;

}


double Foam::tableSolver::sdrFLRmodel
(
    double cvar, double uSgs_pr, double filterSize,
    double sl, double dl, double tau, double kc_s,double beta
)
{
        double KaSgs,c3,c4,kc,DaSgs,theta_5;
        //- The model parameter K_c^* is also obtained from the laminar flame calculation [Kolla et al., 2009] 
        //- and this parameter varies with Z for partially premixed flames
        kc=kc_s*tau;  
        theta_5=0.75;
        if(uSgs_pr < 1.0E-6) uSgs_pr = smaller;

        KaSgs = Foam::pow((uSgs_pr/sl),1.5) * Foam::pow((dl/filterSize),0.5);
        DaSgs = sl*filterSize/uSgs_pr/dl;

        c3=1.5/(1.0/Foam::sqrt(KaSgs)+1.0);
        c4=1.1/Foam::pow((1.0+KaSgs),0.4);

        return (1-Foam::exp(-theta_5*filterSize/dl))*cvar/beta
                            *( 2.0*kc*sl/dl + (c3 - tau*c4*DaSgs)
                                              *(2.0/3.0*uSgs_pr/filterSize) );    
}


double Foam::tableSolver::sdrLRXmodel
(
   double Csdr, double nut, double delta, double var    
)
{
        return Csdr*nut/sqr(delta)*var;    
}

double Foam::tableSolver::lookup1d
(
    int n1, double list_1[], double x1, double table_1d[]
)
{
        int loc_1 = locate_lower(n1,list_1,x1);   
        double fac_1 = intfac(x1,list_1[loc_1],list_1[loc_1+1]);  
        return interp1d(n1,loc_1,fac_1,table_1d);  
}

double Foam::tableSolver::lookup2d
(
    int n1, double list_1[], double x1,
    int n2, double list_2[], double x2,
    double table_2d[]    
)
{
        int loc_1 = locate_lower(n1,list_1,x1);
        double fac_1 = intfac(x1,list_1[loc_1],list_1[loc_1+1]);
        int loc_2 = locate_lower(n2,list_2,x2);
        double fac_2 = intfac(x2,list_2[loc_2],list_2[loc_2+1]);

        return interp2d(n1,n2,loc_1,loc_2,fac_1,fac_2,table_2d);    
}

double Foam::tableSolver::lookup5d
(
    int n1, double list_1[], double x1,
    int n2, double list_2[], double x2,
    int n3, double list_3[], double x3,
    int n4, double list_4[], double x4,
    int n5, double list_5[], double x5,
    double table_5d[]
)
{
        int loc_1 = locate_lower(n1,list_1,x1);
        double fac_1 = intfac(x1,list_1[loc_1],list_1[loc_1+1]);
        int loc_2 = locate_lower(n2,list_2,x2);
        double fac_2 = intfac(x2,list_2[loc_2],list_2[loc_2+1]);
        int loc_3 = locate_lower(n3,list_3,x3);
        double fac_3 = intfac(x3,list_3[loc_3],list_3[loc_3+1]);
        int loc_4 = locate_lower(n4,list_4,x4);
        double fac_4 = intfac(x4,list_4[loc_4],list_4[loc_4+1]);
        int loc_5 = locate_lower(n5,list_5,x5);
        double fac_5 = intfac(x5,list_5[loc_5],list_5[loc_5+1]);

        return interp5d(n1,n2,n3,n4,n5,loc_1,loc_2,loc_3,loc_4,loc_5,
            fac_1,fac_2,fac_3,fac_4,fac_5,table_5d);    
}


double Foam::tableSolver::RANSsdrFLRmodel
(
    double cvar, double epsilon, double k, double nu,
    double sl, double dl, double tau, double kc_s,double rho
)
{
        double beta=6.7, C3, C4, Kc, Ka;

        Ka=(dl/sl)/Foam::sqrt(nu/epsilon);
        Kc=kc_s*tau;
        C3=1.5*Foam::sqrt(Ka)/(1+Foam::sqrt(Ka));
        C4=1.1/Foam::pow((1+Ka),0.4);
        
        return rho/beta*cvar*( (2.0*Kc-tau*C4)*sl/dl + C3*epsilon/k );    
}



} // End Foam namespace