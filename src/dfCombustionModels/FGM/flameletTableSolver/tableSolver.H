/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2021 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License

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
    Foam::tableSolver

Description
    class for the interface between table look-up and combustion model.

SourceFiles
    tableSolver.C
\*---------------------------------------------------------------------------*/

#ifndef tableSolver_H
#define tableSolver_H


#include "hashedWordList.H"
#include "labelList.H"
#include "scalarList.H"
#include "scalarField.H"
#include "scalar.H"
#include "volFields.H"
#include "dimensionedScalar.H"
#include "Switch.H"
#include "PtrList.H"
#include <mpi.h>
#include <vector>

namespace Foam
{

class tableSolver
{
 //- protected data
 protected:
    double small;
    double smaller;
    double smallest;
    double T0;


    //- create table
    FILE *table;
    word tablePath_;

    //- list of MPI windows, used for shared memory
    List<MPI_Win> wins_;

    //- self-defined species name
    hashedWordList speciesNames_table_;
    hashedWordList spc_omegaNames_table_;

    //- List of the used tables
    hashedWordList tableNames_;

    //- tables of all tabulated properties, by tableNames_'s order
    List<scalar*> tableValues_;

    double *d2Yeq_Tb2, *d1Yeq_Tb2;

    //- size of one signle table
    unsigned long long int singleTableSize_;

   //- enthalpy solve witch
    Switch scaledPV_;

    //- T-lookup switch
    // Switch flameletT_;

    //- unscaled max C
    double cMaxAll_;
    //PtrList<scalarField> Yi_Tb3;

    int NZL,NH,NZ,NC,NGZ,NGC,NZC,NY,NS,NYomega;

    double Hox,Hfu,prog;

    int fHox_fu, fNZL, fNH, fNZ, fNC, fNGZ, fNGC, fNZC, fNS_NY;

    int f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10;

    //- fuel and oxidizer enthalpy
    const dimensionedScalar H_fuel;
    const dimensionedScalar H_ox;


    double  *z_Tb5,     //- interpolation z_space
            *sl_Tb5,    //- laminar flame speed
            *th_Tb5,    //- laminar flame thickness
            *tau_Tb5,   //- heat release parameter
            *kctau_Tb5; //- for calculating chi_c

    double  *h_Tb3,     //- h loss space
            *z_Tb3,     //- z space
            *c_Tb3,     //- c space
            *gz_Tb3,    //- z-variance space
            *gc_Tb3,    //- c-variance space
            *gzc_Tb3;   //- z-c co-variance space


    double *omgc_Tb3,    //- chemistry reaction source rate
           *cOc_Tb3,     //- c*omega_c  
           *ZOc_Tb3,     //- z*omega_c  
           *cp_Tb3,      //- heat capacity
           *hiyi_Tb3,    //- formation enthalpy
           *mwt_Tb3,     //- mixture mole mass fraction  
           *Tf_Tb3,      //- flame temperature
           *nu_Tb3,      //- viscosity
           *Ycmax_Tb3;   //- maximum progress variable

    double *Yi01_Tb3,    //- mass fraction of H2O
           *Yi02_Tb3,    //- mass fraction of CO
           *Yi03_Tb3;    //- mass fraction of CO2


    // double *d2Yeq_Tb2;


  public:

    //- Constructor
     tableSolver(word tablePath_);

    //- Member function 

    //- solve variance 
    double cal_gvar(double mean, double var, double Ycmax = -1.0);

    //- Compute correlation coefficient between Z and c
    double cal_gcor(double Z, double c, double Zvar, double cvar, double Zcvar);

    //- Locate lower index for 1D linear interpolation
    int locate_lower(int n, double array[], double x); 

    //- Locate higher index for 1D linear interpolation
    int locate_higher(int n, double array[], double x); 

    //- Compute 1D linear interpolation factor [for Hloss]
    double intfac_high(double xx, double high, double low);

    //- Compute 1D linear interpolation factor
    double intfac(double xx, double low, double high);

    //- 1D linear interpolation
    double interp1d(int n1, int loc_z, double zfac, double table_1d[]); 

    //- 5D linear interpolation
    double interp5d(int nz, int nc, int ngz, int ngc, int ngcz
                    ,int loc_z, int loc_c, int loc_gz, int loc_gc, int loc_gcz
                    ,double zfac, double cfac, double gzfac, double gcfac
                    ,double gczfac, double table_5d[]);

    //- 6D linear interpolation
    double interp6d(int nh,int nz, int nc, int ngz, int ngc, int ngcz
                    , int loc_h, int loc_z, int loc_c, int loc_gz, int loc_gc, int loc_gcz
                    , double hfac, double zfac, double cfac, double gzfac, double gcfac
                    ,double gczfac, double table_6d[]);

    //- 3D linear interpolation
    double interp3d(int n0, int n1, int n2, int loc_h, int loc_z, int loc_gz, double hfac,double zfac
                    ,double gzfac, double table3d[]); 

   //- 2D linear interpolation
    double interp2d(int n1, int n2, int loc_z, int loc_gz, double zfac
                    ,double gzfac, double table2d[]); 


   //- SDR model for progress variable
    double sdrFLRmodel(double cvar, double uSgs_pr, double filterSize
                      ,double sl, double dl, double tau, double kc_s
                      ,double beta);


   //- SDR model for passive scalar
    double sdrLRXmodel(double Csdr, double nut, double delta, double var);


   //- 1D table look-up
    double lookup1d(int n1, double list_1[], double x1, double table_1d[]);

   //- 2D table look-up
    double lookup2d(int n1, double list_1[], double x1,int n2,
                    double list_2[], double x2, double table_2d[]);

   //- 3D table look-up
    double lookup3d(int n1, double list_1[], double x1,int n2,
                    double list_2[], double x2, int n3, double list_3[], double x3, double table_3d[]);

   //- 5D table look-up
    double lookup5d(int n1, double list_1[], double x1,
                    int n2, double list_2[], double x2,
                    int n3, double list_3[], double x3,
                    int n4, double list_4[], double x4,
                    int n5, double list_5[], double x5,
                    double table_5d[]); 

    //- 6D table look-up
    double lookup6d(int n1, double list_1[], double x1,
                    int n2, double list_2[], double x2,
                    int n3, double list_3[], double x3,
                    int n4, double list_4[], double x4,
                    int n5, double list_5[], double x5,
                    int n6, double list_6[], double x6,
                    double table_5d[]); 


   //- RANS SDR model for progress variable
    double RANSsdrFLRmodel(double cvar, double epsilon, double k, double nu,
                          double sl, double dl, double tau, double kc_s,double rho);

   //- Destructor
    virtual ~tableSolver();

};



} // End Foam namespace

#endif /* TABLESOLVER_H_ */
