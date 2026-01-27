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
#include "scalar.H"

namespace Foam
{

//-

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

tableSolver::tableSolver(word tablePath)
:
small(1.0e-4),
smaller(1.0e-6),
smallest(1.0e-12),
T0(298.15),
H_fuel("H_fuel",dimensionSet(0,2,-2,0,0,0,0),Hfu),
H_ox("H_ox",dimensionSet(0,2,-2,0,0,0,0),Hox)
{
    Info<< "* * * * * * * * * * Opening FlaRe table * * * * * * * * * * \n" << endl;
    std::ifstream table(tablePath.c_str());
    if (table.is_open())
    {
        std::string line;

        if (std::getline(table, line))
        {
            std::istringstream iss(line);
            iss >> NH >> NZ >> NC >> NGZ >> NGC >> NZC >> NS >> NYomega >> NY >> NZL;
        }
        Info<< "Reading: NH=" << NH << ", NZ=" << NZ << ", NC="<< NC << ", NGZ=" << NGZ << ", NGC=" 
            << NGC << ", NZC="<< NZC << ", NS=" << NS << ", NYomega=" << NYomega << ", NY=" << NY << ", NZL=" << NZL << "\n" << endl;

        tableNames_ = wordList({"omgc_Tb3", "cOc_Tb3", "ZOc_Tb3", "cp_Tb3", "mwt_Tb3", "hiyi_Tb3", "Tf_Tb3", "nu_Tb3"});

        if(NS == 8+NYomega)
        {
            scaledPV_ = true;
            Info<< "=============== Using scaled PV ==============="
                << "\n" << endl;
        }
        else if(NS == 9+NYomega)
        {
            scaledPV_ = false;
            tableNames_.append("Ycmax_Tb3");
            Info<< "=============== Using unscaled PV ==============="
                << "\n" << endl;
        }
        else
        {
            WarningInFunction << "Number of columns wrong in flare.tbl !!!"
                                << "\n" << endl;
        }


        if (std::getline(table, line))
        {
            std::istringstream iss(line);
            std::string spc_name;
            for (int ii=0; ii<NYomega; ++ii)
            {
                iss >> spc_name;
                spc_omegaNames_table_.append(spc_name);
            }
        }

        if (std::getline(table, line))
        {
            std::istringstream iss(line);
            std::string spc_name;
            for (int ii=0; ii<NY; ++ii)
            {
                iss >> spc_name;
                speciesNames_table_.append(spc_name);
                tableNames_.append(spc_name);
            }
        }


        Info << "Load omega of species: " << spc_omegaNames_table_ << endl;
        Info << "Load species: " << speciesNames_table_ << endl;

        h_Tb3 = { new double[NH]{} };  z_Tb3 = { new double[NZ]{} }; 
        c_Tb3 = { new double[NC]{} };  gz_Tb3 = { new double[NGZ]{} };
        gc_Tb3 = { new double[NGC]{} };  gzc_Tb3 = { new double[NZC]{} };  

        for(int ii = 0; ii < NH; ++ii)
        {
            if (std::getline(table, line))
            {
                std::istringstream iss(line);
                iss >> h_Tb3[ii];
            }
        }
        for(int ii = 0; ii < NZ; ++ii)
        {
            if (std::getline(table, line))
            {
                std::istringstream iss(line);
                iss >> z_Tb3[ii];
            }
        }
        for(int ii = 0; ii < NC; ++ii)
        {
            if (std::getline(table, line))
            {
                std::istringstream iss(line);
                iss >> c_Tb3[ii];
            }
        }
        for(int ii = 0; ii < NGZ; ++ii)
        {
            if (std::getline(table, line))
            {
                std::istringstream iss(line);
                iss >> gz_Tb3[ii];
            }
        }
        for(int ii = 0; ii < NGC; ++ii)
        {
            if (std::getline(table, line))
            {
                std::istringstream iss(line);
                iss >> gc_Tb3[ii];
            }
        }
        for(int ii = 0; ii < NZC; ++ii)
        {
            if (std::getline(table, line))
            {
                std::istringstream iss(line);
                iss >> gzc_Tb3[ii];
            }
        }
        
        if (std::getline(table, line))
        {
            std::istringstream iss(line);
            iss >> Hfu >> Hox;
        }
        Info<< "Reading: H_fuel=" << Hfu << ", H_ox=" << Hox << "\n" << endl;

        z_Tb5 = { new double[NH*NZL]{} }; sl_Tb5 = { new double[NH*NZL]{} };
        th_Tb5 = { new double[NH*NZL]{} }; tau_Tb5 = { new double[NH*NZL]{} };
        kctau_Tb5 = { new double[NH*NZL]{} };

        Info<< "Reading laminar flame properties\n" << endl;
        int count = 0;
        for (int ii=0; ii<NH; ++ii)
        {
            for(int jj = 0; jj < NZL; ++jj)  
            {
                if (std::getline(table, line))
                {
                    std::istringstream iss(line);
                    iss >> z_Tb5[count] >> sl_Tb5[count] >> th_Tb5[count]
                        >> tau_Tb5[count] >> kctau_Tb5[count];
                    count++;
                }
            }
        }

        Info<< "Reading turbulence flame properties\n" << endl;

        singleTableSize_ = NH*NZ*NC*NGZ*NGC*NZC;

        if (Pstream::parRun()) // parallel computing
        {
            // Create node-local communicator
            MPI_Comm nodecomm;
            MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, Pstream::myProcNo(), MPI_INFO_NULL, &nodecomm);

            int noderank;
            MPI_Comm_rank(nodecomm, &noderank);

            // wins_.resize(wins_.size()+1);
            // const int disp_unit = sizeof(double);

            // scalar *table_ptr;

            // Initialise table on node-master with appropriate synchronisation
            // MPI_Win_fence(0, wins_.last());
            // MPI_Barrier(MPI_COMM_WORLD);

            // if (noderank == 0) // node-master
            // {
            //     #include "readThermChemTables.H"
            // } // end if (noderank == 0)

            // MPI_Barrier(MPI_COMM_WORLD);

            std::vector<std::vector<double>> value_temp;

            if (noderank == 0)
            {
                #include "readThermChemTables.H"
            }
            else
            {
                for (int ii=0;ii<singleTableSize_; ++ii) std::getline(table, line);
            }
            
            MPI_Barrier(MPI_COMM_WORLD);
            // MPI_Win_fence(0, wins_.last());

            for (int ivar=0; ivar<NS+NY; ++ivar)
            {
                scalar *table_ptr;
                wins_.resize(wins_.size()+1);
                int disp_unit = sizeof(double);
                
                // Only rank 0 on a node actually allocates memory
                MPI_Win_allocate_shared(noderank==0 ? singleTableSize_/1.0*disp_unit : 0, disp_unit, MPI_INFO_NULL, 
                                        nodecomm, &table_ptr, &wins_.last());

                if (noderank != 0) // node-slave
                {
                    MPI_Aint winsize;
                    int windisp;
                    // table on node-slave, let it points to the allocated address of table on node-master
                    MPI_Win_shared_query(wins_.last(), 0, &winsize, &windisp, &table_ptr);
                }

                MPI_Win_fence(0, wins_.last());

                if (noderank == 0)
                {
                    std::copy(value_temp[ivar].begin(), value_temp[ivar].end(), table_ptr);
                    value_temp[ivar].clear();  //清空内部向量的元素
                    std::vector<double>().swap(value_temp[ivar]);  //使用临时向量进行swap，收缩容量
                }

                Info<<"ivar = "<<ivar<<endl;
                MPI_Win_fence(0, wins_.last());
                tableValues_.append(table_ptr);
                table_ptr = nullptr;
            }
                

        }
        else  // 1 core computing
        {
            std::vector<std::vector<double>> value_temp;

            #include "readThermChemTables.H"

            double *table_ptr;

            // double *table_ptr = nullptr;
            for (int ivar=0; ivar<NS+NY; ++ivar)
            {
                table_ptr = new double[singleTableSize_];

                std::copy(value_temp[ivar].begin(), value_temp[ivar].end(), table_ptr);

                tableValues_.append(table_ptr); // let tableValues_ take over that piece of memory

                // table_ptr = nullptr; // de-pointer with the allocated memory
            }
        }

        Info<<"end reading turbulence properties"<<endl;

        //- findthe maxmum PV value from Ycmax_Tb3
        if (this->scaledPV_)
        {
            cMaxAll_ = 1.0;
        }
        else
        {
            const scalar Ycmaxall{*std::max_element(tableValues_[NS-1],tableValues_[NS-1]+singleTableSize_)};   
            cMaxAll_ = Ycmaxall;
            Info<< "\nunscaled PV -- Ycmaxall = "<<Ycmaxall<< endl;
        }
        
        if (this->scaledPV_)
        {
            // dYeq_Tb2 = std::vector<std::vector<double>>(2,std::vector<double> (NH*NZ*NGZ));
            d2Yeq_Tb2 = { new double[NH*NZ*NGZ]{} }; d1Yeq_Tb2 = { new double[NH*NZ*NGZ]{} };
            Info<< "\nReading non-premixed properties\n" << endl;   //屏幕提示：读取非预混的属性
            count = 0;
            for (int hh=0; hh<NH; ++hh)
            {
                for(int ii = 0; ii < NZ; ++ii)
                {
                    for(int jj = 0; jj < NGZ; ++jj)
                    {
                        if (std::getline(table, line))
                        {
                            std::istringstream iss(line);
                            iss >> d2Yeq_Tb2[count] >> d1Yeq_Tb2[count];
                            count++;
                        }
                    }
                }
            }
        }
        else
        {
            Info << "no need to Reading non-premixed properties"<<endl;
        }


        table.close();
        Info<< "* * * * * * * * * * Complete FlaRe table * * * * * * * * * " << endl;
    }// end if (table.is_open())
    else
    {
        Info << "Error: Unable to open chemTab file!" << endl;
    }


}

// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

tableSolver::~tableSolver()
{
    if (Pstream::parRun())
    {
        forAll(wins_, tableI)
        {
            MPI_Win_free(&wins_[tableI]);
            // after free window object, we cannot delete variables located in this windows
            // because the memory has already been released
        }
    }
    else
    {
        forAll(tableValues_, tableI)
        {
            delete[] tableValues_[tableI];
            tableValues_[tableI] = nullptr;
        }
    }
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

    if ( (cvar < 1.0e-4) or (Zvar < 1.0e-6) )
    {
        gcor=0.0;
    }
    else
    {
        gcor = (Zcvar) / (Foam::sqrt(Zvar) * Foam::sqrt(cvar));
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

        if (n == 1) return 0;

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

int Foam::tableSolver::locate_higher
(
    int n, 
    double array[], 
    double x
)
{
        int i,loc;

        if (n == 1) return 0;

        loc = 0;
        if(x >= array[loc]) return loc;

        loc = n-2;
        if(x <= array[loc+1]) return loc;

        for(i=0; i<n-1; i++)
        {
            if(x <= array[i] && x > array[i+1])
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
        else if((high-low) < 1e-30) fac = 0.0;
        else fac = (xx-low)/(high-low);

        return fac;    
}

double Foam::tableSolver::intfac_high
(
     double xx,
     double high, 
     double low
)
{
        double fac;

        if(xx >= high) fac = 0.0;
        else if(xx <= low) fac = 1.0;
        else fac = (high-xx)/(high-low);

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

/*------------------------------------------------------------------------*\
                        6D linear interpolation
\*------------------------------------------------------------------------*/
double Foam::tableSolver::interp6d
(
    int nh, int nz, int nc, int ngz, int ngc, int ngcz,
    int loc_h, int loc_z, int loc_c, int loc_gz, int loc_gc, int loc_gcz,
    double hfac, double zfac, double cfac, double gzfac, double gcfac,
    double gczfac, double table_6d[]
)
{
    int i0,i1,i2,i3,i4,i5,j0,j1,j2,j3,j4,j5,loc;
    double factor, result;

    result =0.0;
    for (i0=0; i0<2; i0++)
    {
        if(i0 == 1 and nh>1) j0 = loc_h+1;
        else j0 = loc_h;

        for(i1=0; i1<2; i1++)
        {
            if(i1 == 1 and nz>1) j1 = loc_z+1;
            else j1 = loc_z;

            for(i2=0; i2<2; i2++)
            {
                if(i2 == 1 and nc>1) j2 = loc_c+1;
                else j2 = loc_c;

                for(i3=0; i3<2; i3++)
                {
                    if(i3 == 1 and ngz>1) j3 = loc_gz+1;
                    else j3 = loc_gz;

                    for(i4=0; i4<2; i4++)
                    {
                        if(i4 == 1 and ngc>1) j4 = loc_gc+1;
                        else j4 = loc_gc;

                        for(i5=0; i5<2; i5++)
                        {
                            factor = (1.0-hfac+i0*(2.0*hfac-1.0))
                                    *(1.0-zfac+i1*(2.0*zfac-1.0))
                                    *(1.0-cfac+i2*(2.0*cfac-1.0))
                                    *(1.0-gzfac+i3*(2.0*gzfac-1.0))
                                    *(1.0-gcfac+i4*(2.0*gcfac-1.0))
                                    *(1.0-gczfac+i5*(2.0*gczfac-1.0));

                            if(i5 == 1 and ngcz>1) j5 = loc_gcz+1;
                            else j5 = loc_gcz;

                            loc = j0*nz*nc*ngz*ngc*ngcz
                                +j1*nc*ngz*ngc*ngcz
                                +j2*ngz*ngc*ngcz
                                +j3*ngc*ngcz
                                +j4*ngcz
                                +j5;
                            result = result + factor*table_6d[loc] ;
                        }
                    }
                }
            }
        }
    }

    return result;
}

double Foam::tableSolver::interp3d
(
    int n0, int n1, int n2, int loc_h, int loc_z, int loc_gz,
    double hfac, double zfac,double gzfac, double table3d[]
)
{
    int i0,i1,i2,j0,j1,j2,loc;
    double factor,result;

    result = 0.0;
    for (i0=0; i0<2; i0++)
    {
        if(i0 == 1 and n0>1) j0 = loc_h+1;
        else j0 = loc_h;

        for(i1=0; i1<2; i1++)
        {
            if(i1 == 1 and n1>1) j1 = loc_z+1;
            else j1 = loc_z;

            for(i2=0; i2<2; i2++)
            {
                factor = (1.0-zfac+i1*(2.0*zfac-1.0))
                        *(1.0-gzfac+i2*(2.0*gzfac-1.0));

                if(i2 == 1 and n2>1) j2 = loc_gz+1;
                else j2 = loc_gz;

                loc = j0*n1*n2+j1*n2+j2;
                result = result + factor*table3d[loc];
            }
        }
    }

    return result;
}

double Foam::tableSolver::interp2d
(
    int n1, int n2, int loc_h, int loc_z,
    double hfac,double zfac, double table2d[]
)
{
        int i1,i2,j1,j2,loc;
        double factor,result;

        result = 0.0;
        for(i1=0; i1<2; i1++)
        {
            if(i1 == 1 and n1>1) j1 = loc_h+1;
            else j1 = loc_h;

            for(i2=0; i2<2; i2++)
            {
                factor = (1.0-hfac+i1*(2.0*hfac-1.0))
                        *(1.0-zfac+i2*(2.0*zfac-1.0));

                if(i2 == 1 and n2>1) j2 = loc_z+1;
                else j2 = loc_z;

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


double Foam::tableSolver::lookup3d
(
    int n1, double list_1[], double x1,
    int n2, double list_2[], double x2,
    int n3, double list_3[], double x3,
    double table_3d[]    
)
{
    int loc_1 = locate_lower(n1,list_1,x1);
    double fac_1 = intfac(x1,list_1[loc_1],list_1[loc_1+1]);
    int loc_2 = locate_lower(n2,list_2,x2);
    double fac_2 = intfac(x2,list_2[loc_2],list_2[loc_2+1]);
    int loc_3 = locate_lower(n3,list_3,x3);
    double fac_3 = intfac(x3,list_3[loc_3],list_3[loc_3+1]);

    return interp3d(n1,n2,n3,loc_1,loc_2,loc_3,fac_1,fac_2,fac_3,table_3d);
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


double Foam::tableSolver::lookup6d
(
    int n1, double list_1[], double x1,
    int n2, double list_2[], double x2,
    int n3, double list_3[], double x3,
    int n4, double list_4[], double x4,
    int n5, double list_5[], double x5,
    int n6, double list_6[], double x6,
    double table_6d[]
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
    int loc_6 = locate_lower(n6,list_6,x6);
    double fac_6 = intfac(x6,list_6[loc_6],list_6[loc_6+1]);

    return interp6d(n1,n2,n3,n4,n5,n6,loc_1,loc_2,loc_3,loc_4,loc_5,loc_6,
        fac_1,fac_2,fac_3,fac_4,fac_5,fac_6,table_6d);
}

double Foam::tableSolver::RANSsdrFLRmodel
(
    double cvar, double epsilon, double k, double nu,
    double sl, double dl, double tau, double kc_s,double rho
)
{
        double beta=6.7, C3, C4, Kc, Ka;

        Ka=(dl/(sl+SMALL))/(Foam::sqrt(nu/epsilon)+SMALL);
        Kc=kc_s*tau;
        C3=1.5*Foam::sqrt(Ka)/(1+Foam::sqrt(Ka));
        C4=1.1/Foam::pow((1+Ka),0.4);
        
        return rho/beta*cvar*( (2.0*Kc-tau*C4)*sl/(dl+SMALL) + C3*epsilon/(k+SMALL) );    
}



} // End Foam namespace
