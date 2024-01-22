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

Application
    engineSwirl

Description
    Generates a swirling flow for engine calculations.

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "mathematicalConstants.H"
#include <math.h>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <mpi.h>


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    #include "setRootCase.H"
    
    int mpirank = 0;
    int mpisize = 0;
    int flag_mpi_init;
    MPI_Initialized(&flag_mpi_init);

    if(flag_mpi_init){
        MPI_Comm_rank(MPI_COMM_WORLD,&mpirank);
        MPI_Comm_size(MPI_COMM_WORLD,&mpisize);
    }

    double total_start = MPI_Wtime();

    #include "createTime.H"
    double time_1 = MPI_Wtime();
    #include "createMesh.H"
    double time_2 = MPI_Wtime();
    #include "createFields.H"
    double time_3 = MPI_Wtime();
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    // read .txt
    std::vector<double *> filedata;
    std::ifstream infile;
    std::string str;
    std::string adiabatic_file_path;
    int grid_size = 0;

    Info << "adiabatic file path : " << adiabatic_file_path << endl;
    Info << "grid size : " << grid_size << endl;

    std::vector<int> gird_size_list({256,512,1024,2048});
    for(size_t i = 0; i < gird_size_list.size(); ++i){
        std::stringstream ss;
        grid_size = gird_size_list[i];
        ss << "./adiabatic_" << grid_size << ".txt";
        adiabatic_file_path = ss.str();
        Info << "Try" << endl;
        Info << "adiabatic file path : " << adiabatic_file_path << endl;
        Info << "grid size : " << grid_size << endl;
        infile.open(adiabatic_file_path);
        if(infile.good()){
            Info << "Success" << endl;
            break;
        }
    }
    double time_4 = MPI_Wtime();

    // infile.open(adiabatic_file_path);
    while (std::getline(infile, str))
    {
        double * data = new double[grid_size]{0.0};
        std::stringstream sumstr(str);
        string out;
        int i = 0;
        while (sumstr >> out)
        {
            double a = std::stod(out); 
            data[i] = a;
            i++;
        }
        filedata.push_back(data);
    }
    double time_5 = MPI_Wtime();

    Info << "vector size" << filedata.size() << endl;
    Info << "array" << filedata[0][1] << endl;

    scalar x, y, z;
    label index;
    forAll(mesh.C(), celli)
    {
        // vector c = mesh.C()[celli];
        x = mesh.C()[celli][0];
        y = mesh.C()[celli][1];
        z = mesh.C()[celli][2];

        // set filed U
        U[celli] = vector(4*std::sin(x/0.001)*std::cos(y/0.001)*std::cos(z/0.001),-4*std::cos(x/0.001)*std::sin(y/0.001)*std::cos(z/0.001),0.0);

        // set scalar fields
        index = round((x/0.001) / filedata[0][1]);
        T[celli] = filedata[1][index];
        H2[celli] = filedata[2][index];
        H[celli] = filedata[3][index];
        O2[celli] = filedata[4][index];
        OH[celli] = filedata[5][index];
        O[celli] = filedata[6][index];
        H2O[celli] = filedata[7][index];
        HO2[celli] = filedata[8][index];
        H2O2[celli] = filedata[9][index];
        N2[celli] = filedata[10][index];
        
    }
    double time_6 = MPI_Wtime();

    U.correctBoundaryConditions();
    T.correctBoundaryConditions();
    H2.correctBoundaryConditions();
    H.correctBoundaryConditions();
    O2.correctBoundaryConditions();
    OH.correctBoundaryConditions();
    O.correctBoundaryConditions();
    H2O.correctBoundaryConditions();
    HO2.correctBoundaryConditions();
    H2O2.correctBoundaryConditions();
    N2.correctBoundaryConditions();
    double time_7 = MPI_Wtime();

    U.write();
    T.write();
    H2.write();
    H.write();
    O2.write();
    OH.write();
    O.write();
    H2O.write();
    HO2.write();
    H2O2.write();
    N2.write();

    Info << "\n end\n";
    double total_end = MPI_Wtime();

    Info << "total_time : " << total_end - total_start << endl;
    Info << "createTime : " << time_1 - total_start << endl;
    Info << "createMesh : " << time_2 - time_1 << endl;
    Info << "createFields : " << time_3 - time_2 << endl;
    Info << "open file : " << time_4 - time_3 << endl;
    Info << "read file : " << time_5 - time_4 << endl;
    Info << "init field : " << time_6 - time_5 << endl;
    Info << "correctBoundaryConditions : " << time_7 - time_6 << endl;
    Info << "write : " << total_end - time_7 << endl;

    return 0;
}


// ************************************************************************* //
