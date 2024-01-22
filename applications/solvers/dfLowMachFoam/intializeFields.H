Info<< "Constructing fields for refined mesh\n" << endl;
Info<< "Reading data from file\n" << endl;
const string filePath = CanteraTorchProperties.subDict("RefineSettings").lookupOrDefault("filePath", string(""));
const label girdSize = CanteraTorchProperties.subDict("RefineSettings").lookupOrDefault("girdSize", 0);

label row = Y.size() + 2;
label col = girdSize;
double* filedata = new double[row * col];

int mpirank, mpisize;
MPI_Initialized(&flag_mpi_init);

if(flag_mpi_init){
    MPI_Comm_rank(PstreamGlobals::MPI_COMM_FOAM, &mpirank);
    MPI_Comm_size(PstreamGlobals::MPI_COMM_FOAM, &mpisize);
}

if(mpirank == 0){
    std::ifstream infile;
    std::string str;
    infile.open(filePath);
    for(label r = 0; r < row; ++r){
        std::getline(infile, str);
        std::stringstream ss(str);
        string tmp;
        for(label j = 0; j < col; ++j)
        {
            ss >> tmp;
            filedata[r * col + j] = std::stod(tmp); 
        }
    }
}

if(flag_mpi_init){
    MPI_Bcast(filedata, row * col, MPI_DOUBLE, 0, PstreamGlobals::MPI_COMM_FOAM);
}

const double* filedata_x = filedata;
const double* filedata_T = filedata + col;
const double* filedata_Y = filedata + 2 * col;

scalar x, y, z;
label index;
forAll(mesh.C(), celli)
{
    x = mesh.C()[celli][0];
    y = mesh.C()[celli][1];
    z = mesh.C()[celli][2];

    // set scalar fields
    index = round((x/0.001) / filedata_x[1]);
    
    T[celli] = filedata_T[index]; 
}
T.correctBoundaryConditions();

forAll(mesh.C(), celli)
{
    // vector c = mesh.C()[celli];
    x = mesh.C()[celli][0];
    y = mesh.C()[celli][1];
    z = mesh.C()[celli][2];

    // set filed U
    U[celli] = vector(4*std::sin(x/0.001)*std::cos(y/0.001)*std::cos(z/0.001),-4*std::cos(x/0.001)*std::sin(y/0.001)*std::cos(z/0.001),0.0); 
}
U.correctBoundaryConditions();

forAll(mesh.C(), celli)
{
    x = mesh.C()[celli][0];
    y = mesh.C()[celli][1];
    z = mesh.C()[celli][2];

    index = round((x/0.001) / filedata_x[1]);

    for (int speciI = 0; speciI < Y.size(); speciI++)
    {
        Y[speciI][celli] = filedata_Y[speciI * col + index];
    }
}
for (int speciI = 0; speciI < Y.size(); speciI++)
{
    Y[speciI].correctBoundaryConditions();
}

chemistry->correctEnthalpy();
chemistry->correctThermo();
rho = p * psi;
rho.correctBoundaryConditions();

initialized = true;