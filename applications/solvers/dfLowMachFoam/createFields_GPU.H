/*******************************************************************************************
 * This file is used to add OpenCC calls to OpenFoam, 
 * initialize OpenCC scopes and request GPU space.
 * 
 * @author Lynn Dang
 ******************************************************************************************/

volScalarField QdotGPU
(
    IOobject
    (
        "QdotGPU",
        mesh.time().timeName(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar(dimEnergy/dimVolume/dimTime, 0)
);

PtrList<volScalarField::Internal> RRGPU(Y.size());
forAll(RRGPU, fieldi)
{
    RRGPU.set
    (
        fieldi,
        new volScalarField::Internal
        (
            IOobject
            (
                "RRGPU." + Y[fieldi].name(),
                mesh.time().timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            mesh,
            dimensionedScalar(dimMass/dimVolume/dimTime, 0)
        )
    );
}

int num_cells = T.size();
int num_species = Y.size();

double* h_T   = new double[num_cells];
double* h_p   = new double[num_cells];
double* h_y   = new double[num_cells * num_species];
double* h_y_t = new double[num_cells * num_species];
int* h_size = new int[1];

memcpy(h_T, &T[0], num_cells * sizeof(double));
memcpy(h_p, &p[0], num_cells * sizeof(double));

forAll(Y, speciesI) {
    volScalarField& Yi = Y[speciesI];
    memcpy(h_y + speciesI * num_cells, &Yi[0], num_cells * sizeof(double));
}

h_size[0] = num_cells;

int sp_num = num_species;

string mechanismFile = CanteraTorchProperties.lookupOrDefault("CanteraMechanismFile", string(""));
char target_mechanismFile[100]; 
std::strcpy(target_mechanismFile, mechanismFile.c_str()); 

opencc_ode_init(target_mechanismFile, num_cells, h_T, h_p, h_y);

double* Ynew = new double[num_cells * num_species];
