/*******************************************************************************************
 * This file is used to add OpenCC calls to OpenFoam, 
 * initialize OpenCC scopes and request GPU space.
 * 
 * @author Lynn Dang
 ******************************************************************************************/

printf("createFields_GPU in\n");

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

std::unique_ptr<double[]> unique_y(new double[num_cells * num_species]); double* h_y = unique_y.get();
std::unique_ptr<double[]> unique_y_t(new double[num_cells * num_species]); double* h_y_t = unique_y_t.get();

forAll(Y, speciesI) {
    volScalarField& Yi = Y[speciesI];
    memcpy(h_y + speciesI * num_cells, &Yi[0], num_cells * sizeof(double));
}

int sp_num = num_species;

string mechanismFilet = CanteraTorchProperties.lookupOrDefault("CanteraMechanismFile", string(""));
char target_mechanismFile[100];
std::strcpy(target_mechanismFile, mechanismFilet.c_str());

int device_ID = get_device_ID();

opencc_ode_init(target_mechanismFile, num_cells, 2, &T[0], &p[0], h_y, PRESSURE, device_ID);

double* Ynew = new double[num_cells * num_species];
