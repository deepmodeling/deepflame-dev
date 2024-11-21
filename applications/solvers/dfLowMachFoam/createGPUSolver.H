#include <queue>
#include "CSRGAMGAgglomeration.H"
#include "ELLGAMGAgglomeration.H"
#include "mpi.h"

#define checkMpiErrors(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define PARALLEL_
#define iscsr

int myRank = 0;

int **h_ldu_to_csr_no_diag = nullptr;
int **h_csr_row_index_no_diag= nullptr;
int **h_csr_col_index_no_diag= nullptr;

int *h_ell_row_maxcount = nullptr;
int **h_ellCols = nullptr;
int **h_ldu2ellIndex = nullptr;

/* For DeepFlame_Academic */
void createGPUSolverInput(const lduMesh& mesh, const dictionary solverControls, MATRIX_EQUATION EqnID, string AMGXsetting="");
void createGPUGAMGInput(const lduMesh& mesh, volScalarField& eqn);
void setConstantIndexes(mesh_info_para &mesh_paras, const int *owner, const int *neighbor, const int *procRows, 
        const int *procCols, int globalOffset, int num_proc_surfaces);
void setConstantIndexes_GAMG(int nCells, int nFaces, int *owner, int *neighbor, int leveli, int type);


#include <fstream>
#include <iostream>
#include <iomanip> 
void writeDoubleArrayToFile(const double* data, int num, const std::string& filename, bool compareCPUResults = false) {
    if(!compareCPUResults) return;

    int mpi_init_flag;
    checkMpiErrors(MPI_Initialized(&mpi_init_flag));
    if(mpi_init_flag) {
        checkMpiErrors(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
    }

    if(myRank!=0) return;

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
    }

    file << std::scientific << std::setprecision(15);

    for (int i = 0; i < num; ++i) {
        file << data[i];
        if (i != num - 1) {
            file << "\n"; 
        }
    }

    if (!file) {
        std::cerr << "Failed to write to file: " << filename << std::endl;
        file.close();
    }

    file.close();
    std::cout << "--- DEBUG: Write " << filename << " successfully. " << std::endl;
}

void permuteVector2Host(double *&new_order, double *old_order, int num) 
{
    for (int i = 0; i < num; i++) 
    {
        new_order[i * 3 + 0] = old_order[num * 0 + i];
        new_order[i * 3 + 1] = old_order[num * 1 + i];
        new_order[i * 3 + 2] = old_order[num * 2 + i];
    }
}


void permuteVector(double *&new_order, const double *old_order, int num) 
{
    for (int i = 0; i < num; i++) 
    {
        new_order[num * 0 + i] = old_order[i * 3 + 0];
        new_order[num * 1 + i] = old_order[i * 3 + 1];
        new_order[num * 2 + i] = old_order[i * 3 + 2];
    }
}

void permuteTensor(double *&new_order, const double *old_order, int num) 
{
    for (int i = 0; i < num; i++) 
    {
        new_order[num * 0 + i] = old_order[i * 9 + 0];
        new_order[num * 1 + i] = old_order[i * 9 + 1];
        new_order[num * 2 + i] = old_order[i * 9 + 2];
        new_order[num * 3 + i] = old_order[i * 9 + 3];
        new_order[num * 4 + i] = old_order[i * 9 + 4];
        new_order[num * 5 + i] = old_order[i * 9 + 5];
        new_order[num * 6 + i] = old_order[i * 9 + 6];
        new_order[num * 7 + i] = old_order[i * 9 + 7];
        new_order[num * 8 + i] = old_order[i * 9 + 8];
    }
}

void constructBoundarySelectorPerPatch(int *patchTypeSelector, const std::string& patchTypeStr)
{
    boundaryConditions patchCondition;
    std::vector<int> tmpSelector;
    static std::map<std::string, boundaryConditions> BCMap = {
        {"zeroGradient", zeroGradient},
        {"fixedValue", fixedValue},
        {"empty", empty},
        {"gradientEnergy", gradientEnergy},
        {"calculated", calculated},
        {"coupled", coupled},
        {"cyclic", cyclic},
        {"processor", processor},
        {"extrapolated", extrapolated},
        {"fixedEnergy", fixedEnergy},
        {"processorCyclic", processorCyclic},
        {"pressureInletOutletVelocity", pressureInletOutletVelocity},
        {"totalPressure", totalPressure},
        {"wedge", wedge},
        {"noSlip", noSlip},
        {"mapped", mapped},

        {"turbulentMixingLengthDissipationRateInlet", turbulentMixingLengthDissipationRateInlet},
        {"turbulentIntensityKineticEnergyInlet", turbulentIntensityKineticEnergyInlet},
        {"nutkWallFunction", nutkWallFunction},
        {"epsilonWallFunction", epsilonWallFunction},
        {"kqRWallFunction", kqRWallFunction},
        {"compressible::alphatWallFunction", alphatWallFunction},
    };
    auto iter = BCMap.find(patchTypeStr);
    if (iter != BCMap.end()) {
        patchCondition = iter->second;
    } else {
        throw std::runtime_error("Unknown boundary condition: " + patchTypeStr);
    }
    // zeroGradient labeled as 0, fixedValue labeled as 1, coupled labeled as 2
    switch (patchCondition){
        case zeroGradient:
        {
            *patchTypeSelector = 0;
            break;
        }
        case fixedValue:
        {
            *patchTypeSelector = 1;
            break;
        }
        case coupled:
        {
            *patchTypeSelector = 2;
            break;
        }
        case empty:
        {
            *patchTypeSelector = 3;
            break;
        }
        case gradientEnergy:
        {
            *patchTypeSelector = 4;
            break;
        }
        case calculated:
        {
            *patchTypeSelector = 5;
            break;
        }
        case cyclic:
        {
            *patchTypeSelector = 6;
            break;
        }
        case processor:
        {
            *patchTypeSelector = 7;
            break;
        }
        case extrapolated:
        {
            *patchTypeSelector = 8;
            break;
        }
        case fixedEnergy:
        {
            *patchTypeSelector = 9;
            break;
        }
        case processorCyclic:
        {
            *patchTypeSelector = 10;
            break;
        }
        case pressureInletOutletVelocity:
        {
            *patchTypeSelector = 11;
            break;
        }
        case totalPressure:
        {
            *patchTypeSelector = 12;
            break;
        }
        case wedge:
        {
            *patchTypeSelector = 13;
            break;
        }
        case noSlip:
        {
            *patchTypeSelector = 14;
            break;
        }
        case turbulentMixingLengthDissipationRateInlet:
        {
            *patchTypeSelector = 15;
            break;
        }
        case turbulentIntensityKineticEnergyInlet:
        {
            *patchTypeSelector = 16;
            break;
        }
        case nutkWallFunction:
        {
            *patchTypeSelector = 17;
            break;
        }
        case epsilonWallFunction:
        {
            *patchTypeSelector = 18;
            break;
        }
        case kqRWallFunction:
        {
            *patchTypeSelector = 19;
            break;
        }
        case alphatWallFunction:
        {
            *patchTypeSelector = 20;
            break;
        }
        case mapped:
        {
            *patchTypeSelector = 21;
            break;
        }
    }
}

void createGPURhoEqn(mesh_info_para &mesh_paras, init_data_para &init_data, const volScalarField& rho, const surfaceScalarField& phi) 
{
    DEBUG_TRACE;
    std::vector<int> patch_type;
    patch_type.resize(mesh_paras.num_patches);

    double *h_boundary_phi = new double[mesh_paras.num_boundary_surfaces]();
    double *h_boundary_rho = new double[mesh_paras.num_boundary_surfaces]();

    int offset = 0;
    forAll(rho.boundaryField(), patchi)
    {
        constructBoundarySelectorPerPatch(&(patch_type[patchi]), rho.boundaryField()[patchi].type());
        const fvPatchScalarField& patchRho = rho.boundaryField()[patchi];
        const fvsPatchScalarField& patchPhi = phi.boundaryField()[patchi];
        int patchsize = patchRho.size();
        if (patchRho.type() == "processor"
            || patchRho.type() == "processorCyclic") {
            if (dynamic_cast<const processorFvPatchField<scalar>&>(patchRho).doTransform()) {
                Info << "gradU transform = true" << endl;
            } else {
                Info << "gradU transform = false" << endl;
            }
            Info << "rank = " << dynamic_cast<const processorFvPatchField<scalar>&>(patchRho).rank() << endl;

            memcpy(h_boundary_rho + offset, &patchRho[0], patchsize * sizeof(double));
            scalarField patchRhoInternal = 
                    dynamic_cast<const processorFvPatchField<scalar>&>(patchRho).patchInternalField()();
            memcpy(h_boundary_rho + offset + patchsize, &patchRhoInternal[0], patchsize * sizeof(double));

            memcpy(h_boundary_phi + offset, &patchPhi[0], patchsize * sizeof(double));
            memset(h_boundary_phi + offset + patchsize, 0, patchsize * sizeof(double));

            offset += patchsize * 2;
        } else {
            memcpy(h_boundary_rho + offset, &patchRho[0], patchsize * sizeof(double));
            memcpy(h_boundary_phi + offset, &patchPhi[0], patchsize * sizeof(double));
            offset += patchsize;
        }
    }

    init_data.rho = &rho[0];
    init_data.phi = &phi[0];

    init_data.boundary_rho  = new double[mesh_paras.num_boundary_surfaces];
    init_data.boundary_phi  = new double[mesh_paras.num_boundary_surfaces];

    memcpy(init_data.boundary_rho, &h_boundary_rho[0], mesh_paras.num_boundary_surfaces * sizeof(double));
    memcpy(init_data.boundary_phi, &h_boundary_phi[0], mesh_paras.num_boundary_surfaces * sizeof(double));

    init_data.patch_type_rho   = new int[mesh_paras.num_patches]();
    std::copy( patch_type.begin(), patch_type.end(), init_data.patch_type_rho ); 
}

void createGPUUEqn(mesh_info_para &mesh_paras, init_data_para &init_data, const IOdictionary& CanteraTorchProperties, const volVectorField& U) {
    // TODO need remove amgx solver setting
    // prepare mode_string and setting_path
    string mode_string = "dDDI";
    string settingPath;
    settingPath = CanteraTorchProperties.subDict("AmgxSettings").lookupOrDefault("UEqnSettingPath", string(""));
    // UEqn_GPU.setConstantValues(mode_string, settingPath);

    // prepare patch_type
    std::vector<int> patch_type;
    patch_type.resize(mesh_paras.num_patches);
    double *h_boundary_u = new double[mesh_paras.num_boundary_surfaces*3]();
    
    int offset = 0;
    forAll(U.boundaryField(), patchi)
    {
        constructBoundarySelectorPerPatch(&(patch_type[patchi]), U.boundaryField()[patchi].type());
        const fvPatchVectorField& patchU = U.boundaryField()[patchi];
        int patchsize = patchU.size();
        if (patchU.type() == "processor"
            || patchU.type() == "processorCyclic") {
            if (dynamic_cast<const processorFvPatchField<vector>&>(patchU).doTransform()) {
                Info << "U transform = true" << endl;
            } else {
                Info << "U transform = false" << endl;
            }
            Info << "rank = " << dynamic_cast<const processorFvPatchField<vector>&>(patchU).rank() << endl;

            memcpy(h_boundary_u + 3*offset, &patchU[0][0], patchsize * 3 * sizeof(double));
            vectorField patchUInternal = 
                    dynamic_cast<const processorFvPatchField<vector>&>(patchU).patchInternalField()();
            memcpy(h_boundary_u + 3*offset + 3*patchsize, &patchUInternal[0][0], patchsize * 3 * sizeof(double));
            offset += patchsize * 2;
        } else {
            memcpy(h_boundary_u + 3*offset, &patchU[0][0], patchsize * 3 * sizeof(double));
            offset += patchsize;
        }
    }

    double *u_newOrder = new double[mesh_paras.num_cells*3]();
    double *boundary_u_newOrder = new double[mesh_paras.num_boundary_surfaces*3]();
    permuteVector(u_newOrder, &U[0][0], mesh_paras.num_cells);
    permuteVector(boundary_u_newOrder, h_boundary_u, mesh_paras.num_boundary_surfaces);

    init_data.u           = new double[mesh_paras.num_cells*3];
    init_data.boundary_u  = new double[mesh_paras.num_boundary_surfaces*3];

    memcpy(init_data.u,          &u_newOrder[0],          mesh_paras.num_cells * 3 * sizeof(double));
    memcpy(init_data.boundary_u, &boundary_u_newOrder[0], mesh_paras.num_boundary_surfaces * 3 * sizeof(double));

    init_data.patch_type_U   = new int[mesh_paras.num_patches]();
    std::copy( patch_type.begin(), patch_type.end(), init_data.patch_type_U );   
}

void createGPUYEqn(mesh_info_para &mesh_paras, init_data_para &init_data, const IOdictionary& CanteraTorchProperties, PtrList<volScalarField>& Y, const int inertIndex) {
    DEBUG_TRACE;
    // TODO need remove amgx solver setting
    // prepare mode_string and setting_path
    string mode_string = "dDDI";
    string settingPath;
    settingPath = CanteraTorchProperties.subDict("AmgxSettings").lookupOrDefault("UEqnSettingPath", string(""));
    // YEqn_GPU.setConstantValues(mode_string, settingPath, inertIndex);

    // prepare patch_type
    std::vector<int> patch_type;
    patch_type.resize(mesh_paras.num_patches);

    forAll(Y[0].boundaryField(), patchi)
    {
        constructBoundarySelectorPerPatch(&(patch_type[patchi]), Y[0].boundaryField()[patchi].type());
    }
    // set lewis number
    std::vector<double> lewis_number(init_data.num_species, 1.); // unity lewis
    double *h_y = new double[init_data.num_species*mesh_paras.num_cells]();
    double *h_boundary_y = new double[init_data.num_species*mesh_paras.num_boundary_surfaces]();

    // prepare internal and boundary of Y
    int offset = 0;
    forAll(Y, speciesI) {
        volScalarField& Yi = Y[speciesI];
        memcpy(h_y + speciesI * mesh_paras.num_cells, &Yi[0], mesh_paras.num_cells * sizeof(double));
        forAll(Yi.boundaryField(), patchi) {
            const fvPatchScalarField& patchYi = Yi.boundaryField()[patchi];
            int patchsize = patchYi.size();
            if (patchYi.type() == "processor"
                || patchYi.type() == "processorCyclic") {
                scalarField patchYiInternal =
                    dynamic_cast<const processorFvPatchField<scalar>&>(patchYi).patchInternalField()();
                memcpy(h_boundary_y + offset, &patchYi[0], patchsize * sizeof(double));
                memcpy(h_boundary_y + offset + patchsize, &patchYiInternal[0], patchsize * sizeof(double));
                offset += patchsize * 2;
            } else {
                memcpy(h_boundary_y + offset, &patchYi[0], patchsize*sizeof(double));
                offset += patchsize;
            }
        }
    }

    init_data.inertIndex  = inertIndex;
    init_data.y           = new double[mesh_paras.num_cells*init_data.num_species];
    init_data.boundary_y  = new double[mesh_paras.num_boundary_surfaces*init_data.num_species];

    memcpy(init_data.y,          &h_y[0],          mesh_paras.num_cells*init_data.num_species * sizeof(double));
    memcpy(init_data.boundary_y, &h_boundary_y[0], mesh_paras.num_boundary_surfaces*init_data.num_species * sizeof(double));

    init_data.patch_type_Y   = new int[mesh_paras.num_patches]();
    std::copy( patch_type.begin(), patch_type.end(), init_data.patch_type_Y ); 
}

void createGPUEEqn(mesh_info_para &mesh_paras, init_data_para &init_data, const IOdictionary& CanteraTorchProperties, volScalarField& he, volScalarField& K) {
    DEBUG_TRACE;
    // TODO need remove amgx solver setting
    // prepare mode_string and setting_path
    string mode_string = "dDDI";
    string settingPath;
    settingPath = CanteraTorchProperties.subDict("AmgxSettings").lookupOrDefault("UEqnSettingPath", string(""));
    // EEqn_GPU.setConstantValues(mode_string, settingPath);

    // prepare patch_type
    std::vector<int> patch_type_he(mesh_paras.num_patches), patch_type_k(mesh_paras.num_patches);
    forAll(he.boundaryField(), patchi)
    {
        constructBoundarySelectorPerPatch(&(patch_type_he[patchi]), he.boundaryField()[patchi].type());
        constructBoundarySelectorPerPatch(&(patch_type_k[patchi]), K.boundaryField()[patchi].type());
    }

    double h_boundary_he[mesh_paras.num_boundary_surfaces];

    int offset = 0;
    forAll(he.boundaryField(), patchi)
    {
        const fvPatchScalarField& patchHe = he.boundaryField()[patchi];
        int patchsize = patchHe.size();
        if (patchHe.type() == "processor"
            || patchHe.type() == "processorCyclic") {
            scalarField patchHeInternal =
                    dynamic_cast<const processorFvPatchField<scalar>&>(patchHe).patchInternalField()();
            memcpy(h_boundary_he + offset, &patchHe[0], patchsize * sizeof(double));
            memcpy(h_boundary_he + offset + patchsize, &patchHeInternal[0], patchsize * sizeof(double));
            offset += patchsize * 2;
        } else {
            memcpy(h_boundary_he + offset, &patchHe[0], patchsize * sizeof(double));
            offset += patchsize;
        }
    }

    init_data.he = &he[0];

    init_data.boundary_he  = new double[mesh_paras.num_boundary_surfaces];

    memcpy(init_data.boundary_he, &h_boundary_he[0], mesh_paras.num_boundary_surfaces * sizeof(double));

    init_data.patch_type_he   = new int[mesh_paras.num_patches]();
    std::copy( patch_type_he.begin(), patch_type_he.end(), init_data.patch_type_he ); 

    init_data.patch_type_k   = new int[mesh_paras.num_patches]();
    std::copy( patch_type_k.begin(), patch_type_k.end(), init_data.patch_type_k ); 
}

void createGPUpEqn(mesh_info_para &mesh_paras, init_data_para &init_data, const IOdictionary& CanteraTorchProperties, volScalarField& p, const volVectorField& U) {
    DEBUG_TRACE;
    // TODO need remove amgx solver setting
    // prepare mode_string and setting_path
    string mode_string = "dDDI";
    string settingPath;
    settingPath = CanteraTorchProperties.subDict("AmgxSettings").lookupOrDefault("pEqnSettingPath", string(""));
    // pEqn_GPU.setConstantValues(mode_string, settingPath);
    
    // prepare patch_type
    std::vector<int> patch_type_p(mesh_paras.num_patches);
    std::vector<int> patch_type_U(mesh_paras.num_patches);

    double h_boundary_p[mesh_paras.num_boundary_surfaces];

    int offset = 0;
    forAll(p.boundaryField(), patchi)
    {
        constructBoundarySelectorPerPatch(&(patch_type_p[patchi]), p.boundaryField()[patchi].type());
        constructBoundarySelectorPerPatch(&(patch_type_U[patchi]), U.boundaryField()[patchi].type());
        const fvPatchScalarField& patchP = p.boundaryField()[patchi];
        int patchsize = patchP.size();
        if (patchP.type() == "processor"
            || patchP.type() == "processorCyclic") {
            memcpy(h_boundary_p + offset, &patchP[0], patchsize * sizeof(double));
            scalarField patchPInternal = 
                    dynamic_cast<const processorFvPatchField<scalar>&>(patchP).patchInternalField()();
            memcpy(h_boundary_p + offset + patchsize, &patchPInternal[0], patchsize * sizeof(double));

            offset += patchsize * 2;
        } else {
            memcpy(h_boundary_p + offset, &patchP[0], patchsize * sizeof(double));
            offset += patchsize;
        }
        if (patchP.type() == "totalPressure"){
            const scalarField& p0 = dynamic_cast<const totalPressureFvPatchScalarField&>(patchP).p0();
            init_data.p0 = p0[0];
            Info << "totalPressure p0: " << p0[0] << endl;
        }
    }

    init_data.p = &p[0];

    init_data.boundary_p  = new double[mesh_paras.num_boundary_surfaces];

    memcpy(init_data.boundary_p, &h_boundary_p[0], mesh_paras.num_boundary_surfaces * sizeof(double));

    init_data.patch_type_p   = new int[mesh_paras.num_patches]();
    std::copy( patch_type_p.begin(), patch_type_p.end(), init_data.patch_type_p ); 
}

void createGPUThermo(mesh_info_para &mesh_paras, init_data_para &init_data, const IOdictionary& CanteraTorchProperties, volScalarField& T, volScalarField& he, 
        const volScalarField& psi, const volScalarField& alpha, const volScalarField& mu,
        const volScalarField& K, const volScalarField& dpdt, dfChemistryModel<basicThermo>* chemistry) {
    DEBUG_TRACE;
    // initialize dfThermo
    string mechanismFile;
    mechanismFile = CanteraTorchProperties.lookupOrDefault("CanteraMechanismFile", string(""));

    // thermo_GPU.setConstantValue(mechanismFile, num_cells, num_species);
    // init_const_coeff_ptr(thermo_GPU.nasa_coeffs, thermo_GPU.viscosity_coeffs, thermo_GPU.thermal_conductivity_coeffs, 
    //         thermo_GPU.binary_diffusion_coeffs, thermo_GPU.molecular_weights);

    // thermal variables in dataBase
    // TODO: note that h_he & h_boundary_he are transfered to GPU in EEqn_GPU, too. We should delete one of them.
    double *h_boundary_T = new double[mesh_paras.num_boundary_surfaces];
    double *h_boundary_he = new double[mesh_paras.num_boundary_surfaces];
    double *h_boundary_thermo_psi = new double[mesh_paras.num_boundary_surfaces];
    double *h_boundary_thermo_alpha = new double[mesh_paras.num_boundary_surfaces];
    double *h_boundary_mu = new double[mesh_paras.num_boundary_surfaces];
    double *h_boundary_k = new double[mesh_paras.num_boundary_surfaces];
    double *h_boundary_thermo_rhoD = new double[mesh_paras.num_boundary_surfaces * init_data.num_species];
    double *h_thermo_rhoD = new double[mesh_paras.num_cells * init_data.num_species];

    // initialize thermo boundary
    std::vector<int> patch_type_T(mesh_paras.num_patches);
    int offset = 0;
    forAll(T.boundaryField(), patchi)
    {
        constructBoundarySelectorPerPatch(&(patch_type_T[patchi]), T.boundaryField()[patchi].type());
        const fvPatchScalarField& patchT = T.boundaryField()[patchi];
        const fvPatchScalarField& patchHe = he.boundaryField()[patchi];
        const fvPatchScalarField& patchPsi = psi.boundaryField()[patchi];
        const fvPatchScalarField& patchAlpha = alpha.boundaryField()[patchi];
        const fvPatchScalarField& patchMu = mu.boundaryField()[patchi];
        const fvPatchScalarField& patchK = K.boundaryField()[patchi];

        int patchsize = patchT.size();
        if (patchT.type() == "processor"
            || patchT.type() == "processorCyclic") {
            memcpy(h_boundary_T + offset, &patchT[0], patchsize * sizeof(double));
            scalarField patchTInternal = 
                    dynamic_cast<const processorFvPatchField<scalar>&>(patchT).patchInternalField()();
            memcpy(h_boundary_T + offset + patchsize, &patchTInternal[0], patchsize * sizeof(double));

            memcpy(h_boundary_he + offset, &patchHe[0], patchsize * sizeof(double));
            scalarField patchHeInternal = 
                    dynamic_cast<const processorFvPatchField<scalar>&>(patchHe).patchInternalField()();
            memcpy(h_boundary_he + offset + patchsize, &patchHeInternal[0], patchsize * sizeof(double));

            memcpy(h_boundary_thermo_psi + offset, &patchPsi[0], patchsize * sizeof(double));
            scalarField patchPsiInternal = 
                    dynamic_cast<const processorFvPatchField<scalar>&>(patchPsi).patchInternalField()();
            memcpy(h_boundary_thermo_psi + offset + patchsize, &patchPsiInternal[0], patchsize * sizeof(double));

            memcpy(h_boundary_thermo_alpha + offset, &patchAlpha[0], patchsize * sizeof(double));
            scalarField patchAlphaInternal = 
                    dynamic_cast<const processorFvPatchField<scalar>&>(patchAlpha).patchInternalField()();
            memcpy(h_boundary_thermo_alpha + offset + patchsize, &patchAlphaInternal[0], patchsize * sizeof(double));

            memcpy(h_boundary_mu + offset, &patchMu[0], patchsize * sizeof(double));
            scalarField patchMuInternal = 
                    dynamic_cast<const processorFvPatchField<scalar>&>(patchMu).patchInternalField()();
            memcpy(h_boundary_mu + offset + patchsize, &patchMuInternal[0], patchsize * sizeof(double));

            memcpy(h_boundary_k + offset, &patchK[0], patchsize * sizeof(double));
            scalarField patchKInternal = 
                    dynamic_cast<const processorFvPatchField<scalar>&>(patchK).patchInternalField()();
            memcpy(h_boundary_k + offset + patchsize, &patchKInternal[0], patchsize * sizeof(double));

            for (int i = 0; i < init_data.num_species; i++) {
                const fvPatchScalarField& patchRhoD = chemistry->rhoD(i).boundaryField()[patchi];
                memcpy(h_boundary_thermo_rhoD + i * mesh_paras.num_boundary_surfaces + offset, &patchRhoD[0], patchsize * sizeof(double));
                scalarField patchRhoDInternal = 
                        dynamic_cast<const processorFvPatchField<scalar>&>(patchRhoD).patchInternalField()();
                memcpy(h_boundary_thermo_rhoD + i * mesh_paras.num_boundary_surfaces + offset + patchsize, &patchRhoDInternal[0], patchsize * sizeof(double));
            }

            offset += patchsize * 2;
        } else {
            memcpy(h_boundary_T + offset, &patchT[0], patchsize * sizeof(double));
            memcpy(h_boundary_he + offset, &patchHe[0], patchsize * sizeof(double));
            memcpy(h_boundary_thermo_psi + offset, &patchPsi[0], patchsize * sizeof(double));
            memcpy(h_boundary_thermo_alpha + offset, &patchAlpha[0], patchsize * sizeof(double));
            memcpy(h_boundary_mu + offset, &patchMu[0], patchsize * sizeof(double));
            memcpy(h_boundary_k + offset, &patchK[0], patchsize * sizeof(double));

            for (int i = 0; i < init_data.num_species; i++) {
                const fvPatchScalarField& patchRhoD = chemistry->rhoD(i).boundaryField()[patchi];
                memcpy(h_boundary_thermo_rhoD + i * mesh_paras.num_boundary_surfaces + offset, &patchRhoD[0], patchsize * sizeof(double));
            }
            offset += patchsize;
        }
    }
    for (int i = 0; i < init_data.num_species; i++) {
        memcpy(h_thermo_rhoD + i * mesh_paras.num_cells, &chemistry->rhoD(i)[0], mesh_paras.num_cells * sizeof(double));
    }

    init_data.T = &T[0];
    init_data.thermo_psi = &psi[0];
    init_data.thermo_alpha = &alpha[0];
    init_data.mu = &mu[0];
    init_data.k = &K[0];
    init_data.dpdt = &dpdt[0];

    init_data.thermo_rhoD  = new double[mesh_paras.num_cells * init_data.num_species];
    memcpy(init_data.thermo_rhoD, &h_thermo_rhoD[0], mesh_paras.num_cells * init_data.num_species * sizeof(double));

    init_data.boundary_T  = new double[mesh_paras.num_boundary_surfaces];
    init_data.boundary_thermo_psi  = new double[mesh_paras.num_boundary_surfaces];
    init_data.boundary_thermo_alpha  = new double[mesh_paras.num_boundary_surfaces];
    init_data.boundary_mu  = new double[mesh_paras.num_boundary_surfaces];
    init_data.boundary_k  = new double[mesh_paras.num_boundary_surfaces];
    init_data.boundary_thermo_rhoD  = new double[mesh_paras.num_boundary_surfaces * init_data.num_species];

    memcpy(init_data.boundary_T, &h_boundary_T[0], mesh_paras.num_boundary_surfaces * sizeof(double));
    memcpy(init_data.boundary_thermo_psi, &h_boundary_thermo_psi[0], mesh_paras.num_boundary_surfaces * sizeof(double));
    memcpy(init_data.boundary_thermo_alpha, &h_boundary_thermo_alpha[0], mesh_paras.num_boundary_surfaces * sizeof(double));
    memcpy(init_data.boundary_mu, &h_boundary_mu[0], mesh_paras.num_boundary_surfaces * sizeof(double));
    memcpy(init_data.boundary_k, &h_boundary_k[0], mesh_paras.num_boundary_surfaces * sizeof(double));
    memcpy(init_data.boundary_thermo_rhoD, &h_boundary_thermo_rhoD[0], mesh_paras.num_boundary_surfaces * init_data.num_species * sizeof(double));

    init_data.patch_type_T   = new int[mesh_paras.num_patches]();
    std::copy( patch_type_T.begin(), patch_type_T.end(), init_data.patch_type_T ); 

    delete h_boundary_T;
    delete h_boundary_he;
    delete h_boundary_thermo_psi;
    delete h_boundary_thermo_alpha;
    delete h_boundary_mu;
    delete h_boundary_k;
    delete h_boundary_thermo_rhoD;
    delete h_thermo_rhoD;
}

void createGPUSolver(const dictionary solversDict, const IOdictionary& CanteraTorchProperties, 
                     const volScalarField& rho, const volVectorField& U, volScalarField& p, 
                     PtrList<volScalarField>& Y, volScalarField& he, bool setRAS)
{
    dictionary rhoSolverDict = solversDict.subDict("rho"); 
    dictionary USolverDict = solversDict.subDict("U"); 
    dictionary pSolverDict = solversDict.subDict("p"); 
    dictionary YSolverDict = solversDict.subDict("Yi"); 
    dictionary ESolverDict = solversDict.subDict("ha"); 

    string UEqnSettingPath = "";
    string pEqnSettingPath = "";

    bool useAMGX = true;
    if (useAMGX)
    {
        UEqnSettingPath = CanteraTorchProperties.subDict("AmgxSettings").lookupOrDefault("UEqnSettingPath", string(""));
        pEqnSettingPath = CanteraTorchProperties.subDict("AmgxSettings").lookupOrDefault("pEqnSettingPath", string(""));
    }

    /* For DeepFlame_Academic */
    createGPUSolverInput(rho.mesh(), rhoSolverDict, MATRIX_EQUATION::rhoEqn); 
    initialize_matrix(MATRIX_EQUATION::rhoEqn);

    createGPUSolverInput(U.mesh(), USolverDict, MATRIX_EQUATION::UEqn, UEqnSettingPath); 
    initialize_matrix(MATRIX_EQUATION::UEqn);  

    createGPUSolverInput(Y[0].mesh(), YSolverDict, MATRIX_EQUATION::YEqn, UEqnSettingPath); 
    initialize_matrix(MATRIX_EQUATION::YEqn);

    createGPUSolverInput(he.mesh(), ESolverDict, MATRIX_EQUATION::EEqn, UEqnSettingPath); 
    initialize_matrix(MATRIX_EQUATION::EEqn);

    createGPUSolverInput(p.mesh(), pSolverDict, MATRIX_EQUATION::pEqn, pEqnSettingPath); 
    initialize_matrix(MATRIX_EQUATION::pEqn);  

    if (setRAS)
    {
        createGPUSolverInput(U.mesh(), USolverDict, MATRIX_EQUATION::kEqn, UEqnSettingPath); 
        initialize_matrix(MATRIX_EQUATION::kEqn);   

        createGPUSolverInput(U.mesh(), USolverDict, MATRIX_EQUATION::epsilonEqn, UEqnSettingPath); 
        initialize_matrix(MATRIX_EQUATION::epsilonEqn);          
    }
}

void createChemistrySolver(int batch_size, double unReactT){
    
    /* For DeepFlame_Academic */
#ifdef OPENCC
    set_chemistry_solver(OpenCC);
#else
    set_chemistry_solver(DNN);
#endif
    initialize_chemistry_solver(batch_size, unReactT);

}

string getFullSchemesName(const dictionary schemesDict, string foundSchemes)
{
    ITstream& its(schemesDict.lookup(foundSchemes));
    token t;
    its.putBack(t); // Put back the first token to read it again
    its >> t;
    string fullScheme = t.wordToken();

    // Read the rest of the tokens
    while (its.good())
    {
        its >> t;
        if (t.good())
        {
            fullScheme += t.wordToken();
        }
    }
    return fullScheme;
}

int getIntSchemes(string schemeName)
{
    if (schemeName == "Gausslinear")           return 0;
    if (schemeName == "Gaussupwind")           return 1;
    if (schemeName == "GausslimitedLinear")    return 2;
    if (schemeName == "GausslimitedLinear01")  return 3;
    if (schemeName == "GausslimitedLinearV")   return 4;
    return -1;
}

void createGPUSchemesInput(const dictionary schemesControls, fvSchemes_para schemes_para)
{
    std::cout << "- Set operator schemes for DeepFlame_Academic " << std::endl;    
    
    dictionary ddtSchemesDict = schemesControls.subDict("ddtSchemes"); 
    dictionary gradSchemesDict = schemesControls.subDict("gradSchemes"); 
    dictionary divSchemesDict = schemesControls.subDict("divSchemes"); 
    dictionary laplacianSchemesDict = schemesControls.subDict("laplacianSchemes"); 
    dictionary interpolationSchemesDict = schemesControls.subDict("interpolationSchemes"); 
    dictionary snGradSchemesDict = schemesControls.subDict("snGradSchemes"); 

    string divSchemes = getFullSchemesName(divSchemesDict, "default");
    std::cout << "  = Schemes setting [divSchemes default]: " << divSchemes << std::endl;
    if (divSchemes == "none")
    {
        string divSchemes_phi_U = getFullSchemesName(divSchemesDict, "div(phi,U)");
        int divSchemes_phi_UInt = getIntSchemes(divSchemes_phi_U);
        schemes_para.divSchemes_phi_U = static_cast<INTERPOLATION_SCHEME>(divSchemes_phi_UInt);
        std::cout << "    == Schemes setting [divSchemes div(phi,U)]: " << divSchemes_phi_U << ", " << divSchemes_phi_UInt << std::endl;

        string divSchemes_phi_K = getFullSchemesName(divSchemesDict, "div(phi,K)");
        int divSchemes_phi_KInt = getIntSchemes(divSchemes_phi_K);
        schemes_para.divSchemes_phi_K = static_cast<INTERPOLATION_SCHEME>(divSchemes_phi_KInt);
        std::cout << "    == Schemes setting [divSchemes div(phi,K)]: " << divSchemes_phi_K << ", " << divSchemes_phi_KInt << std::endl;

        string divSchemes_phid_p = getFullSchemesName(divSchemesDict, "div(phid,p)");
        int divSchemes_phid_pInt = getIntSchemes(divSchemes_phid_p);
        schemes_para.divSchemes_phid_p = static_cast<INTERPOLATION_SCHEME>(divSchemes_phid_pInt);
        std::cout << "    == Schemes setting [divSchemes div(phid,p)]: " << divSchemes_phid_p << ", " << divSchemes_phid_pInt << std::endl;

        string divSchemes_phi_Yih = getFullSchemesName(divSchemesDict, "div(phi,Yi_h)");
        int divSchemes_phi_YihInt = getIntSchemes(divSchemes_phi_Yih);
        schemes_para.divSchemes_phi_Yih = static_cast<INTERPOLATION_SCHEME>(divSchemes_phi_YihInt);
        std::cout << "    == Schemes setting [divSchemes div(phi,Yi_h)]: " << divSchemes_phi_Yih << ", " << divSchemes_phi_YihInt << std::endl;

        string divSchemes_phi_k = getFullSchemesName(divSchemesDict, "div(phi,k)");
        int divSchemes_phi_kInt = getIntSchemes(divSchemes_phi_k);
        schemes_para.divSchemes_phi_k = static_cast<INTERPOLATION_SCHEME>(divSchemes_phi_kInt);
        std::cout << "    == Schemes setting [divSchemes div(phi,k)]: " << divSchemes_phi_k << ", " << divSchemes_phi_kInt << std::endl;

        string divSchemes_phi_epsilon = getFullSchemesName(divSchemesDict, "div(phi,epsilon)");
        int divSchemes_phi_epsInt = getIntSchemes(divSchemes_phi_epsilon);
        schemes_para.divSchemes_phi_epsilon = static_cast<INTERPOLATION_SCHEME>(divSchemes_phi_epsInt);
        std::cout << "    == Schemes setting [divSchemes div(phi,k)]: " << divSchemes_phi_epsilon << ", " << divSchemes_phi_epsInt << std::endl;

        // string divSchemes_hDiffCorrFlux = getFullSchemesName(divSchemesDict, "div(hDiffCorrFlux)");
        // int divSchemes_hDiffCorrFluxInt = getIntSchemes(divSchemes_hDiffCorrFlux);
        // schemes_para.divSchemes_hDiffCorrFlux = static_cast<INTERPOLATION_SCHEME>(divSchemes_hDiffCorrFluxInt);
        // std::cout << "    == Schemes setting [divSchemes div(hDiffCorrFlux)]: " << divSchemes_hDiffCorrFlux << ", " << divSchemes_hDiffCorrFluxInt << std::endl;

        // string divSchemes_rho_nuEff = getFullSchemesName(divSchemesDict, "div(((rho*nuEff)*dev2(T(grad(U)))))");
        // int divSchemes_rho_nuEffInt = getIntSchemes(divSchemes_rho_nuEff);
        // schemes_para.divSchemes_rho_nuEff = static_cast<INTERPOLATION_SCHEME>(divSchemes_rho_nuEffInt);
        // std::cout << "    == Schemes setting [divSchemes div(((rho*nuEff)*dev2(T(grad(U)))))]: " << divSchemes_rho_nuEff << ", " << divSchemes_rho_nuEffInt << std::endl;
    }

    set_schemes(schemes_para);

}

void createGPUGAMGInput(const lduMesh& mesh, const dictionary solverControls)
{
    std::cout << " -- Set GAMG variables for DeepFlame_Academic " << std::endl;    

    // read control files
    int nVcycles = 1;
    int nCellsInCoarsestLevel = 32;
    int nPreSweeps = 2;
    int nPostSweeps = 2;
    int nFinestSweeps = 2;
    bool interpolateCorrection = false;
    bool scaleCorrection = true;
    bool solveCoarsest = false;

    if(solverControls.found("nVcycles")) solverControls.lookup("nVcycles") >> nVcycles;
    if(solverControls.found("nCellsInCoarsestLevel")) solverControls.lookup("nCellsInCoarsestLevel") >> nCellsInCoarsestLevel;
    if(solverControls.found("nPreSweeps")) solverControls.lookup("nPreSweeps") >> nPreSweeps;
    if(solverControls.found("nPostSweeps")) solverControls.lookup("nPostSweeps") >> nPostSweeps;
    if(solverControls.found("nFinestSweeps")) solverControls.lookup("nFinestSweeps") >> nFinestSweeps;

    if(solverControls.found("interpolateCorrection")) solverControls.lookup("interpolateCorrection") >> interpolateCorrection;
    if(solverControls.found("scaleCorrection")) solverControls.lookup("scaleCorrection") >> scaleCorrection;
    if(solverControls.found("solveCoarsest")) solverControls.lookup("solveCoarsest") >> solveCoarsest;

    std::cout << "   == GAMG config nVcycles: " << nVcycles << std::endl; 
    std::cout << "   == GAMG config nCellsInCoarsestLevel: " << nCellsInCoarsestLevel << std::endl; 
    std::cout << "   == GAMG config nPreSweeps: " << nPreSweeps << std::endl; 
    std::cout << "   == GAMG config nPostSweeps: " << nPostSweeps << std::endl; 
    std::cout << "   == GAMG config nFinestSweeps: " << nFinestSweeps << std::endl; 

    std::cout << "   == GAMG config interpolateCorrection: " << interpolateCorrection << std::endl;  
    std::cout << "   == GAMG config scaleCorrection: " << scaleCorrection << std::endl; 
    std::cout << "   == GAMG config solveCoarsest: " << solveCoarsest << std::endl; 
    
    int sparse_type = -1;
#ifdef iscsr
    #include "CSRGAMGAgglomeration.H"
    const CSRGAMGAgglomeration& agglomeration(CSRGAMGAgglomeration::New(mesh, solverControls));
    sparse_type = 1;
#else
    #include "ELLGAMGAgglomeration.H"
    const ELLGAMGAgglomeration& agglomeration(ELLGAMGAgglomeration::New(mesh, solverControls));
    sparse_type = 2;
#endif

    int agglomeration_level = agglomeration.size();
    std::cout << "   == GAMG agglomeration level: " << agglomeration_level << std::endl;

#ifdef iscsr
    h_ldu_to_csr_no_diag = (int**)malloc(sizeof(int*)*agglomeration_level);
    h_csr_row_index_no_diag = (int**)malloc(sizeof(int*)*agglomeration_level);
    h_csr_col_index_no_diag = (int**)malloc(sizeof(int*)*agglomeration_level);
#else
    h_ell_row_maxcount = (int*)malloc(sizeof(int)*agglomeration_level);
    h_ellCols = (int**)malloc(sizeof(int*)*agglomeration_level);
    h_ldu2ellIndex = (int**)malloc(sizeof(int*)*agglomeration_level);
#endif    

    // ===========================================================================
    // input variables for new linear solver, include in DeepFlame_Academic
    GAMG_control_para GAMG_configs;
    GAMG_mesh_para GAMG_maps[agglomeration_level];

    GAMG_configs.agglomeration_level = agglomeration_level;
    GAMG_configs.cycle_type = 1;
    GAMG_configs.nCellsInCoarsestLevel = nCellsInCoarsestLevel;
    GAMG_configs.nVcycles = nVcycles;
    GAMG_configs.nPreSweeps = nPreSweeps;
    GAMG_configs.nPostSweeps = nPostSweeps;
    GAMG_configs.nFinestSweeps = nFinestSweeps;
    GAMG_configs.interpolateCorrection = interpolateCorrection;
    GAMG_configs.scaleCorrection = scaleCorrection;
    GAMG_configs.solveCoarsest = solveCoarsest;

    forAll(agglomeration, leveli)
    {
        std::vector<int> upperAddrVector( agglomeration.meshLevel(leveli).lduAddr().upperAddr().begin(), 
                                          agglomeration.meshLevel(leveli).lduAddr().upperAddr().end());
        std::vector<int> lowerAddrVector( agglomeration.meshLevel(leveli).lduAddr().lowerAddr().begin(), 
                                          agglomeration.meshLevel(leveli).lduAddr().lowerAddr().end());

        std::vector<int> restrictMapVector( agglomeration.restrictAddressing(leveli).begin(), 
                                            agglomeration.restrictAddressing(leveli).end());
        std::vector<int> faceRestrictMapVector( agglomeration.faceRestrictAddressing(leveli).begin(), 
                                                agglomeration.faceRestrictAddressing(leveli).end());
        std::vector<bool> faceFlipMapVector( agglomeration.faceFlipMap(leveli).begin(), 
                                             agglomeration.faceFlipMap(leveli).end());
        
        // TODO: bool2int needtest: all false in this case
        std::vector<int> faceFlipMapVector_int(faceFlipMapVector.size());
        std::transform(faceFlipMapVector.begin(), faceFlipMapVector.end(), 
                        std::back_inserter(faceFlipMapVector_int), [](bool b) { return static_cast<int>(b); });

        GAMG_maps[leveli].nCell = restrictMapVector.size();
        GAMG_maps[leveli].nFace = faceRestrictMapVector.size();

        GAMG_maps[leveli].upperAddr  = new int[GAMG_maps[leveli].nFace];
        GAMG_maps[leveli].lowerAddr  = new int[GAMG_maps[leveli].nFace];

        GAMG_maps[leveli].restrictMap = new int[GAMG_maps[leveli].nCell];
        GAMG_maps[leveli].faceRestrictMap = new int[GAMG_maps[leveli].nFace];
        GAMG_maps[leveli].faceFlipMap  = new int[GAMG_maps[leveli].nFace];

        memcpy(GAMG_maps[leveli].upperAddr, upperAddrVector.data(), GAMG_maps[leveli].nFace * sizeof(int));
        memcpy(GAMG_maps[leveli].lowerAddr, lowerAddrVector.data(), GAMG_maps[leveli].nFace * sizeof(int));

        memcpy(GAMG_maps[leveli].restrictMap, restrictMapVector.data(), GAMG_maps[leveli].nCell * sizeof(int));
        memcpy(GAMG_maps[leveli].faceRestrictMap, faceRestrictMapVector.data(), GAMG_maps[leveli].nFace * sizeof(int));
        memcpy(GAMG_maps[leveli].faceFlipMap, faceFlipMapVector_int.data(), GAMG_maps[leveli].nFace * sizeof(int));

        setConstantIndexes_GAMG(GAMG_maps[leveli].nCell, GAMG_maps[leveli].nFace, &GAMG_maps[leveli].lowerAddr[0], 
                &GAMG_maps[leveli].upperAddr[0], leveli, sparse_type);

        std::cout << "    === level: " << leveli << ", cell: " << GAMG_maps[leveli].nCell
                                                << ", face: " << GAMG_maps[leveli].nFace << std::endl;

        int patchSize = agglomeration.nPatchFaces(0).size();
        GAMG_maps[leveli].patchSize = patchSize;
        GAMG_maps[leveli].nPatchFaces = (int*)malloc(sizeof(int)*patchSize);

        if(leveli == 0)
        {
            std::vector<int> nPatchFacesVector( agglomeration.nPatchFaces(leveli).begin(), 
                                                agglomeration.nPatchFaces(leveli).end());
            std::copy(nPatchFacesVector.begin(), nPatchFacesVector.end(), GAMG_maps[leveli].nPatchFaces);
        }
        else
        {
            std::vector<int> nPatchFacesVector( agglomeration.nPatchFaces(leveli-1).begin(), 
                                                agglomeration.nPatchFaces(leveli-1).end());
            std::copy(nPatchFacesVector.begin(), nPatchFacesVector.end(), GAMG_maps[leveli].nPatchFaces);
        }

        GAMG_maps[leveli].faceCells            = new int*[patchSize]();
        GAMG_maps[leveli].patchFaceRestrictMap = new int*[patchSize]();

        for(int patchi=0; patchi < patchSize; patchi++)
        {
            if (GAMG_maps[leveli].nPatchFaces[patchi] > 0)
            {
                std::vector<int> faceCellsVector (  agglomeration.interfaceLevel(leveli)[patchi].faceCells().begin(),
                                                    agglomeration.interfaceLevel(leveli)[patchi].faceCells().end());
                std::vector<int> patchFaceRestrictMapVector(agglomeration.patchFaceRestrictAddressing(leveli)[patchi].begin(), 
                                                            agglomeration.patchFaceRestrictAddressing(leveli)[patchi].end());
                if (leveli == 0)
                {
                    GAMG_maps[leveli].nPatchFaces[patchi] = patchFaceRestrictMapVector.size();
                }

                GAMG_maps[leveli].faceCells[patchi]            = (int*)malloc(sizeof(int)*GAMG_maps[leveli].nPatchFaces[patchi]);
                GAMG_maps[leveli].patchFaceRestrictMap[patchi] = (int*)malloc(sizeof(int)*GAMG_maps[leveli].nPatchFaces[patchi]);

                std::copy(faceCellsVector.begin(), faceCellsVector.end(), 
                            GAMG_maps[leveli].faceCells[patchi]);
                std::copy(patchFaceRestrictMapVector.begin(), patchFaceRestrictMapVector.end(), 
                            GAMG_maps[leveli].patchFaceRestrictMap[patchi]);
            }
        }
    }  

    set_GAMG(GAMG_configs, &GAMG_maps[0]);  

#ifdef iscsr
    set_GAMG_matrix_format_csr(h_ldu_to_csr_no_diag, h_csr_row_index_no_diag, h_csr_col_index_no_diag);
#else
    set_GAMG_matrix_format_ell(h_ell_row_maxcount, h_ellCols, h_ldu2ellIndex);
#endif

}

void createGPUSolverInput(const lduMesh& mesh, const dictionary solverControls, MATRIX_EQUATION EqnID, string AMGXsetting)
{
    std::cout << "- Set linear solver variables for DeepFlame_Academic: " << EqnID << std::endl;    
    
    matrix_solver_para solver_configs;

    int minIter=-1, maxIter=-1; // set flag
    double tolerance=-1.0, relTol=-1.0;

    solverControls.readIfPresent("minIter",   minIter);
    solverControls.readIfPresent("maxIter",   maxIter);
    solverControls.readIfPresent("tolerance", tolerance);
    solverControls.readIfPresent("relTol",    relTol);

    std::cout << "  = Solver config minIter: "   << minIter   << std::endl;
    std::cout << "  = Solver config maxIter: "   << maxIter   << std::endl;
    std::cout << "  = Solver config tolerance: " << tolerance << std::endl;
    std::cout << "  = Solver config relTol: "    << relTol    << std::endl;

    if (minIter   >= 0) solver_configs.minIter   = minIter;
    if (maxIter   >= 0) solver_configs.maxIter   = maxIter;
    if (tolerance >= 0) solver_configs.tolerance = tolerance;
    if (relTol    >= 0) solver_configs.reltol    = relTol;

    // Set solver
    const word solver(solverControls.lookup("solver"));
    std::cout << "  = Solver config solver: " << solver << std::endl;

    // read control files
    word preconditioner, smoother;

    bool hasPreconditioner = solverControls.found("preconditioner");
    if (hasPreconditioner)
    {
        const entry& eP = solverControls.lookupEntry("preconditioner", false, true);
        if (eP.isDict())
        {
            eP.dict().lookup("preconditioner") >> preconditioner;
            eP.dict().lookup("smoother") >> smoother;

            std::cout << "  = Solver config preconditioner: " << preconditioner << std::endl;
            std::cout << "  = Solver config smoother: " << smoother << std::endl;
        }
        else
        {
            eP.stream() >> preconditioner;
            std::cout << "  = Solver config preconditioner: " << preconditioner << std::endl;
        }
    }

    // SET MATRIX_SOLVER
    int matrix_solver = -1;
    if (solver == "GAMG") 
    {
        matrix_solver = 1;
        createGPUGAMGInput(mesh, solverControls);
    }
    if (solver == "PBiCGStab") matrix_solver = 2;
    if (solver == "PCG")       matrix_solver = 3;
    if (solver == "diagonal")  matrix_solver = 0;

    // SET MATRIX_PRECONDITIONER
    int matrix_preconditioner = -1;
    if (preconditioner == "GAMG") 
    {
        matrix_preconditioner = 0;
        const entry& eP = solverControls.lookupEntry("preconditioner", false, false);
        createGPUGAMGInput(mesh, eP.dict());
    }

    // SET MATRIX_SMOOTHER
    int matrix_smoother = -1;
    if (smoother == "Jacobi") matrix_smoother = 0;
    if (smoother == "SRJ")    matrix_smoother = 1;

    // SET MATRIX_SPARSE_FORMAT
#ifdef iscsr
    int sparse_format = 0; // CSR
#else
    int sparse_format = 1; // ELL
#endif    

    if (!AMGXsetting.empty()) 
    {
        const char* SettingPath_char = AMGXsetting.c_str();

        size_t length = strlen(SettingPath_char) + 1;
        solver_configs.AMGXsettingPath = new char[length];
        strcpy(solver_configs.AMGXsettingPath, SettingPath_char);

        sparse_format = 2; // AMGX
    }

    set_matrix_solver(  solver_configs, 
                        static_cast<MATRIX_SMOOTHER>(matrix_smoother),
                        static_cast<MATRIX_PRECONDITIONER>(matrix_preconditioner),
                        static_cast<MATRIX_SOLVER>(matrix_solver),
                        static_cast<MATRIX_SPARSE_FORMAT>(sparse_format),
                        EqnID );

}

void createGPUBaseInput(mesh_info_para &mesh_paras, init_data_para &init_data, 
                        const IOdictionary& CanteraTorchProperties, fvMesh& mesh, PtrList<volScalarField>& Y)
{
    std::cout << "- Set mesh info variables for DeepFlame_Academic   " << std::endl;   

    // prepare constant values: num_cells, num_surfaces, num_boundary_surfaces,
    // num_patches, patch_size, num_species, rdelta_t
    const labelUList& owner = mesh.owner();
    const labelUList& neighbour = mesh.neighbour();
    int num_cells = mesh.nCells();
    int num_total_cells = Foam::returnReduce(num_cells, sumOp<label>());
    int num_surfaces = neighbour.size();
    int num_boundary_surfaces = 0;
    int num_patches = 0;
    std::vector<int> patch_size, patch_offset, patch_start;
    forAll(mesh.boundary(), patchi) 
    {
        patch_offset.push_back(num_boundary_surfaces);
        patch_start.push_back(mesh.boundary()[patchi].start());

        const fvPatchScalarField& patchY = Y[0].boundaryField()[patchi];
        int patchsize = patchY.size();
        patch_size.push_back(patchsize);
        if (patchY.type() == "processor"
            || patchY.type() == "processorCyclic") {
            num_boundary_surfaces += patchsize * 2; // patchNeighbourfield and patchInternalfield
        } else {
            num_boundary_surfaces += patchsize;
        }
        num_patches++;
        Info<< "num_boundary_surfaces: " << num_boundary_surfaces << ", patchsize: " << patchsize << endl;
    }

    // For manipulate_matrix
    int nInternalFaces = mesh.nInternalFaces();
    int *cell2face = new int[num_cells * 6]();
    forAll(mesh.cells(), celli)
    {
        const cell& cFaces = mesh.cells()[celli];
        int cFaceSize = cFaces.size();

        if (cFaceSize <= 6)
        {
            for (int j = 0; j < 6; j++)
            {
                if (j < cFaceSize) 
                {
                    cell2face[6 * celli + j] = cFaces[j];
                }
                else 
                {
                    cell2face[6 * celli + j] = -1;
                }
            }
        }
        else 
        {
            std::cout << "Error! Need enlarge cFaceSize (default=6)." << std::endl;
            abort();
        }
    }

    int *face2patch = new int[num_boundary_surfaces]();
    forAll(mesh.boundary(), patchi) 
    {
        for(int facei=0; facei < patch_size[patchi]; facei++) 
        {
            int faceIndex = patch_start[patchi] + facei;
            face2patch[facei+patch_offset[patchi]] = mesh.boundaryMesh().whichPatch(faceIndex);
        }
    }

    // TODO: get deltaT fomr time API
    double rDeltaT = 1 / 1e-6;

    // wyr: now there is no other place to prepare nccl info, thus mpi must be initialized at beginning.
    std::vector<int> GPUNeighbProcNo(num_patches, -1);
    std::vector<int> patch_type(num_patches, -1), interfaceFlag(num_patches, 0);
    // get basic communication info from of
    forAll(Y[0].boundaryField(), patchi) {
        if (Y[0].boundaryField()[patchi].type() == "processor"
            || Y[0].boundaryField()[patchi].type() == "processorCyclic") {
            GPUNeighbProcNo[patchi] = dynamic_cast<const processorFvPatchField<scalar>&>(Y[0].boundaryField()[patchi]).neighbProcNo();
        }
    }

    // get cyclic neighbor when has cyclic patch
    // - get boundary Index
    std::map<std::string, int> patchNameToIndex;
    forAll(Y[0].boundaryField(), patchi) {
        patchNameToIndex[Y[0].boundaryField()[patchi].patch().name()] = patchi;
    }
    // - get cyclic neighbor
    std::vector<int> cyclicNeighbor(num_patches, -1);
    forAll(Y[0].boundaryField(), patchi) {
        if (Y[0].boundaryField()[patchi].type() == "cyclic") {
            cyclicNeighbor[patchi] = patchNameToIndex[dynamic_cast<const cyclicFvPatchField<scalar>&>(Y[0].boundaryField()[patchi]).
                    cyclicPatch().cyclicPatch().neighbPatch().name()];
        }
    } 

    // prepare internal and boundary of sf, mag_sf, weights, delta_coeffs, volume
    double *boundary_sf = new double[3 * num_boundary_surfaces];
    double *boundary_mag_sf = new double[num_boundary_surfaces];
    double *boundary_delta_coeffs = new double[num_boundary_surfaces];
    double *boundary_weights = new double[num_boundary_surfaces];
    double *boundary_weights_delta = new double[3 * num_boundary_surfaces];
    int *boundary_face_cell = new int[num_boundary_surfaces];
    std::vector<int> patch_type_calculated(num_patches, 5); // default patch type is calculated
    std::vector<int> patch_type_extropolated(num_patches, 8); // default patch type is extrapolated

    bool wedgeDataMalloc = false;
    int offset = 0;
    forAll(mesh.boundary(), patchi) {
        const fvPatchScalarField& patchY = Y[0].boundaryField()[patchi];
        const vectorField& pSf = mesh.Sf().boundaryField()[patchi];
        const scalarField& pMagSf = mesh.magSf().boundaryField()[patchi];
        const scalarField& pDeltaCoeffs = mesh.deltaCoeffs().boundaryField()[patchi];
        const scalarField& pWeights = mesh.surfaceInterpolation::weights().boundaryField()[patchi];
        const vectorField& pWeightsDelta = mesh.surfaceInterpolation::weights().boundaryField()[patchi].patch().delta();
        const labelUList& pFaceCells = mesh.boundary()[patchi].faceCells();

        int patchsize = pMagSf.size();

        if (patchY.type() == "processor") {
            patch_type[patchi] = 7;
            interfaceFlag[patchi] = 1;
            patch_type_calculated[patchi] = 7; // patchi type is processor
            patch_type_extropolated[patchi] = 7; // patchi type is processor

            memcpy(boundary_sf + 3*offset, &pSf[0][0], 3*patchsize*sizeof(double));
            memcpy(boundary_sf + 3*offset + 3*patchsize, &pSf[0][0], 3*patchsize*sizeof(double));

            memcpy(boundary_weights_delta + 3*offset, &pWeightsDelta[0], 3*patchsize*sizeof(double));
            memcpy(boundary_weights_delta + 3*offset + 3*patchsize, &pWeightsDelta[0], 3*patchsize*sizeof(double));

            memcpy(boundary_mag_sf + offset, &pMagSf[0], patchsize*sizeof(double));
            memcpy(boundary_mag_sf + offset + patchsize, &pMagSf[0], patchsize*sizeof(double));

            memcpy(boundary_delta_coeffs + offset, &pDeltaCoeffs[0], patchsize*sizeof(double));
            memcpy(boundary_delta_coeffs + offset + patchsize, &pDeltaCoeffs[0], patchsize*sizeof(double));

            memcpy(boundary_weights + offset, &pWeights[0], patchsize*sizeof(double));
            memcpy(boundary_weights + offset + patchsize, &pWeights[0], patchsize*sizeof(double));

            memcpy(boundary_face_cell + offset, &pFaceCells[0], patchsize * sizeof(int));
            memcpy(boundary_face_cell + offset + patchsize, &pFaceCells[0], patchsize * sizeof(int));

            offset += patchsize * 2;
        } else if (patchY.type() == "processorCyclic") {
            patch_type[patchi] = 10;
            interfaceFlag[patchi] = 1;
            patch_type_calculated[patchi] = 10; // patchi type is processorCyclic
            patch_type_extropolated[patchi] = 10; // patchi type is processorCyclic

            memcpy(boundary_sf + 3*offset, &pSf[0][0], 3*patchsize*sizeof(double));
            memcpy(boundary_sf + 3*offset + 3*patchsize, &pSf[0][0], 3*patchsize*sizeof(double));

            memcpy(boundary_weights_delta + 3*offset, &pWeightsDelta[0], 3*patchsize*sizeof(double));
            memcpy(boundary_weights_delta + 3*offset + 3*patchsize, &pWeightsDelta[0], 3*patchsize*sizeof(double));            

            memcpy(boundary_mag_sf + offset, &pMagSf[0], patchsize*sizeof(double));
            memcpy(boundary_mag_sf + offset + patchsize, &pMagSf[0], patchsize*sizeof(double));

            memcpy(boundary_delta_coeffs + offset, &pDeltaCoeffs[0], patchsize*sizeof(double));
            memcpy(boundary_delta_coeffs + offset + patchsize, &pDeltaCoeffs[0], patchsize*sizeof(double));

            memcpy(boundary_weights + offset, &pWeights[0], patchsize*sizeof(double));
            memcpy(boundary_weights + offset + patchsize, &pWeights[0], patchsize*sizeof(double));

            memcpy(boundary_face_cell + offset, &pFaceCells[0], patchsize * sizeof(int));
            memcpy(boundary_face_cell + offset + patchsize, &pFaceCells[0], patchsize * sizeof(int));

            offset += patchsize * 2;
        } else if (patchY.type() == "cyclic") {
            patch_type[patchi] = 6;
            patch_type_calculated[patchi] = 6; // patchi type is cyclic
            patch_type_extropolated[patchi] = 6; // patchi type is cyclic

            memcpy(boundary_sf + 3*offset, &pSf[0][0], 3*patchsize*sizeof(double));
            memcpy(boundary_weights_delta + 3*offset, &pWeightsDelta[0], 3*patchsize*sizeof(double));
            memcpy(boundary_mag_sf + offset, &pMagSf[0], patchsize*sizeof(double));
            memcpy(boundary_delta_coeffs + offset, &pDeltaCoeffs[0], patchsize*sizeof(double));
            memcpy(boundary_weights + offset, &pWeights[0], patchsize*sizeof(double));
            memcpy(boundary_face_cell + offset, &pFaceCells[0], patchsize * sizeof(int));

            offset += patchsize;
        } else {
            memcpy(boundary_sf + 3*offset, &pSf[0][0], 3*patchsize*sizeof(double));
            memcpy(boundary_weights_delta + 3*offset, &pWeightsDelta[0], 3*patchsize*sizeof(double));
            memcpy(boundary_mag_sf + offset, &pMagSf[0], patchsize*sizeof(double));
            memcpy(boundary_delta_coeffs + offset, &pDeltaCoeffs[0], patchsize*sizeof(double));
            memcpy(boundary_weights + offset, &pWeights[0], patchsize*sizeof(double));
            memcpy(boundary_face_cell + offset, &pFaceCells[0], patchsize * sizeof(int));

            offset += patchsize;
        }

        if (patchY.type() == "wedge")
        {
            patch_type[patchi] = 13;
            patch_type_calculated[patchi] = 13; // patchi type is wedge
            patch_type_extropolated[patchi] = 13; // patchi type is wedge
            if (!wedgeDataMalloc) 
            {
                mesh_paras.faceT = new double[num_patches * 9]();
                mesh_paras.cellT = new double[num_patches * 9]();
                wedgeDataMalloc = true;
            }
            std::cout << "wedge type patchi: " << patchi << std::endl;

            const tensor& faceT_ = dynamic_cast<const wedgeFvPatch&>(patchY.patch()).faceT();
            Info << "faceT: " << faceT_ << endl;

            const tensor& cellT_ = dynamic_cast<const wedgeFvPatch&>(patchY.patch()).cellT();
            Info << "cellT: " << cellT_ << endl;

            memcpy(mesh_paras.faceT + patchi * 9, &faceT_[0], 9 * sizeof(double));
            memcpy(mesh_paras.cellT + patchi * 9, &cellT_[0], 9 * sizeof(double));
        }
        if (patchY.type() == "empty")
        {
            patch_type[patchi] = 3;
            patch_type_calculated[patchi] = 3; // patchi type is empty
            patch_type_extropolated[patchi] = 3; // patchi type is empty            
        }

        Info << "patchName: " << patchY.patch().name() << endl;
    }

    // construct mesh distance for limitedLinear scheme
    vectorField meshDistance = mesh.Sf();
    forAll(meshDistance, facei) {
        label own = owner[facei];
        label nei = neighbour[facei];
        meshDistance[facei] = mesh.C()[nei] - mesh.C()[own];
    }

    double *meshDistance_newOrder = new double[num_surfaces*3]();
    double *sf_newOrder = new double[num_surfaces*3]();
    permuteVector(meshDistance_newOrder, &meshDistance[0][0], num_surfaces);
    permuteVector(sf_newOrder, &mesh.Sf()[0][0], num_surfaces);

    // SET CONSTANT VALUES
    mesh_paras.num_cells             = num_cells;
    mesh_paras.num_total_cells       = num_total_cells;
    mesh_paras.num_patches           = num_patches;
    mesh_paras.num_surfaces          = num_surfaces;
    mesh_paras.num_boundary_surfaces = num_boundary_surfaces;

    init_data.rDeltaT                = rDeltaT;
    init_data.num_species            = Y.size();
    init_data.patch_type_calculated   = new int[num_patches]();
    init_data.patch_type_extropolated = new int[num_patches]();

    std::copy( patch_type_calculated.begin(), patch_type_calculated.end(), init_data.patch_type_calculated ); 
    std::copy( patch_type_extropolated.begin(), patch_type_extropolated.end(), init_data.patch_type_extropolated ); 
    
    mesh_paras.patch_size     = new int[num_patches]();
    mesh_paras.patch_offset   = new int[num_patches]();
    mesh_paras.patch_start    = new int[num_patches]();
    mesh_paras.neighbProcNo   = new int[num_patches]();
    mesh_paras.cyclicNeighbor = new int[num_patches]();
    mesh_paras.interfaceFlag  = new int[num_patches]();

    std::copy( patch_size.begin(),      patch_size.end(),      mesh_paras.patch_size );
    std::copy( patch_offset.begin(),    patch_offset.end(),    mesh_paras.patch_offset );
    std::copy( patch_start.begin(),     patch_start.end(),     mesh_paras.patch_start );
    std::copy( GPUNeighbProcNo.begin(), GPUNeighbProcNo.end(), mesh_paras.neighbProcNo );
    std::copy( cyclicNeighbor.begin(),  cyclicNeighbor.end(),  mesh_paras.cyclicNeighbor );   
    std::copy( interfaceFlag.begin(),   interfaceFlag.end(),   mesh_paras.interfaceFlag ); 

    mesh_paras.lowerAddr             = &owner[0];
    mesh_paras.upperAddr             = &neighbour[0];   

    mesh_paras.mag_sf                = &mesh.magSf()[0];
    mesh_paras.delta_coeffs          = &mesh.deltaCoeffs()[0];
    mesh_paras.weight                = &mesh.surfaceInterpolation::weights()[0];
    mesh_paras.volume                = &mesh.V()[0];

    mesh_paras.mesh_dis              = new double[num_surfaces*3];
    mesh_paras.sf                    = new double[num_surfaces*3];

    mesh_paras.cell2face             = new int[num_cells*6];
    mesh_paras.face2patch            = new int[num_boundary_surfaces];
    mesh_paras.boundary_face_cell    = new int[num_boundary_surfaces];
    mesh_paras.boundary_sf           = new double[num_boundary_surfaces*3];
    mesh_paras.boundary_mag_sf       = new double[num_boundary_surfaces];
    mesh_paras.boundary_delta_coeffs = new double[num_boundary_surfaces];
    mesh_paras.boundary_weight       = new double[num_boundary_surfaces];
    mesh_paras.boundary_weight_delta = new double[num_boundary_surfaces*3];

    double *boundary_sf_newOrder = new double[num_boundary_surfaces*3]();
    permuteVector(boundary_sf_newOrder, &boundary_sf[0], num_boundary_surfaces);

    memcpy(mesh_paras.mesh_dis, &meshDistance_newOrder[0], num_surfaces * 3 * sizeof(double));
    memcpy(mesh_paras.sf,       &sf_newOrder[0],           num_surfaces * 3 * sizeof(double));

    memcpy(mesh_paras.cell2face,             &cell2face[0],              num_cells * 6 * sizeof(int));
    memcpy(mesh_paras.face2patch,            &face2patch[0],             num_boundary_surfaces * sizeof(int));
    memcpy(mesh_paras.boundary_face_cell,    &boundary_face_cell[0],     num_boundary_surfaces * sizeof(int));
    memcpy(mesh_paras.boundary_sf,           &boundary_sf_newOrder[0],   num_boundary_surfaces * 3 * sizeof(double));
    memcpy(mesh_paras.boundary_mag_sf,       &boundary_mag_sf[0],        num_boundary_surfaces * sizeof(double));
    memcpy(mesh_paras.boundary_delta_coeffs, &boundary_delta_coeffs[0],  num_boundary_surfaces * sizeof(double));
    memcpy(mesh_paras.boundary_weight,       &boundary_weights[0],       num_boundary_surfaces * sizeof(double));
    memcpy(mesh_paras.boundary_weight_delta, &boundary_weights_delta[0], num_boundary_surfaces * 3 * sizeof(double));

    delete boundary_sf;
    delete boundary_mag_sf;
    delete boundary_delta_coeffs;
    delete boundary_weights;
    delete boundary_face_cell;
    delete cell2face;

    const lduInterfacePtrsList interfaces(mesh.interfaces());
    const lduAddressing& lduAddr = mesh.lduAddr();
    label nProcValues = 0;
    forAll(interfaces, patchi)
    {
        if (interfaces.set(patchi))
        {
            const labelUList& faceCells = lduAddr.patchAddr(patchi);
            const label len = faceCells.size();
            nProcValues += len;
        }
    }
    mesh_paras.num_proc_surfaces = nProcValues;
}


void createGPUSparseFormat(mesh_info_para &mesh_paras, fvMesh& mesh)
{
    std::cout << "- Set sparse format variables for DeepFlame_Academic   " << std::endl;   

    // prepare constant values: num_cells, num_surfaces, num_boundary_surfaces,
    // num_patches, patch_size, num_species, rdelta_t
    const labelUList& owner = mesh.owner();
    const labelUList& neighbour = mesh.neighbour();

    // prepare interface info
    const globalIndex globalNumbering(mesh_paras.num_cells);
    const lduInterfacePtrsList interfaces(mesh.interfaces());
    const lduAddressing& lduAddr = mesh.lduAddr();
    labelList globalCells(mesh_paras.num_cells);
    forAll(globalCells, celli)
    {
        globalCells[celli] = globalNumbering.toGlobal(Pstream::myProcNo(), celli);
    }
    const label nReq = Pstream::nRequests();
    label nProcValues = 0;
    // send global cells
    forAll(interfaces, patchi)
    {
        if (interfaces.set(patchi))
        {
            nProcValues += lduAddr.patchAddr(patchi).size();

            // send patchInternalField
            interfaces[patchi].initInternalFieldTransfer
            (
                Pstream::commsTypes::nonBlocking,
                globalCells
            );
        }
    }

    // prepare constant indexes: owner, neighbor, procRows, procCols
    if (Pstream::parRun())
    {
        Pstream::waitRequests(nReq);
    }
    labelField procRows(nProcValues, 0);
    labelField procCols(nProcValues, 0);
    nProcValues = 0;

    forAll(interfaces, patchi)
    {
        if (interfaces.set(patchi))
        {
            // local cell index
            const labelUList& faceCells = lduAddr.patchAddr(patchi);
            const label len = faceCells.size();

            // global col index
            labelField nbrCells
            (
                interfaces[patchi].internalFieldTransfer
                (
                    Pstream::commsTypes::nonBlocking,
                    globalCells
                )
            );

            if (faceCells.size() != nbrCells.size())
            {
                FatalErrorInFunction
                    << "Mismatch in interface sizes (AMI?)" << nl
                    << "Have " << faceCells.size() << " != "
                    << nbrCells.size() << nl
                    << exit(FatalError);
            }

            // for AMGx: Local rows, Global columns
            SubList<label>(procRows, len, nProcValues) = faceCells;
            SubList<label>(procCols, len, nProcValues) = nbrCells;
            nProcValues += len;
        }
    }
    label globalOffset = globalNumbering.toGlobal(0);

    /* For DeepFlame_Academic */
    setConstantIndexes(mesh_paras, &owner[0], &neighbour[0], &procRows[0], &procCols[0], globalOffset, nProcValues);
}


void setConstantIndexes(mesh_info_para &mesh_paras, const int *owner, const int *neighbor, const int *procRows, 
        const int *procCols, int globalOffset, int num_proc_surfaces)
{
    /**
     * @brief ldu2csr with diagonal
     * 
     */
    // build permTmp, rowIndicesTmp, colIndicesTmp
    int num_cells = mesh_paras.num_cells;
    int num_surfaces = mesh_paras.num_surfaces;
    int num_Nz = num_cells + 2 * num_surfaces + num_proc_surfaces;
    int ell_row_maxcount = 0;

    std::vector<int> lduCSRIndex;
    std::vector<int> lduCSRNoDiagIndex;

    std::vector<int> permTmp(num_Nz);
    std::iota(permTmp.begin(), permTmp.end(), 0);

    // rowIndex of: low, diag, upp, proc
    std::vector<int> rowIndicesTmp(num_Nz);
    std::copy(neighbor, neighbor + num_surfaces, rowIndicesTmp.begin()); // row index of lower entry
    std::iota(rowIndicesTmp.begin() + num_surfaces, rowIndicesTmp.begin() + num_cells + num_surfaces, 0); // row index of diag entry
    std::copy(owner, owner + num_surfaces, rowIndicesTmp.begin() + num_cells + num_surfaces); // row index of upper entry
    std::copy(procRows, procRows + num_proc_surfaces, rowIndicesTmp.begin() + num_cells + 2 * num_surfaces); // row index of proc entry

    // colIndex of: low, diag, upp, proc
    std::vector<int> colIndicesTmp(num_Nz);
    std::copy(owner, owner + num_surfaces, colIndicesTmp.begin()); // col index of lower entry
    std::iota(colIndicesTmp.begin() + num_surfaces, colIndicesTmp.begin() + num_cells + num_surfaces, 0); // col index of diag entry
    std::copy(neighbor, neighbor + num_surfaces, colIndicesTmp.begin() + num_cells + num_surfaces); // col index of upper entry
    std::copy(procCols, procCols + num_proc_surfaces, colIndicesTmp.begin() + num_cells + 2 * num_surfaces); // col index of proc entry

    // premute rowIndicesTmp, get CSRRowIndex and ldu2csrPerm
    std::multimap<int,int> rowIndicesPermutation;
    for (int i = 0; i < num_Nz; ++i){
        rowIndicesPermutation.insert(std::make_pair(rowIndicesTmp[i], permTmp[i]));
    }
    std::vector<std::pair<int, int>> rowIndicesPermPair(rowIndicesPermutation.begin(), rowIndicesPermutation.end());
    
    std::sort(rowIndicesPermPair.begin(), rowIndicesPermPair.end(), []
    (const std::pair<int, int>& pair1, const std::pair<int, int>& pair2){
        if (pair1.first != pair2.first) {
            return pair1.first < pair2.first;
        } else {
            return pair1.second < pair2.second;
        }
    });
    std::vector<int> permRowIndex;
    std::transform(rowIndicesPermPair.begin(), rowIndicesPermPair.end(), std::back_inserter(permRowIndex), []
        (const std::pair<int, int>& pair) {
        return pair.first;
    });
    std::vector<int> CSRRowIndex(num_cells + 1, 0);
    for (int i = 0; i < num_Nz; i++) {
        CSRRowIndex[permRowIndex[i] + 1]++;
    }
    std::partial_sum(CSRRowIndex.begin(), CSRRowIndex.end(), CSRRowIndex.begin());

    std::transform(rowIndicesPermPair.begin(), rowIndicesPermPair.end(), std::back_inserter(lduCSRIndex), []
        (const std::pair<int, int>& pair) {
        return pair.second;
    });

    // get diagCSRIndex
    std::vector<int> diagCSRIndex(num_cells);
    int startIndex = 0;
    for (int i = 0; i < num_cells; i++) {
        int diagIndex = i + num_surfaces; // index of diag entry in permTmp
        for (int j = startIndex; j < num_Nz; j++) {
            if (lduCSRIndex[j] == diagIndex) {
                diagCSRIndex[i] = j;
                startIndex = j + 1;
                break;
            }
        }
    }

    // get CSRColIndex
    // localToGlobalColIndices: add globalOffset to colIndicesTmp
    std::transform(colIndicesTmp.begin(), colIndicesTmp.begin() + num_cells + 2 * num_surfaces, colIndicesTmp.begin(), 
        [globalOffset](int i){return i + globalOffset;});
    
    // permute colIndicesTmp
    std::vector<int> CSRColIndex(num_Nz);
    for (int i = 0; i < num_Nz; ++i){
        CSRColIndex[i] = colIndicesTmp[lduCSRIndex[i]];
    }

    /**
     * @brief ldu2csr with no diagonal
     * 
     */
    permTmp.resize(2 * num_surfaces);
    std::iota(permTmp.begin(), permTmp.end(), 0);

    // rowIndex of: low, upp
    rowIndicesTmp.resize(2 * num_surfaces);
    std::copy(neighbor, neighbor + num_surfaces, rowIndicesTmp.begin()); // row index of lower entry
    std::copy(owner, owner + num_surfaces, rowIndicesTmp.begin() + num_surfaces); // row index of upper entry

    // colIndex of: low, diag, upp, proc
    colIndicesTmp.resize(2 * num_surfaces);
    std::copy(owner, owner + num_surfaces, colIndicesTmp.begin()); // col index of lower entry
    std::copy(neighbor, neighbor + num_surfaces, colIndicesTmp.begin() + num_surfaces); // col index of upper entry

    // premute rowIndicesTmp, get CSRRowIndex and ldu2csrPerm
    rowIndicesPermutation.clear();
    for (int i = 0; i < 2 * num_surfaces; ++i){
        rowIndicesPermutation.insert(std::make_pair(rowIndicesTmp[i], permTmp[i]));
    }
    rowIndicesPermPair.resize(rowIndicesPermutation.size());
    std::copy(rowIndicesPermutation.begin(), rowIndicesPermutation.end(), rowIndicesPermPair.begin());

    std::sort(rowIndicesPermPair.begin(), rowIndicesPermPair.end(), []
    (const std::pair<int, int>& pair1, const std::pair<int, int>& pair2){
        if (pair1.first != pair2.first) {
            return pair1.first < pair2.first;
        } else {
            return pair1.second < pair2.second;
        }
    });
    permRowIndex.clear();
    std::transform(rowIndicesPermPair.begin(), rowIndicesPermPair.end(), std::back_inserter(permRowIndex), []
        (const std::pair<int, int>& pair) {
        return pair.first;
    });
    std::vector<int> CSRRowIndexNoDiag(num_cells + 1, 0);
    for (int i = 0; i < 2 * num_surfaces; i++) {
        CSRRowIndexNoDiag[permRowIndex[i] + 1]++;
    }
    std::partial_sum(CSRRowIndexNoDiag.begin(), CSRRowIndexNoDiag.end(), CSRRowIndexNoDiag.begin());

    std::transform(rowIndicesPermPair.begin(), rowIndicesPermPair.end(), std::back_inserter(lduCSRNoDiagIndex), []
        (const std::pair<int, int>& pair) {
        return pair.second;
    });

    // get CSRColNoDiagIndex
    std::vector<int> CSRColIndexNoDiag(2 * num_surfaces);
    for (int i = 0; i < 2 * num_surfaces; ++i) {
        CSRColIndexNoDiag[i] = colIndicesTmp[lduCSRNoDiagIndex[i]];
    }

    /**
     * @brief ldu2ell with no diagonal
     * 
     */
    ell_row_maxcount = 0;
    int* ellColsperRow;
    ellColsperRow = (int*)malloc(sizeof(int) * num_cells);
    memset(ellColsperRow, 0, sizeof(int) * num_cells);
    for(int i = 0; i < num_surfaces; i++){
        int lowerRow = neighbor[i];
        int upperRow = owner[i];
        ellColsperRow[lowerRow] += 1;
        ellColsperRow[upperRow] += 1;
    }
    for(int i = 0; i < num_cells; i++){
        ell_row_maxcount = std::max(ell_row_maxcount, ellColsperRow[i]);
    }

    int *ellCols;
    ellCols = (int*)malloc(sizeof(int) * num_cells * ell_row_maxcount);
    memset(ellCols, 0, sizeof(int) * num_cells * ell_row_maxcount);

    int *ldu2ellIndex;
    ldu2ellIndex = (int*)malloc(sizeof(int) * num_surfaces * 2);
    memset(ldu2ellIndex, 0, sizeof(int) * num_surfaces * 2);
    std::vector<int> rowOffset(num_cells, 0);
    // lower
    for(int i = 0; i < num_surfaces; i++){
        int row = neighbor[i];
        int index = rowOffset[row] * num_cells + row;
        ldu2ellIndex[i] = index;
        ellCols[index] = owner[i];
        rowOffset[row] += 1;
    }
    // upper
    for(int i = 0; i < num_surfaces; i++){
        int row = owner[i];
        int index = rowOffset[row] * num_cells + row;
        ldu2ellIndex[i + num_surfaces] = index;
        ellCols[index] = neighbor[i];
        rowOffset[row] += 1;
    }

    // TODO: if CSR;
#ifndef iscsr
    set_matrix_format_ell(ell_row_maxcount, ellCols, ldu2ellIndex);
#else
    set_matrix_format_csr(lduCSRNoDiagIndex.data(), CSRRowIndexNoDiag.data(), CSRColIndexNoDiag.data());
#endif

    set_matrix_format_amgx(lduCSRIndex.data(), CSRRowIndex.data(), CSRColIndex.data());

    free(ellColsperRow);
    free(ellCols);
    free(ldu2ellIndex);
}


void setConstantIndexes_GAMG(int nCells, int nFaces, int *owner, int *neighbor, int leveli, int type){
//csr
if(type == 1){
    /**
     * @brief ldu2csr with no diagonal
     * 
     */
    std::vector<int> permTmp(2 * nFaces);
    std::iota(permTmp.begin(), permTmp.end(), 0);

    // rowIndex of: low, upp
    std::vector<int> rowIndicesTmp(2 * nFaces);
    std::copy(neighbor, neighbor + nFaces, rowIndicesTmp.begin()); // row index of lower entry
    std::copy(owner, owner + nFaces, rowIndicesTmp.begin() + nFaces); // row index of upper entry

    // colIndex of: low, diag, upp, proc
    std::vector<int> colIndicesTmp(2 * nFaces);
    std::copy(owner, owner + nFaces, colIndicesTmp.begin()); // col index of lower entry
    std::copy(neighbor, neighbor + nFaces, colIndicesTmp.begin() + nFaces); // col index of upper entry

    // premute rowIndicesTmp, get CSRRowIndex and ldu2csrPerm
    std::multimap<int,int> rowIndicesPermutation;
    for (int i = 0; i < 2 * nFaces; ++i){
        rowIndicesPermutation.insert(std::make_pair(rowIndicesTmp[i], permTmp[i]));
    }
    std::vector<std::pair<int, int>> rowIndicesPermPair(rowIndicesPermutation.size());
    std::copy(rowIndicesPermutation.begin(), rowIndicesPermutation.end(), rowIndicesPermPair.begin());

    std::sort(rowIndicesPermPair.begin(), rowIndicesPermPair.end(), []
    (const std::pair<int, int>& pair1, const std::pair<int, int>& pair2){
        if (pair1.first != pair2.first) {
            return pair1.first < pair2.first;
        } else {
            return pair1.second < pair2.second;
        }
    });
    std::vector<int> permRowIndex;
    std::transform(rowIndicesPermPair.begin(), rowIndicesPermPair.end(), std::back_inserter(permRowIndex), []
        (const std::pair<int, int>& pair) {
        return pair.first;
    });
    std::vector<int> CSRRowIndexNoDiag(nCells + 1, 0);
    for (int i = 0; i < 2 * nFaces; i++) {
        CSRRowIndexNoDiag[permRowIndex[i] + 1]++;
    }
    std::partial_sum(CSRRowIndexNoDiag.begin(), CSRRowIndexNoDiag.end(), CSRRowIndexNoDiag.begin());

    std::vector<int> lduCSRNoDiagIndex_GAMG;
    std::transform(rowIndicesPermPair.begin(), rowIndicesPermPair.end(), std::back_inserter(lduCSRNoDiagIndex_GAMG), []
        (const std::pair<int, int>& pair) {
        return pair.second;
    });

    // get CSRColNoDiagIndex
    std::vector<int> CSRColIndexNoDiag(2 * nFaces);
    for (int i = 0; i < 2 * nFaces; ++i) {
        CSRColIndexNoDiag[i] = colIndicesTmp[lduCSRNoDiagIndex_GAMG[i]];
    }

    h_ldu_to_csr_no_diag[leveli] = (int*)malloc(2 * nFaces * sizeof(int));
    h_csr_row_index_no_diag[leveli] = (int*)malloc((nCells + 1) * sizeof(int));
    h_csr_col_index_no_diag[leveli] = (int*)malloc(2 * nFaces * sizeof(int));
    memcpy(h_ldu_to_csr_no_diag[leveli], lduCSRNoDiagIndex_GAMG.data(), 2 * nFaces * sizeof(int));
    memcpy(h_csr_row_index_no_diag[leveli], CSRRowIndexNoDiag.data(), (nCells + 1) * sizeof(int));
    memcpy(h_csr_col_index_no_diag[leveli], CSRColIndexNoDiag.data(), 2 * nFaces * sizeof(int));
}
// ell
else if (type == 2)
{
    h_ell_row_maxcount[leveli] = 0;
    int* ellColsperRow;
    ellColsperRow = (int*)malloc(sizeof(int) * nCells);
    memset(ellColsperRow, 0, sizeof(int) * nCells);
    for(int i = 0; i < nFaces; i++){
        int lowerRow = neighbor[i];
        int upperRow = owner[i];
        ellColsperRow[lowerRow] += 1;
        ellColsperRow[upperRow] += 1;
    }
    for(int i = 0; i < nCells; i++){
        h_ell_row_maxcount[leveli] = std::max(h_ell_row_maxcount[leveli], ellColsperRow[i]);
    }

    h_ellCols[leveli] = (int*)malloc(nCells * h_ell_row_maxcount[leveli] * sizeof(int));
    h_ldu2ellIndex[leveli] = (int*)malloc(nFaces * 2 * sizeof(int));

    memset(h_ellCols[leveli], 0, nCells * h_ell_row_maxcount[leveli] * sizeof(int));
    memset(h_ldu2ellIndex[leveli], 0, sizeof(int) * nFaces * 2);
    std::vector<int> rowOffset(nCells, 0);
    // lower
    for(int i = 0; i < nFaces; i++){
        int row = neighbor[i];
        int index = rowOffset[row] * nCells + row;
        h_ldu2ellIndex[leveli][i] = index;
        h_ellCols[leveli][index] = owner[i];
        rowOffset[row] += 1;
    }

    // upper
    for(int i = 0; i < nFaces; i++){
        int row = owner[i];
        int index = rowOffset[row] * nCells + row;
        h_ldu2ellIndex[leveli][i + nFaces] = index;
        h_ellCols[leveli][index] = neighbor[i];
        rowOffset[row] += 1;
    }

    free(ellColsperRow);
}
}

void copyGPUResults2Host(mesh_info_para &mesh_paras, volVectorField& U, volScalarField& T, PtrList<volScalarField>& Y, volScalarField& rho, surfaceScalarField& phi)
{
    // DF-A: memcpy results data d2h 
    writeResults();

    double *U_hostOrder = new double[mesh_paras.num_cells * 3]();
    permuteVector2Host(U_hostOrder, resultsData.h_u, mesh_paras.num_cells);
    memcpy(&U[0][0], U_hostOrder, mesh_paras.num_cells * 3 * sizeof(double));

    memcpy(&T[0], resultsData.h_T, mesh_paras.num_cells * sizeof(double));
    memcpy(&rho[0], resultsData.h_rho, mesh_paras.num_cells * sizeof(double));
    memcpy(&phi[0], resultsData.h_phi, mesh_paras.num_surfaces * sizeof(double));

    forAll(Y, speciesI) {
        memcpy(&Y[speciesI][0], resultsData.h_y + speciesI * mesh_paras.num_cells, mesh_paras.num_cells * sizeof(double));
    }

    U.correctBoundaryConditions();

    delete U_hostOrder;
}

void createTurbulenceInput(const dictionary turbulenceControls, mesh_info_para &mesh_paras, const volScalarField& nut, const volScalarField& alphat, 
                            const volScalarField& k, const volScalarField& epsilon)
{
    std::cout << "- Set turbulence model for DeepFlame_Academic " << std::endl;  

    const word simulationType(turbulenceControls.lookup("simulationType"));
    std::cout << "  = simulationType: " << simulationType << std::endl;

    bool useTurbulence = false;
    bool useTCI = false;
    int LESModelTypeInt = 0;
    int RASModelTypeInt = 0;
    std::vector<int> patch_type_nut;
    std::vector<int> patch_type_alphat;
    std::vector<int> patch_type_k;
    std::vector<int> patch_type_epsilon;

    patch_type_nut.resize(mesh_paras.num_patches);
    patch_type_alphat.resize(mesh_paras.num_patches);
    patch_type_k.resize(mesh_paras.num_patches);
    patch_type_epsilon.resize(mesh_paras.num_patches);

    if (simulationType == "laminar")
    {
    }
    else if (simulationType == "LES")
    {
        useTurbulence = true;

        dictionary LESturbDict = turbulenceControls.subDict("LES");
        const word LESModelType(LESturbDict.lookup("LESModel"));
        std::cout << "    == LESModelType: " << LESModelType << std::endl;

        if (LESModelType == "Smagorinsky") LESModelTypeInt = 1;
        if (LESModelType == "Sigma")       LESModelTypeInt = 2;
    }
    else if (simulationType == "RAS")
    {
        useTurbulence = true;

        dictionary RASturbDict = turbulenceControls.subDict("RAS");
        const word RASModelType(RASturbDict.lookup("RASModel"));
        std::cout << "    == RASModelType: " << RASModelType << std::endl;

        if (RASModelType == "kEpsilon") RASModelTypeInt = 1;
    }

    if (useTurbulence)
    {
        forAll(nut.boundaryField(), patchi)
        {
            constructBoundarySelectorPerPatch(&(patch_type_nut[patchi]), nut.boundaryField()[patchi].type());
            constructBoundarySelectorPerPatch(&(patch_type_alphat[patchi]), alphat.boundaryField()[patchi].type());
            constructBoundarySelectorPerPatch(&(patch_type_k[patchi]), k.boundaryField()[patchi].type());
            constructBoundarySelectorPerPatch(&(patch_type_epsilon[patchi]), epsilon.boundaryField()[patchi].type());
        }
    }

    // for turbulence BCs - start
    // get near wall distance
    nearWallDist yDist(k.mesh());
    const volScalarField::Boundary& bf = epsilon.boundaryField();

    double *h_boundary_yDist = new double[mesh_paras.num_boundary_surfaces];
    double *h_boundary_nut = new double[mesh_paras.num_boundary_surfaces];
    double *h_boundary_alphat = new double[mesh_paras.num_boundary_surfaces];
    double *h_boundary_k = new double[mesh_paras.num_boundary_surfaces];
    double *h_boundary_epsilon = new double[mesh_paras.num_boundary_surfaces];
    int offset = 0;
    forAll(bf, patchi)
    {
        const fvPatchScalarField& patchyDist = yDist[patchi];
        const fvPatchScalarField& patchyNut = nut.boundaryField()[patchi];
        const fvPatchScalarField& patchyAlphat = alphat.boundaryField()[patchi];
        const fvPatchScalarField& patchyk = k.boundaryField()[patchi];
        const fvPatchScalarField& patchyEps = epsilon.boundaryField()[patchi];
        int patchsize = patchyDist.size();
        if (patchyDist.type() == "processor"
            || patchyDist.type() == "processorCyclic") {
            memcpy(h_boundary_yDist + offset, &patchyDist[0], patchsize * sizeof(double));
            memcpy(h_boundary_nut + offset, &patchyNut[0], patchsize * sizeof(double));
            memcpy(h_boundary_alphat + offset, &patchyAlphat[0], patchsize * sizeof(double));
            memcpy(h_boundary_k + offset, &patchyk[0], patchsize * sizeof(double));
            memcpy(h_boundary_epsilon + offset, &patchyEps[0], patchsize * sizeof(double));

            scalarField patchyDistInternal = dynamic_cast<const processorFvPatchField<scalar>&>(patchyDist).patchInternalField()();
            scalarField patchNutInternal = dynamic_cast<const processorFvPatchField<scalar>&>(patchyNut).patchInternalField()();
            scalarField patchAlphatInternal = dynamic_cast<const processorFvPatchField<scalar>&>(patchyAlphat).patchInternalField()();
            scalarField patchKInternal = dynamic_cast<const processorFvPatchField<scalar>&>(patchyk).patchInternalField()();
            scalarField patchEpsInternal = dynamic_cast<const processorFvPatchField<scalar>&>(patchyEps).patchInternalField()();

            memcpy(h_boundary_yDist + offset + patchsize, &patchyDistInternal[0], patchsize * sizeof(double));
            memcpy(h_boundary_nut + offset + patchsize, &patchNutInternal[0], patchsize * sizeof(double));
            memcpy(h_boundary_alphat + offset + patchsize, &patchAlphatInternal[0], patchsize * sizeof(double));
            memcpy(h_boundary_k + offset + patchsize, &patchKInternal[0], patchsize * sizeof(double));
            memcpy(h_boundary_epsilon + offset + patchsize, &patchEpsInternal[0], patchsize * sizeof(double));

            offset += patchsize * 2;
        } else {
            memcpy(h_boundary_yDist + offset, &patchyDist[0], patchsize * sizeof(double));
            memcpy(h_boundary_nut + offset, &patchyNut[0], patchsize * sizeof(double));
            memcpy(h_boundary_alphat + offset, &patchyAlphat[0], patchsize * sizeof(double));
            memcpy(h_boundary_k + offset, &patchyk[0], patchsize * sizeof(double));
            memcpy(h_boundary_epsilon + offset, &patchyEps[0], patchsize * sizeof(double));
            offset += patchsize;
        }
    }

    label master = -1;
    forAll(bf, patchi)
    {
        if (isA<epsilonWallFunctionFvPatchScalarField>(bf[patchi]))
        {
            if (master == -1)
            {
                master = patchi;
            }
        }
    }

    const fvMesh& mesh = epsilon.mesh();

    volScalarField weights
    (
        IOobject
        (
            "weights",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            false // do not register
        ),
        mesh,
        dimensionedScalar(dimless, 0)
    );

    DynamicList<label> epsilonPatches(bf.size());
    forAll(bf, patchi)
    {
        if (isA<epsilonWallFunctionFvPatchScalarField>(bf[patchi]))
        {
            epsilonPatches.append(patchi);

            const labelUList& faceCells = bf[patchi].patch().faceCells();
            forAll(faceCells, i)
            {
                weights[faceCells[i]]++;
            }
        }
    }

    List<List<scalar>> cornerWeights_;
    cornerWeights_.setSize(bf.size());
    forAll(epsilonPatches, i)
    {
        label patchi = epsilonPatches[i];
        const fvPatchScalarField& wf = weights.boundaryField()[patchi];
        cornerWeights_[patchi] = 1.0/wf.patchInternalField();
    }

    double *h_boundary_cornerWeights = new double[mesh_paras.num_boundary_surfaces];
    offset = 0;
    forAll(bf, patchi)
    {
        int patchsize = bf[patchi].size();
        if (isA<epsilonWallFunctionFvPatchScalarField>(bf[patchi]))
        {
            List<double> patchycornerWeights = cornerWeights_[patchi];
            int patchsize = patchycornerWeights.size();
            memcpy(h_boundary_cornerWeights + offset, &patchycornerWeights[0], patchsize * sizeof(double));
            offset += patchsize;
        }
        else if (bf[patchi].type() == "processor"
            || bf[patchi].type() == "processorCyclic") {
            offset += patchsize * 2;
        } else {
            offset += patchsize;
        }
    }

    double E = 9.8;
    double kappa = 0.41; 
    double Cmu = 0.09;

    double *intensity = new double[mesh_paras.num_patches]();
    double *mixingLength = new double[mesh_paras.num_patches]();

    // NOTICE: only for case [SandiaD_LTS] - start
    std::cout << "only for case [SandiaD_LTS] boundary setting" << std::endl;
    intensity[3] = 0.0628; // inletPilot
    intensity[4] = 0.0471; // inletAir
    intensity[0] = 0.0458; // inletCH4

    mixingLength[3] = 0.000735;
    mixingLength[4] = 0.019677;
    mixingLength[0] = 0.000504;
    // NOTICE: only for case [SandiaD_LTS] - end

    init_turbulence(useTurbulence, useTCI, static_cast<LES_TURBULENCE_MODEL>(LESModelTypeInt), static_cast<RAS_TURBULENCE_MODEL>(RASModelTypeInt), 
                    patch_type_nut.data(), patch_type_alphat.data(), patch_type_k.data(), patch_type_epsilon.data(), &k[0], &epsilon[0],
                    h_boundary_nut, h_boundary_alphat, h_boundary_k, h_boundary_epsilon,
                    h_boundary_yDist, h_boundary_cornerWeights, intensity, mixingLength, E, kappa, Cmu);
}

void correctVelBCsHost(mesh_info_para &mesh_paras, volVectorField& U)
{
    copy2host_u();

    double *U_hostOrder = new double[mesh_paras.num_cells * 3]();
    permuteVector2Host(U_hostOrder, resultsData.h_u, mesh_paras.num_cells);
    memcpy(&U[0][0], U_hostOrder, mesh_paras.num_cells * 3 * sizeof(double));

    U.correctBoundaryConditions();

    double *h_boundary_u = new double[mesh_paras.num_boundary_surfaces*3]();
    int offset = 0;
    forAll(U.boundaryField(), patchi)
    {
        const fvPatchVectorField& patchU = U.boundaryField()[patchi];
        int patchsize = patchU.size();
        if (patchU.type() == "processor"
            || patchU.type() == "processorCyclic") {
            memcpy(h_boundary_u + 3*offset, &patchU[0][0], patchsize * 3 * sizeof(double));
            vectorField patchUInternal = 
                    dynamic_cast<const processorFvPatchField<vector>&>(patchU).patchInternalField()();
            memcpy(h_boundary_u + 3*offset + 3*patchsize, &patchUInternal[0][0], patchsize * 3 * sizeof(double));
            offset += patchsize * 2;
        } else {
            memcpy(h_boundary_u + 3*offset, &patchU[0][0], patchsize * 3 * sizeof(double));
            offset += patchsize;
        }
    }

    double *boundary_u_newOrder = new double[mesh_paras.num_boundary_surfaces*3]();
    permuteVector(boundary_u_newOrder, h_boundary_u, mesh_paras.num_boundary_surfaces);
    memcpy(resultsData.h_boundary_u, &boundary_u_newOrder[0], mesh_paras.num_boundary_surfaces * 3 * sizeof(double));

    copy2device_boundary_u();

    delete U_hostOrder;
    delete boundary_u_newOrder;
    delete h_boundary_u;
}