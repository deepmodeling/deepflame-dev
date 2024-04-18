#include "GenFvMatrix.H"
#include <filesystem>
#include <mpi.h>

#define GAS_CANSTANT 8314.46261815324
#define SQRT8 2.8284271247461903

size_t total_buffer_size;
Foam::scalar *total_buffer;

Foam::scalar *nasa_coeffs, *viscosity_coeffs, *conductivity_coeffs, *binary_diffusion_coeffs, *molecular_weights,
        *viscosity_constant1, *viscosity_constant2;

size_t nasa_coeffs_size, viscosity_coeffs_size, conductivity_coeffs_size, binary_diffusion_coeffs_size, molecular_weights_size,
    viscosity_constant1_size, viscosity_constant2_size;

Foam::label nSpecies = 0, nBoundarySurfaces = 0, nBoundaryPatches = 0;
Foam::label nCells = 0, nFaces = 0, nProcessBoundarySurfaces = 0;
Foam::label *surfacePerPatch;
Foam::label **faceCellsGroup;
Foam::label *nInterfacesGroup;
int *neighbProcNo;

std::string get_filename(std::string mechanism_file){
    size_t start = mechanism_file.find_last_of('/') + 1;
    size_t end = mechanism_file.find_last_of('.');
    return mechanism_file.substr(start, end - start);
}

void init_const_coeff_ptr(std::string mechanism_file, Foam::PtrList<Foam::volScalarField>& Y)
{
    int mpisize, mpirank;
    int flag_mpi_init;
    MPI_Initialized(&flag_mpi_init);

    if(flag_mpi_init){
        MPI_Comm_rank(MPI_COMM_WORLD,&mpirank);
        MPI_Comm_rank(MPI_COMM_WORLD,&mpisize);
    }

    nSpecies = Y.size();
    nCells = Y[0].size();
    const Foam::fvMesh& mesh = Y[0].mesh();
    const Foam::labelUList& neighbour = mesh.neighbour();
    nFaces = neighbour.size();

    forAll(Y[0].boundaryField(), patchi) {
        const Foam::fvPatchScalarField& patchYi = Y[0].boundaryField()[patchi];
        nBoundaryPatches ++;
        nBoundarySurfaces += patchYi.size();
        if (patchYi.type() == "processor" || patchYi.type() == "processorCyclic") 
            nProcessBoundarySurfaces += 2 * patchYi.size();
        else 
            nProcessBoundarySurfaces += patchYi.size();
    }
    neighbProcNo = new int[nBoundaryPatches];
    surfacePerPatch = new Foam::label[nBoundaryPatches];
    faceCellsGroup = new Foam::label*[nBoundaryPatches];
    nInterfacesGroup = new Foam::label[nBoundaryPatches];

    std::fill(neighbProcNo, neighbProcNo + nBoundaryPatches, -1);
    forAll(Y[0].boundaryField(), patchi) {
        const Foam::fvPatchScalarField& patchYi = Y[0].boundaryField()[patchi];
        surfacePerPatch[patchi] = patchYi.size();
        if (patchYi.type() == "processor" || patchYi.type() == "processorCyclic") {
            neighbProcNo[patchi] = dynamic_cast<const Foam::processorFvPatchField<Foam::scalar>&>(patchYi).neighbProcNo();
            nInterfacesGroup[patchi] = patchYi.size();
        } else {
            nInterfacesGroup[patchi] = 0;
        }
        
        faceCellsGroup[patchi] = new Foam::label[patchYi.size()];
        const Foam::labelUList& faceCells = patchYi.patch().faceCells();
        forAll(faceCells, facei) {
            faceCellsGroup[patchi][facei] = faceCells[facei];
        }
    }

}

namespace Foam
{
void correctThermo
(
    const PtrList<volScalarField>& Y,
    volScalarField& T,
    volScalarField& ha,
    volScalarField& rho,
    volScalarField& psi,
    volScalarField& alpha,
    volScalarField& mu,
    volScalarField& p,
    dfChemistryModel<basicThermo>* chemistry
)
{
    Info << "correctThermo start" << endl;
    Info << "nCells = " << nCells << endl;
    Info << "nSpecies = " << nSpecies << endl;
    Info << "nBoundarySurfaces = " << nBoundarySurfaces << endl;
    Info << "nBoundaryPatches = " << nBoundaryPatches << endl;

    // get ptr
    // get Y ptr
    scalar *h_Y = new scalar[nSpecies * nCells];
    scalar *h_boundary_Y = new scalar[nSpecies * nBoundarySurfaces];
    
    label offset = 0;
    forAll(Y, speciesI) {
        const volScalarField& Yi = Y[speciesI];
        memcpy(h_Y + speciesI * nCells, &Yi[0], nCells * sizeof(scalar));
        forAll(Yi.boundaryField(), patchi) {
            const fvPatchScalarField& patchYi = Yi.boundaryField()[patchi];
            label patchsize = patchYi.size();
            memcpy(h_boundary_Y + offset, &patchYi[0], patchsize*sizeof(scalar));
            offset += patchsize;
        }
    }

    // allocate memory
    scalar *mean_mole_weight = new scalar[nCells];
    scalar *mean_boundary_mole_weight = new scalar[nBoundarySurfaces];
    scalar *T_poly = new scalar[nCells * 5];
    scalar *boundary_T_poly = new scalar[nBoundarySurfaces * 5];
    scalar *mole_fraction = new scalar[nSpecies * nCells];
    scalar *boundary_mole_fraction = new scalar[nSpecies * nBoundarySurfaces];
    scalar *boundary_T = new scalar[nBoundarySurfaces];
    scalar *boundary_ha = new scalar[nBoundarySurfaces];
    scalar *boundary_psi = new scalar[nBoundarySurfaces];
    scalar *boundary_alpha = new scalar[nBoundarySurfaces];
    scalar *boundary_mu = new scalar[nBoundarySurfaces];
    scalar *boundary_rho = new scalar[nBoundarySurfaces];
    scalar *boundary_rhoD = new scalar[nBoundarySurfaces * nSpecies];
    scalar *boundary_p = new scalar[nBoundarySurfaces];
    scalar *h_rhoD = new scalar[nBoundarySurfaces];

    offset = 0;
    // construct boundary fields
    forAll(T.boundaryField(), patchi)
    {
        const fvPatchScalarField& patchT = T.boundaryField()[patchi];
        const fvPatchScalarField& patchHa = ha.boundaryField()[patchi];
        const fvPatchScalarField& patchRho = rho.boundaryField()[patchi];
        const fvPatchScalarField& patchP = p.boundaryField()[patchi];

        label patchsize = patchT.size();
        memcpy(boundary_T + offset, &patchT[0], patchsize * sizeof(scalar));
        memcpy(boundary_ha + offset, &patchHa[0], patchsize * sizeof(scalar));
        memcpy(boundary_rho + offset, &patchRho[0], patchsize * sizeof(scalar));
        memcpy(boundary_p + offset, &patchP[0], patchsize * sizeof(scalar));

        offset += patchsize;
    }
    
    // set mass fraction
    // scalar sum_mass, meanMoleWeight;
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (label i = 0; i < nCells; ++i) {
        scalar sum_mass = 0.;
        scalar meanMoleWeight = 0;
        for (label j = 0; j < nSpecies; ++j) {
            sum_mass += h_Y[j * nCells + i] / molecular_weights[j];
        }
        for (label j = 0; j < nSpecies; ++j) {
            mole_fraction[j * nCells + i] = h_Y[j * nCells + i] / (molecular_weights[j] * sum_mass);
            meanMoleWeight += mole_fraction[j * nCells + i] * molecular_weights[j];
        }
        mean_mole_weight[i] = meanMoleWeight;
    }
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (label i = 0; i < nBoundarySurfaces; ++i) {
        scalar sum_mass = 0.;
        scalar meanMoleWeight = 0;
        for (label j = 0; j < nSpecies; ++j) {
            sum_mass += h_boundary_Y[j * nBoundarySurfaces + i] / molecular_weights[j];
        }
        for (label j = 0; j < nSpecies; ++j) {
            boundary_mole_fraction[j * nBoundarySurfaces + i] = h_boundary_Y[j * nBoundarySurfaces + i] / (molecular_weights[j] * sum_mass);
            meanMoleWeight += boundary_mole_fraction[j * nBoundarySurfaces + i] * molecular_weights[j];
        }
        mean_boundary_mole_weight[i] = meanMoleWeight;
    }

    // calculate temperature
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (label i = 0; i < nCells; ++i) {
        // Newton's method
        scalar T_init = T[i];
        scalar h_target = ha[i];
        scalar h, cp, delta_T;
        for (label n = 0; n < 10; ++n) { // max_iter = 10
            // calculate h
            h = 0.;
            for (label j = 0; j < nSpecies; ++j) {
                if (T_init > nasa_coeffs[j * 15 + 0]) {
                    h += (nasa_coeffs[j * 15 + 1] + nasa_coeffs[j * 15 + 2] * T_init / 2 + nasa_coeffs[j * 15 + 3] * T_init * T_init / 3 + 
                            nasa_coeffs[j * 15 + 4] * T_init * T_init * T_init / 4 + nasa_coeffs[j * 15 + 5] * T_init * T_init * T_init * T_init / 5 + 
                            nasa_coeffs[j * 15 + 6] / T_init) * GAS_CANSTANT * T_init / molecular_weights[j] * h_Y[j * nCells + i];
                } else {
                    h += (nasa_coeffs[j * 15 + 8] + nasa_coeffs[j * 15 + 9] * T_init / 2 + nasa_coeffs[j * 15 + 10] * T_init * T_init / 3 + 
                            nasa_coeffs[j * 15 + 11] * T_init * T_init * T_init / 4 + nasa_coeffs[j * 15 + 12] * T_init * T_init * T_init * T_init / 5 + 
                            nasa_coeffs[j * 15 + 13] / T_init) * GAS_CANSTANT * T_init / molecular_weights[j] * h_Y[j * nCells + i];
                }
            }
            // calculate cp
            cp = 0.;
            for (label j = 0; j < nSpecies; j++) {
                if (T_init > nasa_coeffs[j * 15 + 0]) {
                    cp += h_Y[j * nCells + i] * (nasa_coeffs[j * 15 + 1] + nasa_coeffs[j * 15 + 2] * T_init + nasa_coeffs[j * 15 + 3] * T_init * T_init + 
                            nasa_coeffs[j * 15 + 4] * T_init * T_init * T_init + 
                            nasa_coeffs[j * 15 + 5] * T_init * T_init * T_init * T_init) * GAS_CANSTANT / molecular_weights[j];
                } else {
                    cp += h_Y[j * nCells + i] * (nasa_coeffs[j * 15 + 8] + nasa_coeffs[j * 15 + 9] * T_init + nasa_coeffs[j * 15 + 10] * T_init * T_init + 
                            nasa_coeffs[j * 15 + 11] * T_init * T_init * T_init + 
                            nasa_coeffs[j * 15 + 12] * T_init * T_init * T_init * T_init) * GAS_CANSTANT / molecular_weights[j];
                }
            }
            delta_T = (h - h_target) / cp;
            T_init -= delta_T;
            if (fabs(h - h_target) < 1e-7 || fabs(delta_T / T_init) < 1e-7)  break;
        }
        T[i] = T_init;
    }

    // compute T_poly
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (label i = 0; i < nCells; ++i) {
        T_poly[i * 5 + 0] = 1.;
        T_poly[i * 5 + 1] = std::log(T[i]);
        T_poly[i * 5 + 2] = T_poly[i * 5 + 1] * T_poly[i * 5 + 1];
        T_poly[i * 5 + 3] = T_poly[i * 5 + 2] * T_poly[i * 5 + 1];
        T_poly[i * 5 + 4] = T_poly[i * 5 + 2] * T_poly[i * 5 + 2];
    }

    // calculate Psi, rho
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (label i = 0; i < nCells; ++i) {
        psi[i] = mean_mole_weight[i] / (GAS_CANSTANT * T[i]);
        rho[i] = p[i] * psi[i];
    }

    // calculate viscosity
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (label i = 0; i < nCells; ++i) {
        // scalar *species_viscosity = new scalar[nSpecies];
        scalar species_viscosity[nSpecies];
        scalar sqrt_T = std::sqrt(T[i]);
        for (label j = 0; j < nSpecies; ++j) {
            scalar dot_product = 0.;
            for (label k = 0; k < 5; ++k) {
                dot_product += viscosity_coeffs[j * 5 + k] * T_poly[i * 5 + k];
            }
            species_viscosity[j] = dot_product;
        }
        scalar mu_mix = 0.;
        for (label j = 0; j < nSpecies; ++j) {
            scalar sum = 0.;
            for (label k = 0; k < nSpecies; ++k) {
                scalar temp = 1.0 + (species_viscosity[j] / species_viscosity[k]) *
                          viscosity_constant2[j * nSpecies + k];
                sum += mole_fraction[k * nCells + i] / SQRT8 * viscosity_constant1[j * nSpecies + k] * (temp * temp);
            }
            mu_mix += mole_fraction[j * nCells + i] * (species_viscosity[j] * species_viscosity[j]) / sum;
        }
        mu[i] = mu_mix * sqrt_T;
    }

    // calculate alpha
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (label i = 0; i < nCells; ++i) {
        // scalar *species_thermal_conductivities = new scalar[nSpecies];
        scalar species_thermal_conductivities[nSpecies];
        scalar dot_product;
        for (label j = 0; j < nSpecies; ++j) {
            dot_product = 0.;
            for (label k = 0; k < 5; ++k) {
                dot_product += conductivity_coeffs[j * 5 + k] * T_poly[i * 5 + k];
            }
            species_thermal_conductivities[j] = dot_product * sqrt(T[i]);
        }
        scalar sum_conductivity = 0.;
        scalar sum_inv_conductivity = 0.;
        for (label j = 0; j < nSpecies; ++j) {
            sum_conductivity += mole_fraction[j * nCells + i] * species_thermal_conductivities[j];
            sum_inv_conductivity += mole_fraction[j * nCells + i] / species_thermal_conductivities[j];
        }
        scalar lambda_mix = 0.5 * (sum_conductivity + 1.0 / sum_inv_conductivity);
        // calculate cp
        scalar cp = 0.;
        for (label j = 0; j < nSpecies; ++j) {
            if (T[i] > nasa_coeffs[j * 15 + 0]) {
                cp += h_Y[j * nCells + i] * (nasa_coeffs[j * 15 + 1] + nasa_coeffs[j * 15 + 2] * T[i] + nasa_coeffs[j * 15 + 3] * T[i] * T[i] + 
                        nasa_coeffs[j * 15 + 4] * T[i] * T[i] * T[i] + 
                        nasa_coeffs[j * 15 + 5] * T[i] * T[i] * T[i] * T[i]) * GAS_CANSTANT / molecular_weights[j];
            } else {
                cp += h_Y[j * nCells + i] * (nasa_coeffs[j * 15 + 8] + nasa_coeffs[j * 15 + 9] * T[i] + nasa_coeffs[j * 15 + 10] * T[i] * T[i] + 
                        nasa_coeffs[j * 15 + 11] * T[i] * T[i] * T[i] + 
                        nasa_coeffs[j * 15 + 12] * T[i] * T[i] * T[i] * T[i]) * GAS_CANSTANT / molecular_weights[j];
            }
        }
        alpha[i] = lambda_mix / cp;
    }

    // calculate rhoD
#ifdef MIXED
    for (label i = 0; i < nCells; ++i) {
        scalar powT = T[i] * sqrt(T[i]);
        scalar local_mean_mole_weight = mean_mole_weight[i];
        scalar local_rho_div_p = rho[i] / p[i];
        for (label j = 0; j < nSpecies; ++j) {
            if (mole_fraction[i * nCells + j] + 1e-10 > 1.) {
                rhoD[i * nSpecies + j] = 0.;
                continue;
            }
            scalar sum1 = 0.;
            scalar sum2 = 0.;
            for (label k = 0; k < nSpecies; ++k) {
                if (j == k) continue;
                scalar tmp = 0.;
                for (label l = 0; l < 5; ++l) {
                    tmp += binary_diffusion_coeffs[j * nSpecies * 5 + k * 5 + l] * T_poly[i * 5 + l];
                }
                scalar local_D = tmp * powT;

                sum1 += mole_fraction[i * nCells + k] / local_D;
                sum2 += mole_fraction[i * nCells + k] * molecular_weights[k] / local_D;
            }
            sum2 *= mole_fraction[i * nCells + j] / (local_mean_mole_weight - mole_fraction[i * nCells + j] * molecular_weights[j]);
            rhoD[i * nSpecies + j] = local_rho_div_p * (1. / (sum1 + sum2));
        }
    }
#else
    for (label j = 0; j < nSpecies; ++j) {
        volScalarField& rhoD = chemistry->rhoD(j);
        memcpy(&rhoD[0], &alpha[0], nCells * sizeof(scalar));
    }
#endif

    // boundary conditions
    // TODO: not support fixedValue BC for now
    // calculate boundary temperature
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (label i = 0; i < nBoundarySurfaces; ++i) {
        // Newton's method
        scalar T_init = boundary_T[i];
        scalar h_target = boundary_ha[i];
        scalar h, cp, delta_T;
        for (label n = 0; n < 10; ++n) { // max_iter = 10
            // calculate h
            h = 0.;
            for (label j = 0; j < nSpecies; ++j) {
                if (T_init > nasa_coeffs[j * 15 + 0]) {
                    h += (nasa_coeffs[j * 15 + 1] + nasa_coeffs[j * 15 + 2] * T_init / 2 + nasa_coeffs[j * 15 + 3] * T_init * T_init / 3 + 
                            nasa_coeffs[j * 15 + 4] * T_init * T_init * T_init / 4 + nasa_coeffs[j * 15 + 5] * T_init * T_init * T_init * T_init / 5 + 
                            nasa_coeffs[j * 15 + 6] / T_init) * GAS_CANSTANT * T_init / molecular_weights[j] * h_boundary_Y[j * nBoundarySurfaces + i];
                } else {
                    h += (nasa_coeffs[j * 15 + 8] + nasa_coeffs[j * 15 + 9] * T_init / 2 + nasa_coeffs[j * 15 + 10] * T_init * T_init / 3 + 
                            nasa_coeffs[j * 15 + 11] * T_init * T_init * T_init / 4 + nasa_coeffs[j * 15 + 12] * T_init * T_init * T_init * T_init / 5 + 
                            nasa_coeffs[j * 15 + 13] / T_init) * GAS_CANSTANT * T_init / molecular_weights[j] * h_boundary_Y[j * nBoundarySurfaces + i];
                }
            }
            // calculate cp
            cp = 0.;
            for (label j = 0; j < nSpecies; ++j) {
                if (T_init > nasa_coeffs[j * 15 + 0]) {
                    cp += h_boundary_Y[j * nBoundarySurfaces + i] * (nasa_coeffs[j * 15 + 1] + nasa_coeffs[j * 15 + 2] * T_init + nasa_coeffs[j * 15 + 3] * T_init * T_init + 
                            nasa_coeffs[j * 15 + 4] * T_init * T_init * T_init + 
                            nasa_coeffs[j * 15 + 5] * T_init * T_init * T_init * T_init) * GAS_CANSTANT / molecular_weights[j];
                } else {
                    cp += h_boundary_Y[j * nBoundarySurfaces + i] * (nasa_coeffs[j * 15 + 8] + nasa_coeffs[j * 15 + 9] * T_init + nasa_coeffs[j * 15 + 10] * T_init * T_init + 
                            nasa_coeffs[j * 15 + 11] * T_init * T_init * T_init + 
                            nasa_coeffs[j * 15 + 12] * T_init * T_init * T_init * T_init) * GAS_CANSTANT / molecular_weights[j];
                }
            }
            delta_T = (h - h_target) / cp;
            T_init -= delta_T;
            if (fabs(h - h_target) < 1e-7 || fabs(delta_T / T_init) < 1e-7)  break;
        }
        boundary_T[i] = T_init;
    }

    // calculate boundary_T_poly
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (label i = 0; i < nBoundarySurfaces; ++i) {
        boundary_T_poly[i * 5 + 0] = 1.;
        boundary_T_poly[i * 5 + 1] = log(boundary_T[i]);
        boundary_T_poly[i * 5 + 2] = boundary_T_poly[i * 5 + 1] * boundary_T_poly[i * 5 + 1];
        boundary_T_poly[i * 5 + 3] = boundary_T_poly[i * 5 + 2] * boundary_T_poly[i * 5 + 1];
        boundary_T_poly[i * 5 + 4] = boundary_T_poly[i * 5 + 2] * boundary_T_poly[i * 5 + 2];
    }

    // calculate boundary psi, rho
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (label i = 0; i < nBoundarySurfaces; ++i) {
        boundary_psi[i] = mean_boundary_mole_weight[i] / (GAS_CANSTANT * boundary_T[i]);
        boundary_rho[i] = boundary_p[i] * boundary_psi[i];
    }

    // calculate boundary viscosity
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (label i = 0; i < nBoundarySurfaces; ++i) {
        // scalar *species_viscosity = new scalar[nSpecies];
        scalar species_viscosity[nSpecies];
        scalar sqrt_T = std::sqrt(boundary_T[i]);
        for (label j = 0; j < nSpecies; ++j) {
            scalar dot_product = 0.;
            for (label k = 0; k < 5; ++k) {
                dot_product += viscosity_coeffs[j * 5 + k] * boundary_T_poly[i * 5 + k];
            }
            species_viscosity[j] = dot_product;
        }
        scalar mu_mix = 0.;
        for (label j = 0; j < nSpecies; ++j) {
            scalar sum = 0.;
            for (label k = 0; k < nSpecies; ++k) {
                scalar temp = 1.0 + (species_viscosity[j] / species_viscosity[k]) *
                          viscosity_constant2[j * nSpecies + k];
                sum += boundary_mole_fraction[k * nBoundarySurfaces + i] / SQRT8 * viscosity_constant1[j * nSpecies + k] * (temp * temp);
            }
            mu_mix += boundary_mole_fraction[j * nBoundarySurfaces + i] * (species_viscosity[j] * species_viscosity[j]) / sum;
        }
        boundary_mu[i] = mu_mix * sqrt_T;
    }

    // calculate boundary alpha
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (label i = 0; i < nBoundarySurfaces; ++i) {
        // scalar *species_thermal_conductivities = new scalar[nSpecies];
        scalar species_thermal_conductivities[nSpecies];
        scalar dot_product;
        for (label j = 0; j < nSpecies; ++j) {
            dot_product = 0.;
            for (label k = 0; k < 5; ++k) {
                dot_product += conductivity_coeffs[j * 5 + k] * boundary_T_poly[i * 5 + k];
            }
            species_thermal_conductivities[j] = dot_product * sqrt(boundary_T[i]);
        }
        scalar sum_conductivity = 0.;
        scalar sum_inv_conductivity = 0.;
        for (label j = 0; j < nSpecies; ++j) {
            sum_conductivity += boundary_mole_fraction[j * nBoundarySurfaces + i] * species_thermal_conductivities[j];
            sum_inv_conductivity += boundary_mole_fraction[j * nBoundarySurfaces + i] / species_thermal_conductivities[j];
        }
        scalar lambda_mix = 0.5 * (sum_conductivity + 1.0 / sum_inv_conductivity);
        // calculate cp
        scalar cp = 0.;
        for (label j = 0; j < nSpecies; ++j) {
            if (boundary_T[i] > nasa_coeffs[j * 15 + 0]) {
                cp += h_boundary_Y[j * nBoundarySurfaces + i] * (nasa_coeffs[j * 15 + 1] + nasa_coeffs[j * 15 + 2] * boundary_T[i] + nasa_coeffs[j * 15 + 3] * boundary_T[i] * boundary_T[i] + 
                        nasa_coeffs[j * 15 + 4] * boundary_T[i] * boundary_T[i] * boundary_T[i] + 
                        nasa_coeffs[j * 15 + 5] * boundary_T[i] * boundary_T[i] * boundary_T[i] * boundary_T[i]) * GAS_CANSTANT / molecular_weights[j];
            } else {
                cp += h_boundary_Y[j * nBoundarySurfaces + i] * (nasa_coeffs[j * 15 + 8] + nasa_coeffs[j * 15 + 9] * boundary_T[i] + nasa_coeffs[j * 15 + 10] * boundary_T[i] * boundary_T[i] + 
                        nasa_coeffs[j * 15 + 11] * boundary_T[i] * boundary_T[i] * boundary_T[i] + 
                        nasa_coeffs[j * 15 + 12] * boundary_T[i] * boundary_T[i] * boundary_T[i] * boundary_T[i]) * GAS_CANSTANT / molecular_weights[j];
            }
        }
        boundary_alpha[i] = lambda_mix / cp;
    }

    // calculate rhoD
    // TODO: only support UnityLewis for now
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (label i = 0; i < nSpecies; ++i) {
        for (label j = 0; j < nBoundarySurfaces; ++j) {
            boundary_rhoD[i * nBoundarySurfaces + j] = boundary_alpha[j];
        }
    }

    // copy fields back to OpenFOAM
    offset = 0;
    forAll(T.boundaryField(), patchi)
    {       
        fvPatchScalarField& patchT = const_cast<fvPatchScalarField&>(T.boundaryField()[patchi]);
        fvPatchScalarField& patchHa = const_cast<fvPatchScalarField&>(ha.boundaryField()[patchi]);
        fvPatchScalarField& patchRho = const_cast<fvPatchScalarField&>(rho.boundaryField()[patchi]);
        fvPatchScalarField& patchPsi = const_cast<fvPatchScalarField&>(psi.boundaryField()[patchi]);
        fvPatchScalarField& patchAlpha = const_cast<fvPatchScalarField&>(alpha.boundaryField()[patchi]);
        fvPatchScalarField& patchMu = const_cast<fvPatchScalarField&>(mu.boundaryField()[patchi]);

        label patchSize = patchT.size();

        memcpy(&patchT[0], boundary_T + offset, patchSize * sizeof(scalar));
        memcpy(&patchHa[0], boundary_ha + offset, patchSize * sizeof(scalar));
        memcpy(&patchRho[0], boundary_rho + offset, patchSize * sizeof(scalar));
        memcpy(&patchPsi[0], boundary_psi + offset, patchSize * sizeof(scalar));
        memcpy(&patchAlpha[0], boundary_alpha + offset, patchSize * sizeof(scalar));
        memcpy(&patchMu[0], boundary_mu + offset, patchSize * sizeof(scalar));

        for (label j = 0; j < nSpecies; ++j) {
            fvPatchScalarField& patchRhoD = const_cast<fvPatchScalarField&>(chemistry->rhoD(j).boundaryField()[patchi]);
            memcpy(&patchRhoD[0], boundary_rhoD + offset + j * nBoundarySurfaces, patchSize * sizeof(scalar));
        }
        offset += patchSize;
    }

    delete [] h_Y;
    delete [] h_boundary_Y;
    delete [] mean_mole_weight;
    delete [] mean_boundary_mole_weight;
    delete [] T_poly;
    delete [] boundary_T_poly;
    delete [] mole_fraction;
    delete [] boundary_mole_fraction;
    delete [] boundary_T;
    delete [] boundary_ha;
    delete [] boundary_psi;
    delete [] boundary_alpha;
    delete [] boundary_mu;
    delete [] boundary_rho;
    delete [] boundary_rhoD;
    delete [] boundary_p;
    delete [] h_rhoD;
    Info << "correctThermo end" << endl;
}

} // End namespace Foam
