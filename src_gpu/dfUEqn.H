#pragma once

#include "AmgXSolver.H"
#include <amgx_c.h>
#include "dfMatrixDataBase.H"
#include "dfMatrixOpBase.H"
#include "dfCSRSolver.H"

class dfUEqn
{
private:
	dfMatrixDataBase &dataBase_;

    // cuda resource
    cudaStream_t stream;
#ifdef USE_GRAPH
    // one graph for one eqn before using self-developed solver
    cudaGraph_t graph_pre, graph_post;
    cudaGraphExec_t graph_instance_pre, graph_instance_post;
    bool pre_graph_created=false;
    bool post_graph_created=false;
#endif

	// constant values -- basic
	std::string mode_string;
	std::string setting_path;

	// constant values -- amgx solvers
	AmgXSolver *UxSolver = nullptr;
	AmgXSolver *UySolver = nullptr;
	AmgXSolver *UzSolver = nullptr;

    CSRSolver *USolver = nullptr;
    int num_iteration = 0;

	// constant fields - internal
	// 无

	// constant fields - boundary
	std::vector<int> patch_type;

	// non-constant fields - internal
	// thermophysical fields
	double *d_nu_eff = nullptr;
	// computed on CPU, used on GPU, need memcpyh2d - host
	double *h_nu_eff = nullptr;
    double *h_A_pEqn = nullptr;
    double *h_H_pEqn = nullptr;
	// intermediate fields
	double *d_grad_u = nullptr;
    double *d_delta = nullptr;

	double *d_rho_nueff = nullptr;
	double *d_u_host_order = nullptr;
    double *d_fvc_output = nullptr; // TODO: no need anymore

	// non-constant fields - boundary
	// thermophysical fields
	double *d_boundary_nu_eff = nullptr;
	// computed on CPU, used on GPU, need memcpyh2d - host
	double *h_boundary_nu_eff = nullptr;
	// intermediate fields
	double *d_boundary_grad_u = nullptr;
	double *d_boundary_rho_nueff = nullptr;
    // boundary coeff fields
	double *d_value_internal_coeffs = nullptr;
	double *d_value_boundary_coeffs= nullptr;
	double *d_gradient_internal_coeffs= nullptr;
	double *d_gradient_boundary_coeffs= nullptr;
    // intermediate fields
    double *d_boundary_u_host_order = nullptr;

	// non-constant fields - ldu
    double *d_ldu = nullptr;
	double *d_lower = nullptr;
	double *d_upper = nullptr;
	double *d_diag = nullptr;
    double *d_extern = nullptr;
	double *d_source = nullptr;
	double *d_internal_coeffs = nullptr;
	double *d_boundary_coeffs = nullptr;
    double *d_diag_vector = nullptr;
    double *d_A_pEqn = nullptr;
    double *d_H_pEqn = nullptr;
    double *d_H_pEqn_perm = nullptr;

	// non-constant fields - csr
	double *d_A = nullptr;
	double *d_b = nullptr; // TODO: needless
    double *d_ldu_solve = nullptr;
    double *d_extern_solve = nullptr;
	double *d_source_solve = nullptr;
	double *d_internal_coeffs_solve = nullptr;
	double *d_boundary_coeffs_solve = nullptr;

    // field pointer map
    std::unordered_map<std::string, double*> fieldPointerMap;

public:
	// constructor
    dfUEqn(dfMatrixDataBase &dataBase)
        : dataBase_(dataBase) {}

	// destructor
	~dfUEqn(){}

	// member function

    // getter function
    double* getFieldPointer(const char* fieldAlias, location loc, position pos);

	// initialization
	void setConstantValues(const std::string &mode_string, const std::string &setting_path); 
	void setConstantFields(const std::vector<int> patch_type);
	void createNonConstantFieldsInternal();
	void createNonConstantFieldsBoundary();
	void createNonConstantLduAndCsrFields();
	// dfUEqn has no internal non-constant fields to be init
	void initNonConstantFieldsInternal(const double *u, const double *boundary_u);
	void initNonConstantFieldsBoundary();

    void cleanCudaResources();

	// run equation
    void preProcessForRhoEqn(const double *h_rho, const double *h_phi, const double *h_boundary_phi);
	void preProcess(const double *h_u, const double *h_boundary_u, const double *h_p, const double *h_boundary_p, const double *h_nu_eff, const double *h_boundary_nu_eff);
	void process();
	void postProcess();

    void A(double *Psi);
    void H(double *Psi);
    void getrAU(cudaStream_t stream, ncclComm_t comm, int num_cells, int num_surfaces, int num_boundary_surfaces, 
        const int *neighbor_peer, int num_patches, const int *patch_size, const int *patch_type,
        const int *boundary_cell_face, const double *boundary_delta_coeffs, const double *internal_coeffs, const double *volume, 
        const double *diag, double *rAU, double *boundary_rAU);
    void UEqnGetHbyA(cudaStream_t stream, ncclComm_t comm, const int *neighbor_peer, 
        int num_cells, int num_surfaces, int num_boundary_surfaces, 
        const int *lowerAddr, const int *upperAddr, const double *volume, const double *u,
        int num_patches, const int *patch_size, const int *patch_type, const int *patch_type_U,
        const int *boundary_cell_face, const double *internal_coffs, const double *boundary_coeffs, const double *boundary_weight,
        const double *lower, const double *upper, const double *source, const double *psi, 
        const double *rAU, const double *boundary_rAU, const double *boundary_u,
        const int *cyclicNeighbor, const int *patchSizeOffset,
        double *HbyA, double *boundary_HbyA);
    void getHbyA();
    void getTurbulenceKEpsilon_Smagorinsky(cudaStream_t stream, int num_cells, int num_boundary_surfaces, 
        const double *grad_U_tsr, const double *volume, double *delta, double *turbulence_k, double *turbulence_epsilon);
    void correctPsi(double *Psi, double *boundary_psi);
    void ueqn_ldu_to_csr(cudaStream_t stream, int num_cells, int num_surfaces, int num_boundary_surface, int num_Nz, 
        const int* boundary_cell_face, const int *ldu_to_csr_index, const int *diag_to_csr_index,
        int num_patches, const int *patch_size, const int *patch_type, const double *vf, const double *boundary_vf, 
        const double *ldu, double *external, const double *source, const double *internal_coeffs, const double *boundary_coeffs,
        const int *cyclicNeighbor, const int *patchSizeOffset, double *A, double *b);

    void solve();
    void sync();

// #if defined DEBUG_
    void compareResult(const double *lower, const double *upper, const double *diag, const double *source, const double *internal_coeffs, const double *boundary_coeffs, 
    // const double *tmpVal, const double *boundary_val,
    bool printFlag);
// #endif 
    void compareHbyA(const double *HbyA, const double *boundary_HbyA, bool printFlag);
    void comparerAU(const double *rAU, const double *boundary_rAU, bool printFlag);
    void compareU(const double *U, const double *boundary_U, bool printFlag);
};
