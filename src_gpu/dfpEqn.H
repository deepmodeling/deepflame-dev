#pragma once

#include "AmgXSolver.H"
#include <amgx_c.h>
#include "dfMatrixDataBase.H"
#include "dfMatrixOpBase.H"
#include "dfCSRSolver.H"

class dfpEqn
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
	AmgXSolver *pSolver = nullptr;
    CSRSolver *pCSRSolver = nullptr;
    int num_iteration = 0;

	// constant fields - internal

	// constant fields - boundary
	std::vector<int> patch_type_U;
    std::vector<int> patch_type_p;

	// non-constant fields - internal
	// intermediate fields
    double *d_rhorAUf = nullptr;
    double *d_phiHbyA = nullptr;
    double *d_flux = nullptr;

	// non-constant fields - boundary
    // intermediate boundary fields
    double *d_boundary_rhorAUf = nullptr;
    double *d_boundary_phiHbyA = nullptr;
    double *d_boundary_flux = nullptr;
    // boundary coeff fields
	double *d_value_internal_coeffs = nullptr;
	double *d_value_boundary_coeffs= nullptr;
	double *d_gradient_internal_coeffs= nullptr;
	double *d_gradient_boundary_coeffs= nullptr;

	// non-constant fields - ldu
    double *d_ldu = nullptr;
	double *d_lower = nullptr;
	double *d_upper = nullptr;
	double *d_diag = nullptr;
    double *d_extern = nullptr;
	double *d_source = nullptr;
	double *d_internal_coeffs = nullptr;
	double *d_boundary_coeffs = nullptr;

	// non-constant fields - csr
	double *d_A = nullptr;
    double *d_off_diag = nullptr;

    // field pointer map
    std::unordered_map<std::string, double*> fieldPointerMap;

public:
	// constructor
    dfpEqn(dfMatrixDataBase &dataBase)
        : dataBase_(dataBase) {}

	// destructor
	  ~dfpEqn(){}

	// member function

    // getter function
    double* getFieldPointer(const char* fieldAlias, location loc, position pos);

	// initialization
	void setConstantValues(const std::string &mode_string, const std::string &setting_path,
                        const dfMatrixDataBase& dataBase, GAMGStruct *GAMGdata_, int agglomeration_level); 
	void setConstantFields(const std::vector<int> patch_type_U, const std::vector<int> patch_type_p);
    void initNonConstantFields(const double *p, const double *boundary_p);
    void createNonConstantFieldsInternal();
	void createNonConstantFieldsBoundary();
	void createNonConstantLduAndCsrFields();
	// dfUEqn has no internal non-constant fields to be init
	//void initNonConstantFieldsInternal(xxx);

    void cleanCudaResources();

	// run equation
	void preProcess(double *h_phi, double *h_boundary_phi);
    void correctPsi(const double *h_thermoPsi, double *h_boundary_thermoPsi); // tmp
    void correctP(const double *h_p, double *h_boundary_p); // tmp
    //void getFlux();
	void process(GAMGStruct *GAMGdata, int agglomeration_level);
	void postProcess();

    // get intermediate variable
    void getrhorAUf(cudaStream_t stream, int num_cells, int num_surfaces,
        const int *lowerAddr, const int *upperAddr, 
        const double *weight, const double *vf1, const double *vf2, double *output, // end for internal
        int num_patches, const int *patch_size, const int *patch_type, const double *boundary_weight,
        const double *boundary_vf1, const double *boundary_vf2, double *boundary_output, double sign = 1.);
    void getphiHbyA(cudaStream_t stream, int num_cells, int num_surfaces, int num_boundary_surfaces, double rDeltaT, 
        const int *lowerAddr, const int *upperAddr, 
        const double *weight, const double *u_old, const double *rho_old, const double *phi_old, const double *rho, 
        const double *rhorAUf, const double *HbyA, const double *Sf, double *output, // end for internal
        int num_patches, const int *patch_size, const int *patch_type,
        const double *boundary_Sf, const double *boundary_velocity_old, const double *boundary_rho, 
        const double *boundary_rho_old, const double *boundary_phi_old, const double *boundary_rhorAUf, const double *boundary_HbyA,
        const double *boundary_weight, double *boundary_output, double sign = 1.);
    void correctionDiagMtxMultiTPsi(cudaStream_t stream, int num_cells, const double *psi, const double *thermo_psi, double *diag, double *source);

    void solve(GAMGStruct *GAMGdata, int agglomeration_level);
    void sync();

// #if defined DEBUG_
    void compareResult(const double *lower, const double *upper, const double *diag, const double *source, const double *internal_coeffs, const double *boundary_coeffs,  
        bool printFlag);
// #endif 
    void comparerhorAUf(const double *rhorAUf, const double *boundary_rhorAUf, bool printFlag);
    void comparephiHbyA(const double *phiHbyA, const double *boundary_phiHbyA, bool printFlag);
    void comparephi(const double *phi, const double *boundary_phi, bool printFlag);
    void comparephiFlux(const double *flux, const double *boundary_flux, bool printFlag);
    void comparep(const double *p, const double *boundary_p, bool printFlag);
    void compareU(const double *U, const double *boundary_U, bool printFlag);
    void comparedpdt(const double *dpdt, bool printFlag);

    void peqn_ldu_to_csr_no_diag(int row_, int num_surfaces, int *off_diag_rowptr_, double *lower, double *upper, int *lowerAddr, int *upperAddr, double *off_diag_value_);
};
