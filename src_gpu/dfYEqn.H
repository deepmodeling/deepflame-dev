#pragma once

#include "AmgXSolver.H"
#include <amgx_c.h>
#include "dfMatrixDataBase.H"
#include "dfMatrixOpBase.H"
#include "dfChemistrySolver.H"

class dfYEqn
{
private:
	dfMatrixDataBase &dataBase_;
    dfChemistrySolver &chemistrySolver_;

    // cuda resource
    cudaStream_t stream;
#ifdef USE_GRAPH
    // one graph for one eqn before using self-developed solver
    cudaGraph_t graph;
    cudaGraphExec_t graph_instance;
    bool graph_created=false;
#endif

	// constant values -- basic
	std::string mode_string;
	std::string setting_path;
    int inertIndex;

	// constant values -- amgx solvers
    std::vector<AmgXSolver *> YSolverSet;
    int num_iteration = 0;

	// constant fields - internal
	// 无

	// constant fields - boundary
	std::vector<int> patch_type;

    // const fields - lewis number
    double *d_lewis_number = nullptr;
    std::vector<double> lewis_number;

	// non-constant fields - internal
	// thermophysical fields
    double *d_hai = nullptr;
    double *d_mut_sct = nullptr;
    // intermediate fields
    double *d_grad_y = nullptr;
    double *d_sumY_diff_error = nullptr;
    double *d_phiUc = nullptr;
    double *d_DEff = nullptr;
    double *d_permute = nullptr;
    // combustion fields
    double *d_RR = nullptr;
    // computed on CPU, used on GPU, need memcpyh2d
	double *h_hai = nullptr;
	double *h_rhoD = nullptr;
	double *h_mut_sct = nullptr;

	// non-constant fields - boundary
	// thermophysical fields
    double *d_boundary_hai = nullptr;
    double *d_boundary_mut_sct = nullptr;
    // intermediate fields
    double *d_boundary_grad_y = nullptr;
    double *d_boundary_sumY_diff_error = nullptr;
    double *d_boundary_phiUc = nullptr;
    double *d_boundary_DEff = nullptr;
    double *d_boundary_permute = nullptr;
	// computed on CPU, used on GPU, need memcpyh2d - host
	double *h_boundary_hai = nullptr;
	double *h_boundary_rhoD = nullptr;
	double *h_boundary_mut_sct = nullptr;
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
	double *d_b = nullptr;

    // field pointer map
    std::unordered_map<std::string, double*> fieldPointerMap;

public:
	// 构造函数
    dfYEqn(dfMatrixDataBase &dataBase, dfChemistrySolver &chemistrySolver)
        : dataBase_(dataBase), chemistrySolver_(chemistrySolver) {}

	// 析构函数
	~dfYEqn(){}

	// 成员函数

    // getter函数
    double* getFieldPointer(const char* fieldAlias, location loc, position pos);

	// 初始化构建
	void setConstantValues(const std::string &mode_string, const std::string &setting_path, const int inertIndex);
	void setConstantFields(const std::vector<int> patch_type, const std::vector<double> lewis_number);
	void createNonConstantFieldsInternal();
	void createNonConstantFieldsBoundary();
	void createNonConstantLduAndCsrFields();
	void initNonConstantFieldsInternal(const double *y);
	void initNonConstantFieldsBoundary(const double *boundary_y);

    void cleanCudaResources();

	// 方程运行
    void preProcess(const double *h_rhoD, const double *h_boundary_rhoD,
        const double *h_hai, const double *h_boundary_hai,
        const double *h_mut_sct, const double *h_boundary_mut_sct);
	void process();
    void postProcess(double *h_y, double *h_boundary_y);
    void solve(int speciesIndex);
    void sync();

    // 方程特化版离散函数
    void yeqn_compute_thermo_alpha(cudaStream_t stream,
            int num_cells, const double *rhoD, double *thermo_alpha,
            int num_boundary_surfaces, const double *boundary_rhoD, double *boundary_thermo_alpha);
    void yeqn_compute_DEff_via_lewisNumber(cudaStream_t stream, int num_species, int num_cells, int num_boundary_surfaces, 
            double *lewis_number, const double *alpha, const double *mut_sct, double *DEff,
            const double *boundary_alpha, const double *boundary_mut_sct, double *boundary_DEff);
    void yeqn_compute_RR(dfChemistrySolver& chemistrySolver, cudaStream_t stream, const double *h_T, const double *d_T, 
            const double *p, const double *y, const double *rho, double *RR);
    void yeqn_fvc_laplacian_scalar(cudaStream_t stream, ncclComm_t comm, const int *neighbor_peer,
            int num_species, int num_cells, int num_surfaces, int num_boundary_surfaces,
            const int *lowerAddr, const int *upperAddr,
            const double *weight, const double *mag_sf, const double *delta_coeffs, const double *volume,
            const double *thermo_alpha, const double *hai, const double *vf, double *output, // end for internal
            int num_patches, const int *patch_size, const int *patch_type, const int *boundary_cell_face,
            const double *boundary_weight, const double *boundary_mag_sf, const double *boundary_delta_coeffs,
            const double *boundary_thermo_alpha, const double *boundary_hai, const double *boundary_vf,
            const int *cyclicNeighbor, const int *patchSizeOffset, double *boundary_output);
    void yeqn_compute_sumYDiffError_and_hDiffCorrFlux(cudaStream_t stream, int num_species, int num_cells, int num_boundary_surfaces,
            const double *rhoD, const double *hai, const double *y, const double *grad_y,
            double *sumY_diff_error, double *hDiff_corr_flux,
            const double *boundary_hai, const double *boundary_y, const double *boundary_grad_y, const double *boundary_rhoD,
            double *boundary_sumY_diff_error, double *boundary_hDiff_corr_flux);
    void yeqn_compute_phiUc(cudaStream_t stream, int num_cells, int num_surfaces, int num_boundary_surfaces,
            const int *lowerAddr, const int *upperAddr,
            const double *weight, const double *sf, const double *sumY_diff_error, double *phiUc,
            const double *boundary_sf, const double *boundary_sumY_diff_error, double *boundary_phiUc);
    void yeqn_compute_y_inertIndex(cudaStream_t stream, int num_species, int inertIndex, int num_cells, double *y);

// #if defined DEBUG_
    void comparediffAlphaD(const double *diffAlphaD, const double *boundary_diffAlphaD, bool printFlag);
    void comparegradyi(const double *grad_yi, const double *boundary_grad_yi, int specie_index, bool printFlag);
    void comparesumYDiffError(const double *sumYDiffError, const double *boundary_sumYDiffError, bool printFlag);
    void comparehDiffCorrFlux(const double *hDiffCorrFlux, const double *boundary_hDiffCorrFlux, bool printFlag);
    void comparephiUc(const double *phiUc, const double *boundary_phiUc, bool printFlag);
    void compareResult(const double *lower, const double *upper, const double *diag, const double *source,
            const double *internal_coeffs, const double *boundary_coeffs, bool printFlag);
    void compareYi(const double *yi, int specie_index, bool printFlag);
// #endif
};
