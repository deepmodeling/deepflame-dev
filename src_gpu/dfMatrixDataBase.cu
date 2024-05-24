#include "dfMatrixDataBase.H"
#include "dfMatrixOpBase.H"
#include "dfNcclBase.H"

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
        {"processorCyclic", processorCyclic}
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
    }
}

dfMatrixDataBase::dfMatrixDataBase() {}

dfMatrixDataBase::~dfMatrixDataBase() {}

void dfMatrixDataBase::setCommInfo(MPI_Comm mpi_comm, ncclComm_t nccl_comm, ncclUniqueId nccl_id,
        int nRanks, int myRank, int localRank, std::vector<int> &neighbProcNo) {
    this->mpi_comm = mpi_comm;
    this->nccl_comm = nccl_comm;
    this->nccl_id = nccl_id;
    this->nRanks = nRanks;
    this->myRank = myRank;
    this->localRank = localRank;
    this->neighbProcNo = neighbProcNo;
}
 
void dfMatrixDataBase::prepareCudaResources() {
    checkCudaErrors(cudaStreamCreate(&stream));
}

void dfMatrixDataBase::cleanCudaResources() {
    // destroy cuda resources
    checkCudaErrors(cudaStreamDestroy(stream));
    //ncclDestroy(nccl_comm);
    // TODO: free pointers
}

void dfMatrixDataBase::setConstantValues(int num_cells, int num_total_cells, int num_surfaces, 
        int num_boundary_surfaces, int num_patches, int num_proc_surfaces, 
        std::vector<int> patch_size, int num_species, double rdelta_t) {
    // constant values -- basic
    this->num_cells = num_cells;
    this->num_total_cells = num_total_cells;
    this->num_surfaces = num_surfaces;
    this->num_boundary_surfaces = num_boundary_surfaces;
    this->num_patches = num_patches;
    this->num_proc_surfaces = num_proc_surfaces;
    this->patch_size = patch_size;
    this->num_species = num_species;
    this->rdelta_t = rdelta_t;
    this->num_Nz = num_cells + 2 * num_surfaces + num_proc_surfaces;
    this->interfaceFlag = new int[num_patches]();

    // constant values -- ldu bytesize
    cell_value_bytes = num_cells * sizeof(double);
    cell_value_vec_bytes = num_cells * 3 * sizeof(double);
    cell_value_tsr_bytes = num_cells * 9 * sizeof(double);
    cell_index_bytes = num_cells * sizeof(int);
    surface_value_bytes = num_surfaces * sizeof(double);
    surface_index_bytes = num_surfaces * sizeof(int);
    surface_value_vec_bytes = num_surfaces * 3 * sizeof(double);
    boundary_surface_value_bytes = num_boundary_surfaces * sizeof(double);
    boundary_surface_value_vec_bytes = num_boundary_surfaces * 3 * sizeof(double);
    boundary_surface_value_tsr_bytes = num_boundary_surfaces * 9 * sizeof(double);
    boundary_surface_index_bytes = num_boundary_surfaces * sizeof(int);

    // constant values -- csr bytesize
    csr_row_index_bytes = (num_cells + 1) * sizeof(int);
    csr_col_index_bytes = num_Nz * sizeof(int);
    csr_col_index_bytes_no_diag = 2 * num_surfaces * sizeof(int);
    csr_value_bytes = num_Nz * sizeof(double);
    csr_value_vec_bytes = num_Nz * 3 * sizeof(double);
}

void dfMatrixDataBase::setAmgxSolvers(const std::string &mode_string, const std::string &u_setting_path, const std::string &p_setting_path) {
    // amgx solvers
    u_setting_solver = new AmgXSolver(mode_string, u_setting_path, localRank);
    p_setting_solver = new AmgXSolver(mode_string, p_setting_path, localRank);
}

void dfMatrixDataBase::resetAmgxSolvers() {
    if (u_setting_solver) {
        delete u_setting_solver;
        u_setting_solver = nullptr;
    }
    if (p_setting_solver) {
        delete p_setting_solver;
        p_setting_solver = nullptr;
    }
}
    
void dfMatrixDataBase::solve(int num_iteration, AMGXSetting setting, double *d_A, double *d_x, double *d_b) {
    AmgXSolver *solver = (setting == AMGXSetting::u_setting) ? u_setting_solver : p_setting_solver;
    if (num_iteration == 0)                                     // first interation
    {
        solver->setOperator(num_cells, num_total_cells, num_Nz, d_csr_row_index, d_csr_col_index, d_A);
    }
    else
    {
        solver->updateOperator(num_cells, num_Nz, d_A);
    }
    solver->solve(num_cells, d_x, d_b);
}

void dfMatrixDataBase::setCyclicInfo(std::vector<int> &cyclicNeighbor)
{
    this->cyclicNeighbor = cyclicNeighbor;
}

void dfMatrixDataBase::setConstantIndexes_GAMG(int nCells, int nFaces, int *owner, int *neighbor, int leveli, int type){
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
    memset(h_ldu2ellIndex, 0, sizeof(int) * nFaces * 2);
    std::vector<int> rowOffset(nCells, 0);
    // lower
    for(int i = 0; i < nFaces; i++){
        int row = neighbor[i];
        int index = h_ell_row_maxcount[leveli] * row + rowOffset[row];
        h_ldu2ellIndex[leveli][i] = index;
        h_ellCols[leveli][index] = owner[i];
        rowOffset[row] += 1;
    }
    // upper
    for(int i = 0; i < num_surfaces; i++){
        int row = owner[i];
        int index = ell_row_maxcount * row + rowOffset[row];
        h_ldu2ellIndex[leveli][i + num_surfaces] = index;
        h_ellCols[leveli][index] = neighbor[i];
        rowOffset[row] += 1;
    }

    free(ellColsperRow);
}
}

void dfMatrixDataBase::setConstantIndexes(const int *owner, const int *neighbor, const int *procRows, 
        const int *procCols, int globalOffset) {
    // build d_owner, d_neighbor
    checkCudaErrors(cudaMalloc((void**)&d_owner, surface_index_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_neighbor, surface_index_bytes));
    checkCudaErrors(cudaMemcpyAsync(d_owner, owner, surface_index_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_neighbor, neighbor, surface_index_bytes, cudaMemcpyHostToDevice, stream));
    DEBUG_TRACE;

    /**
     * @brief ldu2csr with diagonal
     * 
     */
    // build permTmp, rowIndicesTmp, colIndicesTmp
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

    checkCudaErrors(cudaMalloc((void**)&d_ldu_to_csr_index, csr_col_index_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_diag_to_csr_index, cell_index_bytes));
    checkCudaErrors(cudaMemcpy(d_ldu_to_csr_index, lduCSRIndex.data(), csr_col_index_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_diag_to_csr_index, diagCSRIndex.data(), cell_index_bytes, cudaMemcpyHostToDevice));

    // build d_csr_row_index, d_csr_col_index
    checkCudaErrors(cudaMalloc((void**)&d_csr_row_index, csr_row_index_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_csr_col_index, csr_col_index_bytes));
    checkCudaErrors(cudaMemcpy(d_csr_row_index, CSRRowIndex.data(), csr_row_index_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csr_col_index, CSRColIndex.data(), csr_col_index_bytes, cudaMemcpyHostToDevice));

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

    checkCudaErrors(cudaMalloc((void**)&d_ldu_to_csr_no_diag, csr_col_index_bytes_no_diag));
    checkCudaErrors(cudaMalloc((void**)&d_csr_row_index_no_diag, csr_row_index_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_csr_col_index_no_diag, csr_col_index_bytes_no_diag));
    checkCudaErrors(cudaMemcpy(d_ldu_to_csr_no_diag, lduCSRNoDiagIndex.data(), csr_col_index_bytes_no_diag, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csr_row_index_no_diag, CSRRowIndexNoDiag.data(), csr_row_index_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csr_col_index_no_diag, CSRColIndexNoDiag.data(), csr_col_index_bytes_no_diag, cudaMemcpyHostToDevice));

    // ell
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

    checkCudaErrors(cudaMalloc((void**)&d_ellCols, num_cells * ell_row_maxcount * sizeof(int)));
    checkCudaErrors(cudaMemset(d_ellCols, -1, num_cells * ell_row_maxcount * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&d_ldu2ellIndex, num_surfaces * 2 * sizeof(int)));
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
        int index = ell_row_maxcount * row + rowOffset[row];
        ldu2ellIndex[i] = index;
        ellCols[index] = owner[i];
        rowOffset[row] += 1;
    }
    // upper
    for(int i = 0; i < num_surfaces; i++){
        int row = owner[i];
        int index = ell_row_maxcount * row + rowOffset[row];
        ldu2ellIndex[i + num_surfaces] = index;
        ellCols[index] = neighbor[i];
        rowOffset[row] += 1;
    }

    checkCudaErrors(cudaMemcpy(d_ellCols, ellCols, sizeof(int) * num_cells * ell_row_maxcount, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_ldu2ellIndex, ldu2ellIndex, sizeof(int) * num_surfaces * 2, cudaMemcpyHostToDevice));

    free(ellColsperRow);
    free(ellCols);
    free(ldu2ellIndex);
}

void dfMatrixDataBase::createConstantFieldsInternal() {
    checkCudaErrors(cudaMalloc((void**)&d_sf, surface_value_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_mesh_dis, surface_value_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_mag_sf, surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_weight, surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_phi_weight, surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_delta_coeffs, surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_volume, cell_value_bytes));
    fieldPointerMap["d_sf"] = d_sf;
    fieldPointerMap["d_mesh_dis"] = d_mesh_dis;
    fieldPointerMap["d_mag_sf"] = d_mag_sf;
    fieldPointerMap["d_weight"] = d_weight;
    fieldPointerMap["d_phi_weight"] = d_phi_weight;
    fieldPointerMap["d_delta_coeffs"] = d_delta_coeffs;
    fieldPointerMap["d_volume"] = d_volume;

    checkCudaErrors(cudaMallocHost((void**)&h_sf, surface_value_vec_bytes));
    checkCudaErrors(cudaMallocHost((void**)&h_mesh_dis, surface_value_vec_bytes));
    fieldPointerMap["h_sf"] = h_sf;
    fieldPointerMap["h_mesh_dis"] = h_mesh_dis;
}

void dfMatrixDataBase::createConstantFieldsBoundary() {
    checkCudaErrors(cudaMalloc((void**)&d_boundary_sf, boundary_surface_value_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_mag_sf, boundary_surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_delta_coeffs, boundary_surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_weight, boundary_surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_face_cell, boundary_surface_index_bytes));
    fieldPointerMap["d_boundary_sf"] = d_boundary_sf;
    fieldPointerMap["d_boundary_mag_sf"] = d_boundary_mag_sf;
    fieldPointerMap["d_boundary_delta_coeffs"] = d_boundary_delta_coeffs;
    fieldPointerMap["d_boundary_weight"] = d_boundary_weight;

    checkCudaErrors(cudaMallocHost((void**)&h_boundary_sf, boundary_surface_value_vec_bytes));
    fieldPointerMap["h_boundary_sf"] = h_boundary_sf;
}

void dfMatrixDataBase::initConstantFieldsInternal(const double *sf, const double *mag_sf, 
        const double *weight, const double *delta_coeffs, const double *volume, const double *mesh_distance) {
    // permute sf
    for (int i = 0; i < num_surfaces; i++) {
        h_sf[num_surfaces * 0 + i] = sf[i * 3 + 0];
        h_sf[num_surfaces * 1 + i] = sf[i * 3 + 1];
        h_sf[num_surfaces * 2 + i] = sf[i * 3 + 2];
        h_mesh_dis[num_surfaces * 0 + i] = mesh_distance[i * 3 + 0];
        h_mesh_dis[num_surfaces * 1 + i] = mesh_distance[i * 3 + 1];
        h_mesh_dis[num_surfaces * 2 + i] = mesh_distance[i * 3 + 2];
    }
    checkCudaErrors(cudaMemcpyAsync(d_sf, h_sf, surface_value_vec_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_mesh_dis, h_mesh_dis, surface_value_vec_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_mag_sf, mag_sf, surface_value_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_weight, weight, surface_value_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_delta_coeffs, delta_coeffs, surface_value_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_volume, volume, cell_value_bytes, cudaMemcpyHostToDevice, stream));
}

void dfMatrixDataBase::initConstantFieldsBoundary(const double *boundary_sf, const double *boundary_mag_sf, 
        const double *boundary_delta_coeffs, const double *boundary_weight, const int *boundary_face_cell, std::vector<int>& patch_type_calculated,
        std::vector<int>& patch_type_extropolated) {
    this->patch_type_calculated = patch_type_calculated;
    this->patch_type_extropolated = patch_type_extropolated;
    // permute bouSf
    for (int i = 0; i < num_boundary_surfaces; i++) {
        h_boundary_sf[num_boundary_surfaces * 0 + i] = boundary_sf[i * 3 + 0];
        h_boundary_sf[num_boundary_surfaces * 1 + i] = boundary_sf[i * 3 + 1];
        h_boundary_sf[num_boundary_surfaces * 2 + i] = boundary_sf[i * 3 + 2];
    }
    checkCudaErrors(cudaMemcpyAsync(d_boundary_sf, h_boundary_sf, boundary_surface_value_vec_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_boundary_mag_sf, boundary_mag_sf, boundary_surface_value_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_boundary_delta_coeffs, boundary_delta_coeffs, boundary_surface_value_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_boundary_weight, boundary_weight, boundary_surface_value_bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_boundary_face_cell, boundary_face_cell, boundary_surface_index_bytes, cudaMemcpyHostToDevice, stream));  
}

void dfMatrixDataBase::createNonConstantFieldsInternal() {
    checkCudaErrors(cudaMalloc((void**)&d_rho, cell_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_u, cell_value_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_u_old, cell_value_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_u_old_host_order, cell_value_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_y, cell_value_bytes * num_species));
    checkCudaErrors(cudaMalloc((void**)&d_he, cell_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_p, cell_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_k, cell_value_bytes));
    fieldPointerMap["d_rho"] = d_rho;
    fieldPointerMap["d_u"] = d_u;
    fieldPointerMap["d_u_old"] = d_u_old;
    fieldPointerMap["d_y"] = d_y;
    fieldPointerMap["d_he"] = d_he;
    fieldPointerMap["d_p"] = d_p;
    fieldPointerMap["d_k"] = d_k;
    
    checkCudaErrors(cudaMalloc((void**)&d_rho_old, cell_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_k_old, cell_value_bytes));
    fieldPointerMap["d_rho_old"] = d_rho_old;
    fieldPointerMap["d_k_old"] = d_k_old;
    // checkCudaErrors(cudaMalloc((void**)&d_u_old, cell_value_vec_bytes));
    // checkCudaErrors(cudaMalloc((void**)&d_y_old, cell_value_bytes * num_species));
    // checkCudaErrors(cudaMalloc((void**)&d_he_old, cell_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_p_old, cell_value_bytes));
    fieldPointerMap["d_p_old"] = d_p_old;
    
    checkCudaErrors(cudaMalloc((void**)&d_phi, surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_phi_old, surface_value_bytes));
    fieldPointerMap["d_phi"] = d_phi;
    fieldPointerMap["d_phi_old"] = d_phi_old;

    // thermophysical fields
    checkCudaErrors(cudaMalloc((void**)&d_thermo_psi, cell_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_thermo_alpha, cell_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_thermo_rhoD, num_species * cell_value_bytes));

    checkCudaErrors(cudaMalloc((void**)&d_hDiff_corr_flux, cell_value_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_diff_alphaD, cell_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_dpdt, cell_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_T, cell_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_mu, cell_value_bytes));

    // turbulence fields
    checkCudaErrors(cudaMalloc((void**)&d_turbulence_k, cell_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_turbulence_epsilon, cell_value_bytes));

    // internal fields used between eqns
    checkCudaErrors(cudaMalloc((void**)&d_rAU, cell_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_HbyA, cell_value_vec_bytes));

    // computed on GPU, used on CPU, need memcpyd2h
    checkCudaErrors(cudaMallocHost((void**)&h_T, cell_value_bytes));
    checkCudaErrors(cudaMallocHost((void**)&h_rho, cell_value_bytes));
    checkCudaErrors(cudaMallocHost((void**)&h_rho_old, cell_value_bytes));
    checkCudaErrors(cudaMallocHost((void**)&h_u, cell_value_vec_bytes));
    checkCudaErrors(cudaMallocHost((void**)&h_u_old, cell_value_vec_bytes));
    checkCudaErrors(cudaMallocHost((void**)&h_y, cell_value_bytes * num_species));
    checkCudaErrors(cudaMallocHost((void**)&h_he, cell_value_bytes));
    checkCudaErrors(cudaMallocHost((void**)&h_k, cell_value_bytes));
    checkCudaErrors(cudaMallocHost((void**)&h_k_old, cell_value_bytes));
    fieldPointerMap["h_T"] = h_T;
    fieldPointerMap["h_rho"] = h_rho;
    fieldPointerMap["h_rho_old"] = h_rho_old;
    fieldPointerMap["h_u"] = h_u;
    fieldPointerMap["h_u_old"] = h_u_old;
    fieldPointerMap["h_y"] = h_y;
    fieldPointerMap["h_he"] = h_he;
    fieldPointerMap["h_k"] = h_k;
    fieldPointerMap["h_k_old"] = h_k_old;

    // computed on CPU, used on GPU, need memcpyh2d
    checkCudaErrors(cudaMallocHost((void**)&h_p, cell_value_bytes));
    checkCudaErrors(cudaMallocHost((void**)&h_p_old, cell_value_bytes));
    checkCudaErrors(cudaMallocHost((void**)&h_phi, surface_value_bytes));
    fieldPointerMap["h_p"] = h_p;
    fieldPointerMap["h_p_old"] = h_p_old;
    fieldPointerMap["h_phi"] = h_phi;
}

void dfMatrixDataBase::createNonConstantFieldsBoundary() {
    checkCudaErrors(cudaMalloc((void**)&d_boundary_rho, boundary_surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_u, boundary_surface_value_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_u_old, boundary_surface_value_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_u_old_host_order, boundary_surface_value_vec_bytes));

    checkCudaErrors(cudaMalloc((void**)&d_boundary_y, boundary_surface_value_bytes * num_species));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_he, boundary_surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_p, boundary_surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_k, boundary_surface_value_bytes));
    fieldPointerMap["d_boundary_rho"] = d_boundary_rho;
    fieldPointerMap["d_boundary_u"] = d_boundary_u;
    fieldPointerMap["d_boundary_u_old"] = d_boundary_u_old;
    fieldPointerMap["d_boundary_y"] = d_boundary_y;
    fieldPointerMap["d_boundary_he"] = d_boundary_he;
    fieldPointerMap["d_boundary_p"] = d_boundary_p;
    fieldPointerMap["d_boundary_k"] = d_boundary_k;

    checkCudaErrors(cudaMalloc((void**)&d_boundary_rho_old, boundary_surface_value_bytes));
    fieldPointerMap["d_boundary_rho_old"] = d_boundary_rho_old;
    // checkCudaErrors(cudaMalloc((void**)&d_boundary_u_old, boundary_surface_value_vec_bytes));
    // checkCudaErrors(cudaMalloc((void**)&d_boundary_y_old, boundary_surface_value_bytes * num_species));
    // checkCudaErrors(cudaMalloc((void**)&d_boundary_he_old, boundary_surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_p_old, boundary_surface_value_bytes));

    checkCudaErrors(cudaMalloc((void**)&d_boundary_phi, boundary_surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_phi_old, boundary_surface_value_bytes));
    fieldPointerMap["d_boundary_phi"] = d_boundary_phi;
    fieldPointerMap["d_boundary_phi_old"] = d_boundary_phi_old;

    // thermophysical fields
    checkCudaErrors(cudaMalloc((void**)&d_boundary_thermo_psi, boundary_surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_thermo_alpha, boundary_surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_thermo_rhoD, num_species * boundary_surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_hDiff_corr_flux, boundary_surface_value_vec_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_diff_alphaD, boundary_surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_T, boundary_surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_mu, boundary_surface_value_bytes));

    // internal fields used between eqns
    checkCudaErrors(cudaMalloc((void**)&d_boundary_rAU, boundary_surface_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_boundary_HbyA, boundary_surface_value_vec_bytes));

    // computed on GPU, used on CPU, need memcpyd2h
    checkCudaErrors(cudaMallocHost((void**)&h_boundary_rho, boundary_surface_value_bytes));
    checkCudaErrors(cudaMallocHost((void**)&h_boundary_rho_old, boundary_surface_value_bytes));
    checkCudaErrors(cudaMallocHost((void**)&h_boundary_u, boundary_surface_value_vec_bytes));
    checkCudaErrors(cudaMallocHost((void**)&h_boundary_u_old, boundary_surface_value_vec_bytes));
    checkCudaErrors(cudaMallocHost((void**)&h_boundary_y, boundary_surface_value_bytes * num_species));
    checkCudaErrors(cudaMallocHost((void**)&h_boundary_he, boundary_surface_value_bytes));
    checkCudaErrors(cudaMallocHost((void**)&h_boundary_k, boundary_surface_value_bytes));
    fieldPointerMap["h_boundary_rho"] = h_boundary_rho;
    fieldPointerMap["h_boundary_rho_old"] = h_boundary_rho_old;
    fieldPointerMap["h_boundary_u"] = h_boundary_u;
    fieldPointerMap["h_boundary_u_old"] = h_boundary_u_old;
    fieldPointerMap["h_boundary_y"] = h_boundary_y;
    fieldPointerMap["h_boundary_he"] = h_boundary_he;
    fieldPointerMap["h_boundary_k"] = h_boundary_k;

    // computed on CPU, used on GPU, need memcpyh2d
    checkCudaErrors(cudaMallocHost((void**)&h_boundary_p, boundary_surface_value_bytes));
    checkCudaErrors(cudaMallocHost((void**)&h_boundary_p_old, boundary_surface_value_bytes));
    checkCudaErrors(cudaMallocHost((void**)&h_boundary_phi, boundary_surface_value_bytes));
    checkCudaErrors(cudaMallocHost((void**)&h_boundary_phi_old, boundary_surface_value_bytes));
    fieldPointerMap["h_boundary_p"] = h_boundary_p;
    fieldPointerMap["h_boundary_p_old"] = h_boundary_p_old;
    fieldPointerMap["h_boundary_phi"] = h_boundary_phi;
    fieldPointerMap["h_boundary_phi_old"] = h_boundary_phi_old;
}

void dfMatrixDataBase::preTimeStep() {
    checkCudaErrors(cudaMemcpyAsync(d_rho_old, d_rho, cell_value_bytes, cudaMemcpyDeviceToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_boundary_rho_old, d_boundary_rho, boundary_surface_value_bytes, cudaMemcpyDeviceToDevice, stream));
    
    checkCudaErrors(cudaMemcpyAsync(d_phi_old, d_phi, surface_value_bytes, cudaMemcpyDeviceToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_boundary_phi_old, d_boundary_phi, boundary_surface_value_bytes, cudaMemcpyDeviceToDevice, stream));

    checkCudaErrors(cudaMemcpyAsync(d_u_old, d_u, cell_value_vec_bytes, cudaMemcpyDeviceToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_boundary_u_old, d_boundary_u, boundary_surface_value_vec_bytes, cudaMemcpyDeviceToDevice, stream));
    
    checkCudaErrors(cudaMemcpyAsync(d_k_old, d_k, cell_value_bytes, cudaMemcpyDeviceToDevice, stream));

    checkCudaErrors(cudaMemcpyAsync(d_p_old, d_p, cell_value_bytes, cudaMemcpyDeviceToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_boundary_p_old, d_boundary_p, boundary_surface_value_bytes, cudaMemcpyDeviceToDevice, stream));
}

void dfMatrixDataBase::postTimeStep() {}

double* dfMatrixDataBase::getFieldPointer(const char* fieldAlias, location loc, position pos) {
    char mergedName[256];
    if (pos == position::internal) {
        sprintf(mergedName, "%s_%s", (loc == location::cpu) ? "h" : "d", fieldAlias);
    } else if (pos == position::boundary) {
        sprintf(mergedName, "%s_boundary_%s", (loc == location::cpu) ? "h" : "d", fieldAlias);
    }

    double *pointer = nullptr;
    if (fieldPointerMap.find(std::string(mergedName)) != fieldPointerMap.end()) {
        pointer = fieldPointerMap[std::string(mergedName)];
    }
    if (pointer == nullptr) {
        fprintf(stderr, "Warning! getFieldPointer of %s returns nullptr!\n", mergedName);
    }
    //fprintf(stderr, "fieldAlias: %s, mergedName: %s, pointer: %p\n", fieldAlias, mergedName, pointer);

    return pointer;
}
