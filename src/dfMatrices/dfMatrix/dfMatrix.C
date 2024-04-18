#include "dfMatrix.H"

void checkVectorEqual(int count, double* basevec, double* vec, double max_relative_error) {
    for (size_t i = 0; i < count; ++i)
    {
        double abs_diff = fabs(basevec[i] - vec[i]);
        double rel_diff = fabs(basevec[i] - vec[i]) / fabs(basevec[i]);
        if (abs_diff > 1e-16 && rel_diff > max_relative_error)
            fprintf(stderr, "mismatch index %d, cpu data: %.16lf, gpu data: %.16lf, relative error: %.16lf\n", i, basevec[i], vec[i], rel_diff);
    }
}

// constructor (construct mesh variable)
dfMatrix::dfMatrix(){}
dfMatrix::dfMatrix(int num_surfaces, int num_cells, int num_boundary_faces, int & num_boundary_cells_output,
    const int *neighbour, const int *owner, const double* volume, const double* weight, const double* face_vector, 
    std::vector<double> boundary_face_vector_init, std::vector<int> boundary_cell_id_init)
: num_cells(num_cells), num_faces(num_surfaces*2), num_surfaces(num_surfaces), num_boundary_faces(num_boundary_faces), 
  h_volume(volume), time_monitor_CPU(0.), time_monitor_GPU(0.)
{   
    h_weight_vec_init.resize(num_faces);
    h_weight_vec.resize(num_faces);
    h_face_vector_vec_init.resize(num_faces*3);
    h_face_vector_vec.resize(num_faces*3);
    h_turbSrc_init_mtx_vec.resize(num_faces + num_cells);
    h_turbSrc_init_1mtx.resize(num_faces + num_cells);
    h_turbSrc_init_src_vec.resize(3*num_cells);
    h_turbSrc_src_vec.resize(3*num_cells);

    h_csr_row_index_vec.resize(num_cells + 1, 0);
    h_csr_diag_index_vec.resize(num_cells);

    h_A_csr = new double[(num_cells + num_faces) * 3];
    h_b = new double[num_cells * 3];
    h_psi = new double[num_cells * 3];
    h_phi_init = new double[num_faces];
    h_phi = new double[num_faces];
    

    // byte sizes
    cell_bytes = num_cells * sizeof(double);
    cell_vec_bytes = num_cells * 3 * sizeof(double);
    cell_index_bytes = num_cells * sizeof(int);

    face_bytes = num_faces * sizeof(double);
    face_vec_bytes = num_faces * 3 * sizeof(double);
    face_index_bytes = num_faces * sizeof(int);

    // A_csr has one more element in each row: itself
    csr_row_index_bytes = (num_cells + 1) * sizeof(int);
    csr_col_index_bytes = (num_cells + num_faces) * sizeof(int);
    csr_value_bytes = (num_cells + num_faces) * sizeof(double);
    csr_value_vec_bytes = (num_cells + num_faces) * 3 * sizeof(double);

    /************************construct mesh variables****************************/
    /**
     * 1. h_csr_row_index & h_csr_diag_index
    */
    std::vector<int> h_mtxEntry_perRow_vec(num_cells);

    for (int faceI = 0; faceI < num_surfaces; faceI++)
    {
        h_csr_diag_index_vec[neighbour[faceI]]++;
        h_mtxEntry_perRow_vec[neighbour[faceI]]++;
        h_mtxEntry_perRow_vec[owner[faceI]]++;
    }

    // - consider diagnal element in each row
    std::transform(h_mtxEntry_perRow_vec.begin(), h_mtxEntry_perRow_vec.end(), h_mtxEntry_perRow_vec.begin(), [](int n)
        {return n + 1;});
    // - construct h_csr_row_index & h_csr_diag_index
    std::partial_sum(h_mtxEntry_perRow_vec.begin(), h_mtxEntry_perRow_vec.end(), h_csr_row_index_vec.begin()+1);
    
    /**
     * 2. h_csr_col_index
    */
    std::vector<int> rowIndex(num_faces + num_cells), colIndex(num_faces + num_cells), diagIndex(num_cells);
    std::iota(diagIndex.begin(), diagIndex.end(), 0);

    // initialize the RowIndex (rowIndex of lower + upper + diagnal)
    std::copy(neighbour, neighbour + num_surfaces, rowIndex.begin());
    std::copy(owner, owner + num_surfaces, rowIndex.begin() + num_surfaces);
    std::copy(diagIndex.begin(), diagIndex.end(), rowIndex.begin() + num_faces);
    // initialize the ColIndex (colIndex of lower + upper + diagnal)
    std::copy(owner, owner + num_surfaces, colIndex.begin());
    std::copy(neighbour, neighbour + num_surfaces, colIndex.begin() + num_surfaces);
    std::copy(diagIndex.begin(), diagIndex.end(), colIndex.begin() + num_faces);

    // - construct hashTable for sorting
    std::multimap<int,int> rowColPair;
    for (int i = 0; i < 2*num_surfaces+num_cells; i++)
    {
        rowColPair.insert(std::make_pair(rowIndex[i], colIndex[i]));
    }
    // - sort
    std::vector<std::pair<int, int>> globalPerm(rowColPair.begin(), rowColPair.end());
    std::sort(globalPerm.begin(), globalPerm.end(), []
    (const std::pair<int, int>& pair1, const std::pair<int, int>& pair2){
    if (pair1.first != pair2.first) {
        return pair1.first < pair2.first;
    } else {
        return pair1.second < pair2.second;
    }
    });

    std::transform(globalPerm.begin(), globalPerm.end(), std::back_inserter(h_csr_col_index_vec), []
        (const std::pair<int, int>& pair) {
        return pair.second;
    });

    // construct a tmp permutated List for fvMatrix addition
    std::vector<int> tmp_permutation(2*num_surfaces + num_cells);
    std::vector<int> tmp_rowIndex(2*num_surfaces + num_cells);
    std::iota(tmp_permutation.begin(), tmp_permutation.end(), 0);
    std::copy(neighbour, neighbour + num_surfaces, tmp_rowIndex.begin());
    std::copy(diagIndex.begin(), diagIndex.end(), tmp_rowIndex.begin() + num_surfaces);
    std::copy(owner, owner + num_surfaces, tmp_rowIndex.begin() + num_surfaces + num_cells);
    std::multimap<int,int> tmpPair;
    for (int i = 0; i < 2*num_surfaces+num_cells; i++)
    {
        tmpPair.insert(std::make_pair(tmp_rowIndex[i], tmp_permutation[i]));
    }
    std::vector<std::pair<int, int>> tmpPerm(tmpPair.begin(), tmpPair.end());
    std::sort(tmpPerm.begin(), tmpPerm.end(), []
    (const std::pair<int, int>& pair1, const std::pair<int, int>& pair2){
    if (pair1.first != pair2.first) {
        return pair1.first < pair2.first;
    } else {
        return pair1.second < pair2.second;
    }
    });
    std::transform(tmpPerm.begin(), tmpPerm.end(), std::back_inserter(tmpPermutatedList), []
        (const std::pair<int, int>& pair) {
        return pair.second;
    });

    /**
     * 3. boundary imformations
    */
    // get boundPermutation and offset lists
    std::vector<int> boundPermutationListInit(num_boundary_faces);
    std::vector<int> boundOffsetList;
    std::iota(boundPermutationListInit.begin(), boundPermutationListInit.end(), 0);

    // - construct hashTable for sorting
    std::multimap<int,int> boundPermutation;
    for (int i = 0; i < num_boundary_faces; i++)
    {
        boundPermutation.insert(std::make_pair(boundary_cell_id_init[i], boundPermutationListInit[i]));
    }

    // - sort 
    std::vector<std::pair<int, int>> boundPermPair(boundPermutation.begin(), boundPermutation.end());
    std::sort(boundPermPair.begin(), boundPermPair.end(), []
    (const std::pair<int, int>& pair1, const std::pair<int, int>& pair2){
        if (pair1.first != pair2.first) {
            return pair1.first < pair2.first;
        } else {
            return pair1.second < pair2.second;
        }
    });

    // - construct boundPermedIndex and boundary_cell_id
    boundPermutationList.clear();
    std::transform(boundPermPair.begin(), boundPermPair.end(), std::back_inserter(boundary_cell_id), []
        (const std::pair<int, int>& pair) {
        return pair.first;
    });
    std::transform(boundPermPair.begin(), boundPermPair.end(), std::back_inserter(boundPermutationList), []
        (const std::pair<int, int>& pair) {
        return pair.second;
    });

    // construct boundary_cell_offset
    std::map<int, int> countMap;
    std::vector<int> boundaryCellcount;
    for (const auto& cellIndex : boundary_cell_id)
        ++ countMap[cellIndex];
    for (const auto& [cellIndex, count] : countMap)
        boundaryCellcount.push_back(count);

    num_boundary_cells = boundaryCellcount.size();
    num_boundary_cells_output = num_boundary_cells;

    boundary_cell_offset.resize(boundaryCellcount.size() + 1, 0);
    std::partial_sum(boundaryCellcount.begin(), boundaryCellcount.end(), boundary_cell_offset.begin()+1);
    
    // assign h_boundary_cell_offset & h_boundary_cell_id
    h_boundary_cell_offset = boundary_cell_offset.data();
    h_boundary_cell_id = boundary_cell_id.data();

    // 
    boundary_cell_bytes = num_boundary_cells * sizeof(double);
    boundary_cell_vec_bytes = num_boundary_cells * 3 * sizeof(double);
    boundary_cell_index_bytes = num_boundary_cells * sizeof(int);

    boundary_face_bytes = num_boundary_faces * sizeof(double);
    boundary_face_vec_bytes = num_boundary_faces * 3 * sizeof(double);
    boundary_face_index_bytes = num_boundary_faces * sizeof(int);

    ueqn_internalCoeffs.resize(3*num_boundary_faces);
    ueqn_boundaryCoeffs.resize(3*num_boundary_faces);

    boundary_face_vector.resize(3*num_boundary_faces);
    boundary_pressure.resize(num_boundary_faces);


    /**
     * 4. construct permutation list for field variables
    */
    std::vector<int> offdiagRowIndex(2*num_surfaces), permIndex(2*num_surfaces);
    // - initialize the offdiagRowIndex (rowIndex of lower + rowIndex of upper)
    std::copy(neighbour, neighbour + num_surfaces, offdiagRowIndex.begin());
    std::copy(owner, owner + num_surfaces, offdiagRowIndex.begin() + num_surfaces);

    // - initialize the permIndex (0, 1, ..., 2*num_surfaces)
    std::iota(permIndex.begin(), permIndex.end(), 0);

    // - construct hashTable for sorting
    std::multimap<int,int> permutation;
    for (int i = 0; i < 2*num_surfaces; i++)
    {
        permutation.insert(std::make_pair(offdiagRowIndex[i], permIndex[i]));
    }
    // - sort 
    std::vector<std::pair<int, int>> permPair(permutation.begin(), permutation.end());
    std::sort(permPair.begin(), permPair.end(), []
    (const std::pair<int, int>& pair1, const std::pair<int, int>& pair2){
        if (pair1.first != pair2.first) {
            return pair1.first < pair2.first;
        } else {
            return pair1.second < pair2.second;
        }
    });
    // - form permedIndex list
    std::transform(permPair.begin(), permPair.end(), std::back_inserter(permedIndex), []
        (const std::pair<int, int>& pair) {
        return pair.second;
    });

    // copy and permutate cell variables
    std::copy(weight, weight + num_surfaces, h_weight_vec_init.begin());
    std::copy(weight, weight + num_surfaces, h_weight_vec_init.begin() + num_surfaces);
    std::copy(face_vector, face_vector + 3*num_surfaces, h_face_vector_vec_init.begin());
    std::copy(face_vector, face_vector + 3*num_surfaces, h_face_vector_vec_init.begin() + 3*num_surfaces);
    for (int i = 0; i < num_faces; i++)
    {
        h_weight_vec[i] = h_weight_vec_init[permedIndex[i]];
        h_face_vector_vec[i*3] = h_face_vector_vec_init[3*permedIndex[i]];
        h_face_vector_vec[i*3+1] = h_face_vector_vec_init[3*permedIndex[i]+1];
        h_face_vector_vec[i*3+2] = h_face_vector_vec_init[3*permedIndex[i]+2];
    }
    h_face_vector = h_face_vector_vec.data();

    for (int i = 0; i < num_boundary_faces; i++)
    {
        boundary_face_vector[3*i] = boundary_face_vector_init[3*boundPermutationList[i]];
        boundary_face_vector[3*i+1] = boundary_face_vector_init[3*boundPermutationList[i]+1];
        boundary_face_vector[3*i+2] = boundary_face_vector_init[3*boundPermutationList[i]+2];
    }
    h_boundary_face_vector = boundary_face_vector.data();
}

dfMatrix::~dfMatrix()
{
}

void dfMatrix::fvm_ddt(double *rho_old, double *rho_new, double* vector_old)
{
    clock_t start = std::clock();
    // copy cell variables directly
    h_velocity_old = vector_old;
    
    int csr_dim = num_cells + num_faces;
    int row_index, diag_index, csr_index;
    double ddt_diag, ddt_part_term;
    for (int index = 0; index < num_cells; index++)
    {
        row_index = h_csr_row_index_vec[index];
        diag_index = h_csr_diag_index_vec[index];
        csr_index = row_index + diag_index;

        ddt_diag = rdelta_t * rho_new[index] * h_volume[index];
        ddt_part_term = rdelta_t * rho_old[index] * h_volume[index];
        
        h_A_csr[csr_dim * 0 + csr_index] = h_A_csr[csr_dim * 0 + csr_index] + ddt_diag;
        h_A_csr[csr_dim * 1 + csr_index] = h_A_csr[csr_dim * 1 + csr_index] + ddt_diag;
        h_A_csr[csr_dim * 2 + csr_index] = h_A_csr[csr_dim * 2 + csr_index] + ddt_diag;

        h_b[num_cells * 0 + index] = h_b[num_cells * 0 + index] + ddt_part_term * h_velocity_old[index * 3 + 0];
        h_b[num_cells * 1 + index] = h_b[num_cells * 1 + index] + ddt_part_term * h_velocity_old[index * 3 + 1];
        h_b[num_cells * 2 + index] = h_b[num_cells * 2 + index] + ddt_part_term * h_velocity_old[index * 3 + 2];

        h_psi[num_cells * 0 + index] = h_velocity_old[index * 3 + 0];
        h_psi[num_cells * 1 + index] = h_velocity_old[index * 3 + 1];
        h_psi[num_cells * 2 + index] = h_velocity_old[index * 3 + 2];
    }
    clock_t end = std::clock();
    time_monitor_GPU += double(end - start) / double(CLOCKS_PER_SEC);
}

void dfMatrix::fvm_div(double* phi, double* ueqn_internalCoeffs_init,
    double* ueqn_boundaryCoeffs_init, double* boundary_pressure_init)
{
    // copy and permutate face variables
    std::copy(phi, phi + num_surfaces, h_phi_init);
    std::copy(phi, phi + num_surfaces, h_phi_init + num_surfaces);

    for (size_t i = 0; i < num_faces; i++)
        h_phi[i] = h_phi_init[permedIndex[i]];
    
    for (size_t index = 0; index < num_boundary_faces; index++)
    {
        ueqn_internalCoeffs[3*index] = ueqn_internalCoeffs_init[3*boundPermutationList[index]];
        ueqn_internalCoeffs[3*index + 1] = ueqn_internalCoeffs_init[3*boundPermutationList[index] + 1];
        ueqn_internalCoeffs[3*index + 2] = ueqn_internalCoeffs_init[3*boundPermutationList[index] + 2];
        ueqn_boundaryCoeffs[3*index] = ueqn_boundaryCoeffs_init[3*boundPermutationList[index]];
        ueqn_boundaryCoeffs[3*index + 1] = ueqn_boundaryCoeffs_init[3*boundPermutationList[index] + 1];
        ueqn_boundaryCoeffs[3*index + 2] = ueqn_boundaryCoeffs_init[3*boundPermutationList[index] + 2];
        boundary_pressure[index] = boundary_pressure_init[boundPermutationList[index]];
    }

    int csr_dim = num_cells + num_faces;
    for (size_t index = 0; index < num_cells; index++)
    {
        int row_index = h_csr_row_index_vec[index];
        int next_row_index = h_csr_row_index_vec[index + 1];
        int diag_index = h_csr_diag_index_vec[index];
        int neighbor_offset = h_csr_row_index_vec[index] - index;

        double div_diag = 0;
        for (int i = row_index; i < next_row_index; i++) 
        {
            int inner_index = i - row_index;
            // lower
            if (inner_index < diag_index) {
                int neighbor_index = neighbor_offset + inner_index;
                double w = h_weight_vec[neighbor_index];
                double f = h_phi[neighbor_index];
                h_A_csr[csr_dim * 0 + i] = h_A_csr[csr_dim * 0 + i] + (-w) * f;
                h_A_csr[csr_dim * 1 + i] = h_A_csr[csr_dim * 1 + i] + (-w) * f;
                h_A_csr[csr_dim * 2 + i] = h_A_csr[csr_dim * 2 + i] + (-w) * f;
                // lower neighbors contribute to sum of -1
                div_diag += (w - 1) * f;
            }
            // upper
            if (inner_index > diag_index) {
                // upper, index - 1, consider of diag
                int neighbor_index = neighbor_offset + inner_index - 1;
                double w = h_weight_vec[neighbor_index];
                double f = h_phi[neighbor_index];
                h_A_csr[csr_dim * 0 + i] = h_A_csr[csr_dim * 0 + i] + (1 - w) * f;
                h_A_csr[csr_dim * 1 + i] = h_A_csr[csr_dim * 1 + i] + (1 - w) * f;
                h_A_csr[csr_dim * 2 + i] = h_A_csr[csr_dim * 2 + i] + (1 - w) * f;
                // upper neighbors contribute to sum of 1
                div_diag += w * f;
            }
        }
        h_A_csr[csr_dim * 0 + row_index + diag_index] = h_A_csr[csr_dim * 0 + row_index + diag_index] + div_diag; // diag
        h_A_csr[csr_dim * 1 + row_index + diag_index] = h_A_csr[csr_dim * 1 + row_index + diag_index] + div_diag; // diag
        h_A_csr[csr_dim * 2 + row_index + diag_index] = h_A_csr[csr_dim * 2 + row_index + diag_index] + div_diag; // diag
    }
    for (size_t index = 0; index < num_boundary_cells; index++)
    {
        int cell_offset = boundary_cell_offset[index];
        int cell_index = boundary_cell_id[cell_offset];
        int loop_size = boundary_cell_offset[index + 1] - cell_offset;

        int row_index = h_csr_row_index_vec[cell_index];
        int diag_index = h_csr_diag_index_vec[cell_index];
        int csr_index = row_index + diag_index;

        double internal_coeffs_x = 0;
        double internal_coeffs_y = 0;
        double internal_coeffs_z = 0;
        double boundary_coeffs_x = 0;
        double boundary_coeffs_y = 0;
        double boundary_coeffs_z = 0;
        for (int i = 0; i < loop_size; i++) 
        {
            internal_coeffs_x += ueqn_internalCoeffs[(cell_offset + i) * 3 + 0];
            internal_coeffs_y += ueqn_internalCoeffs[(cell_offset + i) * 3 + 1];
            internal_coeffs_z += ueqn_internalCoeffs[(cell_offset + i) * 3 + 2];
            boundary_coeffs_x += ueqn_boundaryCoeffs[(cell_offset + i) * 3 + 0];
            boundary_coeffs_y += ueqn_boundaryCoeffs[(cell_offset + i) * 3 + 1];
            boundary_coeffs_z += ueqn_boundaryCoeffs[(cell_offset + i) * 3 + 2];
        }
        h_A_csr[csr_dim * 0 + csr_index] = h_A_csr[csr_dim * 0 + csr_index] + internal_coeffs_x;
        h_A_csr[csr_dim * 1 + csr_index] = h_A_csr[csr_dim * 1 + csr_index] + internal_coeffs_y;
        h_A_csr[csr_dim * 2 + csr_index] = h_A_csr[csr_dim * 2 + csr_index] + internal_coeffs_z;
        h_b[num_cells * 0 + cell_index] = h_b[num_cells * 0 + cell_index] + boundary_coeffs_x;
        h_b[num_cells * 1 + cell_index] = h_b[num_cells * 1 + cell_index] + boundary_coeffs_y;
        h_b[num_cells * 2 + cell_index] = h_b[num_cells * 2 + cell_index] + boundary_coeffs_z;
    }
}

void dfMatrix::fvc_grad(double* pressure)
{
    // copy cell variables directly
    h_pressure = pressure;

    // Copy the host input array in host memory to the device input array in device memory
    for (int index = 0; index < num_cells; index++)
    {
        int row_index = h_csr_row_index_vec[index];
        int next_row_index = h_csr_row_index_vec[index + 1];
        int diag_index = h_csr_diag_index_vec[index];
        int neighbor_offset = h_csr_row_index_vec[index] - index;

        double own_cell_p = pressure[index];
        double grad_bx = 0;
        double grad_by = 0;
        double grad_bz = 0;
        for (int i = row_index; i < next_row_index; i++) {
        int inner_index = i - row_index;
        // lower
        if (inner_index < diag_index) {
            int neighbor_index = neighbor_offset + inner_index;
            double w = h_weight_vec[neighbor_index];
            double sfx = h_face_vector[neighbor_index * 3 + 0];
            double sfy = h_face_vector[neighbor_index * 3 + 1];
            double sfz = h_face_vector[neighbor_index * 3 + 2];
            int neighbor_cell_id = h_csr_col_index_vec[row_index + inner_index];
            double neighbor_cell_p = pressure[neighbor_cell_id];
            double face_p = (1 - w) * own_cell_p + w * neighbor_cell_p;
            grad_bx -= face_p * sfx;
            grad_by -= face_p * sfy;
            grad_bz -= face_p * sfz;
        }
        // upper
        if (inner_index > diag_index) {
            int neighbor_index = neighbor_offset + inner_index - 1;
            double w = h_weight_vec[neighbor_index];
            double sfx = h_face_vector[neighbor_index * 3 + 0];
            double sfy = h_face_vector[neighbor_index * 3 + 1];
            double sfz = h_face_vector[neighbor_index * 3 + 2];
            int neighbor_cell_id = h_csr_col_index_vec[row_index + inner_index + 1];
            double neighbor_cell_p = pressure[neighbor_cell_id];
            double face_p = (1 - w) * own_cell_p + w * neighbor_cell_p;
            grad_bx += face_p * sfx;
            grad_by += face_p * sfy;
            grad_bz += face_p * sfz;
        }
        }
        h_b[num_cells * 0 + index] = h_b[num_cells * 0 + index] + grad_bx;
        h_b[num_cells * 1 + index] = h_b[num_cells * 1 + index] + grad_by;
        h_b[num_cells * 2 + index] = h_b[num_cells * 2 + index] + grad_bz;
    }
    for (int index = 0; index < num_boundary_cells; index++)
    {
        int cell_offset = boundary_cell_offset[index];
        int next_cell_offset = boundary_cell_offset[index + 1];
        int cell_index = boundary_cell_id[cell_offset];

        // compute boundary gradient
        double grad_bx = 0; 
        double grad_by = 0; 
        double grad_bz = 0; 
        for (int i = cell_offset; i < next_cell_offset; i++) 
        {
            double sfx = boundary_face_vector[i * 3 + 0];
            double sfy = boundary_face_vector[i * 3 + 1];
            double sfz = boundary_face_vector[i * 3 + 2];
            double face_p = boundary_pressure[i];
            grad_bx += face_p * sfx;
            grad_by += face_p * sfy;
            grad_bz += face_p * sfz;
        }
        h_b[num_cells * 0 + cell_index] = h_b[num_cells * 0 + cell_index] + grad_bx;
        h_b[num_cells * 1 + cell_index] = h_b[num_cells * 1 + cell_index] + grad_by;
        h_b[num_cells * 2 + cell_index] = h_b[num_cells * 2 + cell_index] + grad_bz;
    }
}

void dfMatrix::add_fvMatrix(double* turbSrc_low, double* turbSrc_diag, double* turbSrc_upp, double* turbSrc_source)
{
    // copy and permutate matrix variables
    std::copy(turbSrc_low, turbSrc_low + num_surfaces, h_turbSrc_init_mtx_vec.begin());
    std::copy(turbSrc_diag, turbSrc_diag + num_cells, h_turbSrc_init_mtx_vec.begin() + num_surfaces);
    std::copy(turbSrc_upp, turbSrc_upp + num_surfaces, h_turbSrc_init_mtx_vec.begin() + num_surfaces + num_cells);

    // permutate
    for (int index = 0; index < num_cells + num_faces; index++)
        h_turbSrc_init_1mtx[index] = h_turbSrc_init_mtx_vec[tmpPermutatedList[index]];
    
    for (int index = 0; index < num_cells; index++)
    {
        int row_index = h_csr_row_index_vec[index];
        int next_row_index = h_csr_row_index_vec[index + 1];
        int csr_dim = num_cells + num_faces;
        double A_entry;
        for (int i = row_index; i < next_row_index; i++)
        {
            A_entry = h_turbSrc_init_1mtx[i];
            h_A_csr[csr_dim * 0 + i] = h_A_csr[csr_dim * 0 + i] + A_entry;
            h_A_csr[csr_dim * 1 + i] = h_A_csr[csr_dim * 1 + i] + A_entry;
            h_A_csr[csr_dim * 2 + i] = h_A_csr[csr_dim * 2 + i] + A_entry;
        }
        h_b[num_cells * 0 + index] = h_b[num_cells * 0 + index] + turbSrc_source[index * 3 + 0];
        h_b[num_cells * 1 + index] = h_b[num_cells * 1 + index] + turbSrc_source[index * 3 + 1];
        h_b[num_cells * 2 + index] = h_b[num_cells * 2 + index] + turbSrc_source[index * 3 + 2];
    }
}

void dfMatrix::checkValue(bool print)
{
    if (print)
    {
        for (int i = 0; i < (2*num_surfaces + num_cells); i++)
            fprintf(stderr, "h_A_csr[%d]: %.15lf\n", i, h_A_csr[i]);
        for (int i = 0; i < num_cells * 3; i++)
            fprintf(stderr, "h_b[%d]: %.15lf\n", i, h_b[i]);
    }

    char *input_file = "of_output.txt";
    FILE *fp = fopen(input_file, "rb+");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open input file: %s!\n", input_file);
    }
    int readfile = 0;
    double *of_b = new double[3*num_cells];
    double *of_A = new double[2*num_surfaces + num_cells];
    readfile = fread(of_b, num_cells * 3 * sizeof(double), 1, fp);
    readfile = fread(of_A, (2*num_surfaces + num_cells) * sizeof(double), 1, fp);

    std::vector<double> h_A_of_init_vec(num_cells+2*num_surfaces);
    std::copy(of_A, of_A + num_cells+2*num_surfaces, h_A_of_init_vec.begin());

    std::vector<double> h_A_of_vec_1mtx(2*num_surfaces + num_cells, 0);
    for (int i = 0; i < 2*num_surfaces + num_cells; i++)
    {
        h_A_of_vec_1mtx[i] = h_A_of_init_vec[tmpPermutatedList[i]];
    }

    std::vector<double> h_A_of_vec((2*num_surfaces + num_cells)*3);
    for (int i =0; i < 3; i ++)
    {
        std::copy(h_A_of_vec_1mtx.begin(), h_A_of_vec_1mtx.end(), h_A_of_vec.begin()+i*(2*num_surfaces + num_cells));
    }

    // b
    std::vector<double> h_b_of_init_vec(3*num_cells);
    std::copy(of_b, of_b + 3*num_cells, h_b_of_init_vec.begin());
    std::vector<double> h_b_of_vec;
    for (int i = 0; i < 3*num_cells; i+=3)
    {
        h_b_of_vec.push_back(h_b_of_init_vec[i]);
    }
    // fill RHS_y
    for (int i = 1; i < 3*num_cells; i+=3)
    {
        h_b_of_vec.push_back(h_b_of_init_vec[i]);
    }
    // fill RHS_z
    for (int i = 2; i < 3*num_cells; i+=3)
    {
        h_b_of_vec.push_back(h_b_of_init_vec[i]);
    }

    if (print)
    {
        for (int i = 0; i < (2*num_surfaces + num_cells); i++)
            fprintf(stderr, "h_A_of_vec_1mtx[%d]: %.15lf\n", i, h_A_of_vec_1mtx[i]);
        for (int i = 0; i < 3*num_cells; i++)
            fprintf(stderr, "h_b_of_vec[%d]: %.15lf\n", i, h_b_of_vec[i]);
    }

    // check
    fprintf(stderr, "check of h_A_csr\n");
    checkVectorEqual(2*num_surfaces + num_cells, h_A_of_vec_1mtx.data(), h_A_csr, 1e-5);
    fprintf(stderr, "check of h_b\n");
    checkVectorEqual(3*num_cells, h_b_of_vec.data(), h_b, 1e-5);
}

void dfMatrix::solve(){}

void dfMatrix::updatePsi(double* Psi)
{
    for (size_t i = 0; i < num_cells; i++)
    {
        Psi[i*3] = h_psi[i];
        Psi[i*3 + 1] = h_psi[num_cells + i];
        Psi[i*3 + 2] = h_psi[num_cells*2 + i];
    }
}
