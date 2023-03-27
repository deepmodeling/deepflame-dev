/**
 * \file AmgXSolver.cpp
 * \brief Definition of member functions of the class AmgXSolver.
 * \author Pi-Yueh Chuang (pychuang@gwu.edu)
 * \author Matt Martineau (mmartineau@nvidia.com)
 * \date 2015-09-01
 * \copyright Copyright (c) 2015-2019 Pi-Yueh Chuang, Lorena A. Barba.
 * \copyright Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
 *            This project is released under MIT License.
 */

// AmgXWrapper
#include "AmgXSolver.H"
#include <numeric>
#include <limits>

// initialize AmgXSolver::count to 0
int AmgXSolver::count = 0;

// initialize AmgXSolver::rsrc to nullptr;
AMGX_resources_handle AmgXSolver::rsrc = nullptr;


/* \implements AmgXSolver::AmgXSolver */
AmgXSolver::AmgXSolver(const MPI_Comm &comm,
        const std::string &modeStr, const std::string &cfgFile)
{
    initialize(comm, modeStr, cfgFile);
}


/* \implements AmgXSolver::~AmgXSolver */
AmgXSolver::~AmgXSolver()
{
    if (isInitialised) finalize();
}


/* \implements AmgXSolver::initialize */
void AmgXSolver::initialize(const MPI_Comm &comm,
        const std::string &modeStr, const std::string &cfgFile)
{
    
    // if this instance has already been initialized, skip
    if (isInitialised) {
        fprintf(stderr,
                "This AmgXSolver instance has been initialized on this process.\n");
        exit(0);
    }

    // increase the number of AmgXSolver instances
    count += 1;

    // get the name of this node
    int     len;
    char    name[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(name, &len);  
    nodeName = name;

    // get the mode of AmgX solver
    setMode(modeStr);  

    // initialize communicators and corresponding information
    initMPIcomms(comm);  

    // only processes in gpuWorld are required to initialize AmgX
    if (gpuProc == 0)
    {
        initAmgX(cfgFile);  
    }

    // a bool indicating if this instance is initialized
    isInitialised = true;

    return;
}

void AmgXSolver::initialiseMatrixComms(
    AmgXCSRMatrix& matrix)
{
    matrix.initialiseComms(devWorld, gpuProc);
}

/* \implements AmgXSolver::setMode */
void AmgXSolver::setMode(const std::string &modeStr)
{
    if (modeStr == "dDDI")
        mode = AMGX_mode_dDDI;
    else if (modeStr == "dDFI")
        mode = AMGX_mode_dDFI;
    else if (modeStr == "dFFI")
        mode = AMGX_mode_dFFI;
    else if (modeStr[0] == 'h') {
        printf("CPU mode, %s, is not supported in this wrapper!",
                modeStr.c_str());
        exit(0);
    }
    else {
        printf("%s is not an available mode! Available modes are: "
                "dDDI, dDFI, dFFI.\n", modeStr.c_str());
        exit(0);
    }
}


/* \implements AmgXSolver::initAmgX */
 void AmgXSolver::initAmgX(const std::string &cfgFile)
{
    // only the first instance (AmgX solver) is in charge of initializing AmgX
    if (count == 1)
    {
        // initialize AmgX
        AMGX_SAFE_CALL(AMGX_initialize());

        // intialize AmgX plugings
        AMGX_SAFE_CALL(AMGX_initialize_plugins());

        // only the master process can output something on the screen
        // AMGX_SAFE_CALL(AMGX_register_print_callback(
        //             [](const char *msg, int length)->void
        //             {PetscPrintf(PETSC_COMM_WORLD, "%s", msg);}));

        // let AmgX to handle errors returned
        AMGX_SAFE_CALL(AMGX_install_signal_handler());
    }

    // create an AmgX configure object
    AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg, cfgFile.c_str()));

    // let AmgX handle returned error codes internally
    AMGX_SAFE_CALL(AMGX_config_add_parameters(&cfg, "exception_handling=1"));

    // create an AmgX resource object, only the first instance is in charge
    if (count == 1) AMGX_resources_create(&rsrc, cfg, &gpuWorld, 1, &devID);

    // create AmgX vector object for unknowns and RHS
    AMGX_vector_create(&AmgXP, rsrc, mode);
    AMGX_vector_create(&AmgXRHS, rsrc, mode);

    // create AmgX matrix object for unknowns and RHS
    AMGX_matrix_create(&AmgXA, rsrc, mode);

    // create an AmgX solver object
    AMGX_solver_create(&solver, rsrc, mode, cfg);

    // obtain the default number of rings based on current configuration
    AMGX_config_get_default_number_of_rings(cfg, &ring);
}

/* \implements AmgXSolver::finalize */
void AmgXSolver::finalize()
{
    // skip if this instance has not been initialised
    if (!isInitialised)
    {
        fprintf(stderr,
                "This AmgXWrapper has not been initialised. "
                "Please initialise it before finalization.\n");
        exit(0);
    }

    // only processes using GPU are required to destroy AmgX content
    if (gpuProc == 0)
    {
        // destroy solver instance
        AMGX_solver_destroy(solver);

        // destroy matrix instance
        AMGX_matrix_destroy(AmgXA);

        // destroy RHS and unknown vectors
        AMGX_vector_destroy(AmgXP);
        AMGX_vector_destroy(AmgXRHS);

        // only the last instance need to destroy resource and finalizing AmgX
        if (count == 1)
        {
            AMGX_resources_destroy(rsrc);
            AMGX_SAFE_CALL(AMGX_config_destroy(cfg));

            AMGX_SAFE_CALL(AMGX_finalize_plugins());
            AMGX_SAFE_CALL(AMGX_finalize());
        }
        else
        {
            AMGX_config_destroy(cfg);
        }

        // destroy gpuWorld
        MPI_Comm_free(&gpuWorld);  
    }

    // re-set necessary variables in case users want to reuse
    // the variable of this instance for a new instance
    gpuProc = MPI_UNDEFINED;
    MPI_Comm_free(&globalCpuWorld);
    MPI_Comm_free(&localCpuWorld);
    MPI_Comm_free(&devWorld);

    // decrease the number of instances
    count -= 1;

    // change status
    isInitialised = false;
}

/* \implements AmgXSolver::setOperator */
void AmgXSolver::setOperator
(
    const int nLocalRows,
    const int nGlobalRows,
    const int nLocalNz,
    AmgXCSRMatrix& matrix
)
{

    // Check the matrix size is not larger than tolerated by AmgX
    if(nGlobalRows > std::numeric_limits<int>::max())
    {
        fprintf(stderr,
                "AmgX does not support a global number of rows greater than "
                "what can be stored in 32 bits (nGlobalRows = %d).\n",
                nGlobalRows);
        exit(0);
    }

    const int nRows = (matrix.isConsolidated()) ? matrix.getNConsRows() : nLocalRows;
    const int nNz = (matrix.isConsolidated()) ? matrix.getNConsNz() : nLocalNz;

    if (nNz > std::numeric_limits<int>::max())
    {
        fprintf(stderr,
                "AmgX does not support non-zeros per (consolidated) rank greater than"
                "what can be stored in 32 bits (nLocalNz = %d).\n",
                nNz);
        exit(0);
    }

    // upload matrix A to AmgX
    if (gpuWorld != MPI_COMM_NULL)
    {
        MPI_Barrier(gpuWorld);  

        AMGX_distribution_handle dist;
        AMGX_distribution_create(&dist, cfg);

        // Must persist until after we call upload
        std::vector<int> offsets(gpuWorldSize + 1, 0);

        // Determine the number of rows per GPU
        std::vector<int> nRowsPerGPU(gpuWorldSize);
        MPI_Allgather(&nRows, 1, MPI_INT, nRowsPerGPU.data(), 1, MPI_INT, gpuWorld);  

        // Calculate the global offsets
        std::partial_sum(nRowsPerGPU.begin(), nRowsPerGPU.end(), offsets.begin() + 1);

        AMGX_distribution_set_partition_data(
            dist, AMGX_DIST_PARTITION_OFFSETS, offsets.data());

        // Set the column indices size, 32- / 64-bit
        AMGX_distribution_set_32bit_colindices(dist, true);

        AMGX_matrix_upload_distributed(
            AmgXA, nGlobalRows, nRows, nNz, 1, 1, matrix.getRowOffsets(),
            matrix.getColIndices(), matrix.getValues(), nullptr, dist);

        AMGX_distribution_destroy(dist);

        // bind the matrix A to the solver
        AMGX_solver_setup(solver, AmgXA);

        // connect (bind) vectors to the matrix
        AMGX_vector_bind(AmgXP, AmgXA);
        AMGX_vector_bind(AmgXRHS, AmgXA);
    }

    MPI_Barrier(globalCpuWorld);  
}


/* \implements AmgXSolver::updateOperator */
void AmgXSolver::updateOperator
(
    const int nLocalRows,
    const int nLocalNz,
    AmgXCSRMatrix& matrix
)
{
    const int nRows = (matrix.isConsolidated()) ? matrix.getNConsRows() : nLocalRows;
    const int nNz = (matrix.isConsolidated()) ? matrix.getNConsNz() : nLocalNz;

    // Replace the coefficients for the CSR matrix A within AmgX
    if (gpuWorld != MPI_COMM_NULL)
    {
        AMGX_matrix_replace_coefficients(AmgXA, nRows, nNz, matrix.getValues(), nullptr);

        // Re-setup the solver (a reduced overhead setup that accounts for consistent matrix structure)
        AMGX_solver_resetup(solver, AmgXA);
    }

    MPI_Barrier(globalCpuWorld);
}

/* \implements AmgXSolver::solve */
// void AmgXSolver::solve(
//     int nLocalRows, Vec& p, Vec& b, AmgXCSRMatrix& matrix)
// {
//     double* pscalar;
//     double* bscalar;

//     // get pointers to the raw data of local vectors
//     VecGetArray(p, &pscalar);
//     VecGetArray(b, &bscalar);

//     solve(nLocalRows, pscalar, bscalar, matrix);

//     VecRestoreArray(p, &pscalar);
//     VecRestoreArray(b, &bscalar);
// }


/* \implements AmgXSolver::solve */
void AmgXSolver::solve(
    int nLocalRows, double* pscalar, const double* bscalar, AmgXCSRMatrix& matrix)
{
    double* p;
    const double* b;
    int nRows;

    if (matrix.isConsolidated())
    {
        p = matrix.getPCons();
        b = matrix.getRHSCons();

        const int* rowDispls = matrix.getRowDispls();
        CHECK(cudaMemcpy((void **)&p[rowDispls[myDevWorldRank]], pscalar, sizeof(double) * nLocalRows, cudaMemcpyDefault));
        CHECK(cudaMemcpy((void **)&b[rowDispls[myDevWorldRank]], bscalar, sizeof(double) * nLocalRows, cudaMemcpyDefault));

        // Override the number of rows as the consolidated number of rows
        nRows = matrix.getNConsRows();

        // Sync as cudaMemcpy to IPC buffers so device to device copies, which are non-blocking w.r.t host
        // All ranks in devWorld have the same value for isConsolidated
        CHECK(cudaDeviceSynchronize());
        MPI_Barrier(devWorld);  
    }
    else
    {
        p = pscalar;
        b = bscalar;
        nRows = nLocalRows;
    }

    if (gpuWorld != MPI_COMM_NULL)
    {
        // Upload potentially consolidated vectors to AmgX
        AMGX_vector_upload(AmgXP, nRows, 1, p);
        AMGX_vector_upload(AmgXRHS, nRows, 1, b);

        MPI_Barrier(gpuWorld);  

        // Solve
        AMGX_solver_solve(solver, AmgXRHS, AmgXP);

        // Get the status of the solver
        AMGX_SOLVE_STATUS status;
        AMGX_solver_get_status(solver, &status);

        // Check whether the solver successfully solved the problem
        if (status != AMGX_SOLVE_SUCCESS)
        {
            fprintf(stderr, "AmgX solver failed to solve the system! "
                            "The error code is %d.\n",
                    status);
        }

        // Download data from device
        AMGX_vector_download(AmgXP, p);

        if(matrix.isConsolidated())
        {
            // AMGX_vector_download invokes a device to device copy, so it is essential that
            // the root rank blocks the host before other ranks copy from the consolidated solution
            CHECK(cudaDeviceSynchronize());
        }
    }

    // If the matrix is consolidated, scatter the solution
    if (matrix.isConsolidated())
    {
        // Must synchronise before each rank attempts to read from the consolidated solution
        MPI_Barrier(devWorld);  

        const int* rowDispls = matrix.getRowDispls();

        // Ranks copy the portion of the solution they own into their rank-local buffers
        CHECK(cudaMemcpy((void **)pscalar, &p[rowDispls[myDevWorldRank]], sizeof(double) * nLocalRows, cudaMemcpyDefault));

        // Sync as cudaMemcpy to IPC buffers so device to device copies, which are non-blocking w.r.t host
        // All ranks in devWorld have the same value for isConsolidated
        CHECK(cudaDeviceSynchronize());
    }

    MPI_Barrier(globalCpuWorld);  
}


/* \implements AmgXSolver::getIters */
void AmgXSolver::getIters(int &iter)
{
    // only processes using AmgX will try to get # of iterations
    if (gpuProc == 0)
        AMGX_solver_get_iterations_number(solver, &iter);
}


/* \implements AmgXSolver::getResidual */
void AmgXSolver::getResidual(const int &iter, double &res)
{
    // only processes using AmgX will try to get residual
    if (gpuProc == 0)
        AMGX_solver_get_iteration_residual(solver, iter, 0, &res);
}

