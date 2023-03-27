/**
 * \file AmgXMPIComms.cu
 * \brief ***.
 * \author Pi-Yueh Chuang (pychuang@gwu.edu)
 * \author Matt Martineau (mmartineau@nvidia.com)
 * \date 2015-09-01
 * \copyright Copyright (c) 2015-2019 Pi-Yueh Chuang, Lorena A. Barba.
 * \copyright Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
 *            This project is released under MIT License.
 */

// AmgXWrapper
# include "AmgXSolver.H"


/* \implements AmgXSolver::initMPIcomms */
void AmgXSolver::initMPIcomms(const MPI_Comm &comm)
{
    // duplicate the global communicator
    MPI_Comm_dup(comm, &globalCpuWorld);  
    MPI_Comm_set_name(globalCpuWorld, "globalCpuWorld");  

    // get size and rank for global communicator
    MPI_Comm_size(globalCpuWorld, &globalSize);  
    MPI_Comm_rank(globalCpuWorld, &myGlobalRank);  


    // Get the communicator for processors on the same node (local world)
    MPI_Comm_split_type(globalCpuWorld,
            MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localCpuWorld);  
    MPI_Comm_set_name(localCpuWorld, "localCpuWorld");  

    // get size and rank for local communicator
    MPI_Comm_size(localCpuWorld, &localSize);  
    MPI_Comm_rank(localCpuWorld, &myLocalRank);  


    // set up the variable nDevs
    setDeviceCount();  


    // set up corresponding ID of the device used by each local process
    setDeviceIDs();  
    MPI_Barrier(globalCpuWorld);  


    // split the global world into a world involved in AmgX and a null world
    MPI_Comm_split(globalCpuWorld, gpuProc, 0, &gpuWorld);  

    // get size and rank for the communicator corresponding to gpuWorld
    if (gpuWorld != MPI_COMM_NULL)
    {
        MPI_Comm_set_name(gpuWorld, "gpuWorld");  
        MPI_Comm_size(gpuWorld, &gpuWorldSize);  
        MPI_Comm_rank(gpuWorld, &myGpuWorldRank);  
    }
    else // for those can not communicate with GPU devices
    {
        gpuWorldSize = MPI_UNDEFINED;
        myGpuWorldRank = MPI_UNDEFINED;
    }

    // split local world into worlds corresponding to each CUDA device
    MPI_Comm_split(localCpuWorld, devID, 0, &devWorld);  
    MPI_Comm_set_name(devWorld, "devWorld");  

    // get size and rank for the communicator corresponding to myWorld
    MPI_Comm_size(devWorld, &devWorldSize);  
    MPI_Comm_rank(devWorld, &myDevWorldRank);  

    MPI_Barrier(globalCpuWorld);  
}


/* \implements AmgXSolver::setDeviceCount */
void AmgXSolver::setDeviceCount()
{
    // get the number of devices that AmgX solvers can use
    switch (mode)
    {
        case AMGX_mode_dDDI: // for GPU cases, nDevs is the # of local GPUs
        case AMGX_mode_dDFI: // for GPU cases, nDevs is the # of local GPUs
        case AMGX_mode_dFFI: // for GPU cases, nDevs is the # of local GPUs
            // get the number of total cuda devices
            CHECK(cudaGetDeviceCount(&nDevs));
            if (myLocalRank == 0) printf("Number of GPU devices :: %d \n", nDevs);

            // Check whether there is at least one CUDA device on this node
            if (nDevs == 0) {
                printf("There is no CUDA device on the node %s !\n", nodeName.c_str());
                exit(0);
            }
            break;
        case AMGX_mode_hDDI: // for CPU cases, nDevs is the # of local processes
        case AMGX_mode_hDFI: // for CPU cases, nDevs is the # of local processes
        case AMGX_mode_hFFI: // for CPU cases, nDevs is the # of local processes
        default:
            nDevs = localSize;
            break;
    }
}


/* \implements AmgXSolver::setDeviceIDs */
void AmgXSolver::setDeviceIDs()
{
    // set the ID of device that each local process will use
    if (nDevs == localSize) // # of the devices and local precosses are the same
    {
        devID = myLocalRank;
        gpuProc = 0;
    }
    else if (nDevs > localSize) // there are more devices than processes
    {
        if (myLocalRank == 0) printf("CUDA devices on the node %s "
                "are more than the MPI processes launched. Only %d CUDA "
                "devices will be used.\n", nodeName.c_str(), localSize); 

        devID = myLocalRank;
        gpuProc = 0;
    }
    else // there more processes than devices
    {
        int     nBasic = localSize / nDevs,
                nRemain = localSize % nDevs;

        if (myLocalRank < (nBasic+1)*nRemain)
        {
            devID = myLocalRank / (nBasic + 1);
            if (myLocalRank % (nBasic + 1) == 0)  gpuProc = 0;
        }
        else
        {
            devID = (myLocalRank - (nBasic+1)*nRemain) / nBasic + nRemain;
            if ((myLocalRank - (nBasic+1)*nRemain) % nBasic == 0) gpuProc = 0;
        }
    }

    // Set the device for each rank
    cudaSetDevice(devID);
}

