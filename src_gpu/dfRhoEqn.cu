#include "dfRhoEqn.H"

void dfRhoEqn::createNonConstantLduAndCsrFields()
{
#ifndef STREAM_ALLOCATOR
    checkCudaErrors(cudaMalloc((void**)&d_source, dataBase_.cell_value_bytes));
    checkCudaErrors(cudaMalloc((void**)&d_diag, dataBase_.cell_value_bytes));
    DEBUG_TRACE;
#endif
}

void dfRhoEqn::setConstantValues() {
    this->stream = dataBase_.stream;
}

void dfRhoEqn::setConstantFields(const std::vector<int> patch_type) {
    this->patch_type = patch_type;
}

void dfRhoEqn::initNonConstantFields(const double *rho, const double *phi, 
            const double *boundary_rho, const double *boundary_phi) {
    checkCudaErrors(cudaMemcpy(dataBase_.d_rho, rho, dataBase_.cell_value_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dataBase_.d_phi, phi, dataBase_.surface_value_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dataBase_.d_boundary_rho, boundary_rho, dataBase_.boundary_surface_value_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dataBase_.d_boundary_phi, boundary_phi, dataBase_.boundary_surface_value_bytes, cudaMemcpyHostToDevice));
}

void dfRhoEqn::cleanCudaResources() {
#ifdef USE_GRAPH
    if (graph_created) {
        checkCudaErrors(cudaGraphExecDestroy(graph_instance));
        checkCudaErrors(cudaGraphDestroy(graph));
    }
#endif
}

void dfRhoEqn::preProcess()
{
}

void dfRhoEqn::process()
{
    TICK_INIT_EVENT;
    TICK_START_EVENT;
#ifdef USE_GRAPH
    if(!graph_created) {
        DEBUG_TRACE;
        checkCudaErrors(cudaStreamBeginCapture(dataBase_.stream, cudaStreamCaptureModeGlobal));
#endif

#ifdef STREAM_ALLOCATOR
    checkCudaErrors(cudaMallocAsync((void**)&d_source, dataBase_.cell_value_bytes, dataBase_.stream));
    checkCudaErrors(cudaMallocAsync((void**)&d_diag, dataBase_.cell_value_bytes, dataBase_.stream));
#endif
    // checkCudaErrors(cudaMemcpyAsync(dataBase_.d_phi, dataBase_.h_phi, dataBase_.surface_value_bytes, cudaMemcpyHostToDevice, dataBase_.stream));
    // checkCudaErrors(cudaMemcpyAsync(dataBase_.d_boundary_phi, dataBase_.h_boundary_phi, dataBase_.boundary_surface_value_bytes, cudaMemcpyHostToDevice, dataBase_.stream));
    // checkCudaErrors(cudaMemcpyAsync(dataBase_.d_boundary_rho, dataBase_.h_boundary_rho, dataBase_.boundary_surface_value_bytes, cudaMemcpyHostToDevice, dataBase_.stream));

    checkCudaErrors(cudaMemsetAsync(d_diag, 0, dataBase_.cell_value_bytes, dataBase_.stream));
    checkCudaErrors(cudaMemsetAsync(d_source, 0, dataBase_.cell_value_bytes, dataBase_.stream));

    fvm_ddt_scalar(dataBase_.stream, dataBase_.num_cells, dataBase_.rdelta_t, dataBase_.d_rho_old, dataBase_.d_volume, 
            d_diag, d_source);
    fvc_div_surface_scalar(dataBase_.stream, dataBase_.num_cells, dataBase_.num_surfaces, dataBase_.num_boundary_surfaces, dataBase_.d_owner,
            dataBase_.d_neighbor, dataBase_.d_phi, dataBase_.d_boundary_face_cell, 
            dataBase_.num_patches, dataBase_.patch_size.data(), patch_type.data(),
            dataBase_.d_boundary_phi, dataBase_.d_volume, d_source, -1);
    solve();
    correct_boundary_conditions_scalar(dataBase_.stream, dataBase_.nccl_comm, dataBase_.neighbProcNo.data(),
            dataBase_.num_boundary_surfaces, dataBase_.num_patches, dataBase_.patch_size.data(),
            patch_type.data(), dataBase_.d_boundary_delta_coeffs, dataBase_.d_boundary_face_cell, dataBase_.d_rho, dataBase_.d_boundary_rho,
            dataBase_.cyclicNeighbor.data(), dataBase_.patchSizeOffset.data(), dataBase_.d_boundary_weight);
#ifdef USE_GRAPH
        checkCudaErrors(cudaStreamEndCapture(dataBase_.stream, &graph));
        checkCudaErrors(cudaGraphInstantiate(&graph_instance, graph, NULL, NULL, 0));
        graph_created = true;
    }
    checkCudaErrors(cudaGraphLaunch(graph_instance, dataBase_.stream));
#endif
    TICK_END_EVENT(rhoEqn process);

    TICK_START_EVENT;
#ifdef STREAM_ALLOCATOR
    checkCudaErrors(cudaFreeAsync(d_source, dataBase_.stream));
    checkCudaErrors(cudaFreeAsync(d_diag, dataBase_.stream));
#endif
    TICK_END_EVENT(rhoEqn post process free);
    TICK_START_EVENT;
    // checkCudaErrors(cudaMemcpyAsync(dataBase_.h_rho, dataBase_.d_rho, dataBase_.cell_value_bytes, cudaMemcpyDeviceToHost, dataBase_.stream));
    TICK_END_EVENT(rhoEqn post process copy back);
    sync();
}

void dfRhoEqn::sync()
{
    checkCudaErrors(cudaStreamSynchronize(dataBase_.stream));
}

void dfRhoEqn::solve()
{
    solve_explicit_scalar(dataBase_.stream, dataBase_.num_cells, d_diag, d_source, dataBase_.d_rho);
}

void dfRhoEqn::postProcess(double *h_rho) {}

void dfRhoEqn::compareResult(const double *diag, const double *source, bool printFlag)
{
    std::vector<double> h_diag;
    h_diag.resize(dataBase_.num_cells);
    checkCudaErrors(cudaMemcpy(h_diag.data(), d_diag, dataBase_.cell_value_bytes, cudaMemcpyDeviceToHost));
    fprintf(stderr, "check h_diag\n");
    checkVectorEqual(dataBase_.num_cells, diag, h_diag.data(), 1e-14, printFlag);
    DEBUG_TRACE;

    std::vector<double> h_source;
    h_source.resize(dataBase_.num_cells);
    checkCudaErrors(cudaMemcpy(h_source.data(), d_source, dataBase_.cell_value_bytes, cudaMemcpyDeviceToHost));
    fprintf(stderr, "check h_source\n");
    checkVectorEqual(dataBase_.num_cells, source, h_source.data(), 1e-14, printFlag);
    DEBUG_TRACE;
}

void dfRhoEqn::compareRho(const double *rho, const double *boundary_rho, bool printFlag)
{
    std::vector<double> h_rho;
    h_rho.resize(dataBase_.num_cells);
    checkCudaErrors(cudaMemcpy(h_rho.data(), dataBase_.d_rho, dataBase_.cell_value_bytes, cudaMemcpyDeviceToHost));
    fprintf(stderr, "check h_rho\n");
    checkVectorEqual(dataBase_.num_cells, rho, h_rho.data(), 1e-14, printFlag);
    DEBUG_TRACE;

    std::vector<double> h_boundary_rho;
    h_boundary_rho.resize(dataBase_.num_boundary_surfaces);
    checkCudaErrors(cudaMemcpy(h_boundary_rho.data(), dataBase_.d_boundary_rho, dataBase_.boundary_surface_value_bytes, cudaMemcpyDeviceToHost));
    fprintf(stderr, "check h_boundary_rho\n");
    checkVectorEqual(dataBase_.num_boundary_surfaces, boundary_rho, h_boundary_rho.data(), 1e-14, printFlag);
    DEBUG_TRACE;
}

void dfRhoEqn::correctPsi(const double *rho, const double *boundary_rho)
{
    checkCudaErrors(cudaMemcpy(dataBase_.d_rho, rho, dataBase_.cell_value_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dataBase_.d_boundary_rho, boundary_rho, dataBase_.boundary_surface_value_bytes, cudaMemcpyHostToDevice));
}

