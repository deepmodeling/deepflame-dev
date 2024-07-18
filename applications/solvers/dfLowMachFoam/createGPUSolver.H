#include <queue>
#include "CSRGAMGAgglomeration.H"
#include "ELLGAMGAgglomeration.H"

#define PARALLEL_
#define iscsr

ncclUniqueId nccl_id;
ncclComm_t nccl_comm;
int nRanks, myRank, localRank, mpi_init_flag = 0;

dfMatrixDataBase dfDataBase;

/* For DeepFlame_Academic */
void createGPUSolverInput(mesh_info_para &mesh_paras, const lduMesh& mesh, const dictionary solverControls, std::vector<int> patch_type, MATRIX_EQUATION EqnID);
void createGPUGAMGInput(const lduMesh& mesh, volScalarField& eqn);
void createMatrixNoDiag();
void createGAMGMatrixNoDiag();


void permuteVector(double *new_order, const double *old_order, int num) 
{
    for (int i = 0; i < num; i++) 
    {
        new_order[num * 0 + i] = old_order[i * 3 + 0];
        new_order[num * 1 + i] = old_order[i * 3 + 1];
        new_order[num * 2 + i] = old_order[i * 3 + 2];
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

#if defined(DEBUG_)
template <typename T>
void getTypeInfo(size_t *stride, size_t *internal_size, size_t *boundary_size) {
    size_t s = 1;
    bool isVol = false;
    if (typeid(T) == typeid(surfaceScalarField)) {
        s = 1;
        isVol = false;
    } else if (typeid(T) == typeid(surfaceVectorField)) {
        s = 3;
        isVol = false;
    } else if (typeid(T) == typeid(surfaceTensorField)) {
        s = 9;
        isVol = false;
    } else if (typeid(T) == typeid(volScalarField)) {
        s = 1;
        isVol = true;
    } else if (typeid(T) == typeid(volVectorField)) {
        s = 3;
        isVol = true;
    } else if (typeid(T) == typeid(volTensorField)) {
        s = 9;
        isVol = true;
    } else {
        fprintf(stderr, "ERROR! Unsupported field type()!\n");
        exit(EXIT_FAILURE);
    }
    *stride = s;
    *internal_size = (isVol ? dfDataBase.num_cells : dfDataBase.num_surfaces) * s;
    *boundary_size = dfDataBase.num_boundary_surfaces * s;
}

template <typename T>
void getFieldPtr(std::queue<double*>& fieldPtrQue, T& field){
    fieldPtrQue.push(&field[0]);
    forAll(field.boundaryField(), patchi){
        auto& patchField = field.boundaryFieldRef()[patchi];
        fieldPtrQue.push(&patchField[0]);
    }
};

template <typename T>
void randomInitField(T& field) {
    size_t stride = 0;
    size_t internal_size = 0;
    size_t boundary_size = 0;
    getTypeInfo<T>(&stride, &internal_size, &boundary_size);
    size_t internal_value_bytes = internal_size * sizeof(double) * stride;
    std::queue<double*> fieldPtrQue;
    // std::vector<double*> fieldPtrQue;
    getFieldPtr(fieldPtrQue, field);

    // random init field value to (-0.5, 0.5)
    // internal
    double *&field_internal_ptr = fieldPtrQue.front(); fieldPtrQue.pop();
    // double *field_internal_ptr = fieldPtrQue[0];
    std::vector<double> init_field_internal;
    init_field_internal.resize(internal_size * stride);
    for (size_t i = 0; i < internal_size * stride; i++) {
        init_field_internal[i] = (rand() % 10000 - 5000) / 10000.0;
    }
    memcpy(field_internal_ptr, init_field_internal.data(), internal_value_bytes);
    // boundary
    int ptrIndex = 1;
    forAll(field.boundaryField(), patchi)
    {
        auto& patchField = field.boundaryFieldRef()[patchi];
        size_t patchsize = patchField.size();
        double *&field_boundary_ptr = fieldPtrQue.front(); fieldPtrQue.pop();
        // double *field_boundary_ptr = fieldPtrQue[ptrIndex];
        // ptrIndex ++;
        std::vector<double> init_field_boundary;
        init_field_boundary.resize(patchsize * stride);
        for (size_t i = 0; i < patchsize * stride; i++) {
            init_field_boundary[i] = (rand() % 10000 - 5000) / 10000.0;
        }
        memcpy(field_boundary_ptr, init_field_boundary.data(), patchsize * stride * sizeof(double));
    }

    field.correctBoundaryConditions();
}
#endif

void initNccl() {
    ncclInit(PstreamGlobals::MPI_COMM_FOAM, nccl_comm, nccl_id, &nRanks, &myRank, &localRank, &mpi_init_flag);
}

void createGPUBase(const IOdictionary& CanteraTorchProperties, fvMesh& mesh, PtrList<volScalarField>& Y) {
    // prepare constant values: num_cells, num_surfaces, num_boundary_surfaces,
    // num_patches, patch_size, num_species, rdelta_t
    const labelUList& owner = mesh.owner();
    const labelUList& neighbour = mesh.neighbour();
    int num_cells = mesh.nCells();
    int num_total_cells = Foam::returnReduce(num_cells, sumOp<label>());
    int num_surfaces = neighbour.size();
    int num_boundary_surfaces = 0;
    int num_patches = 0;
    std::vector<int> patch_size;
    forAll(mesh.boundary(), patchi) {
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
    }
    // prepare interface info
    const globalIndex globalNumbering(num_cells);
    const lduInterfacePtrsList interfaces(mesh.interfaces());
    const lduAddressing& lduAddr = mesh.lduAddr();
    labelList globalCells(num_cells);
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
    // TODO: get deltaT fomr time API
    double rDeltaT = 1 / 1e-6;
    dfDataBase.setConstantValues(num_cells, num_total_cells, num_surfaces, num_boundary_surfaces, 
            num_patches, nProcValues, patch_size, Y.size(), rDeltaT);

    // wyr: now there is no other place to prepare nccl info, thus mpi must be initialized at beginning.
    label flag_mpi_init;
    MPI_Initialized(&flag_mpi_init);
    if(flag_mpi_init) {
        std::vector<int> GPUNeighbProcNo(dfDataBase.num_patches, -1);
        // get basic communication info from of
        forAll(Y[0].boundaryField(), patchi) {
            if (Y[0].boundaryField()[patchi].type() == "processor"
                || Y[0].boundaryField()[patchi].type() == "processorCyclic") {
                GPUNeighbProcNo[patchi] = dynamic_cast<const processorFvPatchField<scalar>&>(Y[0].boundaryField()[patchi]).neighbProcNo();
            }
        }
        // prepare mpi and nccl info
        dfDataBase.setCommInfo(PstreamGlobals::MPI_COMM_FOAM, nccl_comm, nccl_id, nRanks, myRank, localRank, GPUNeighbProcNo);
    }

    // get cyclic neighbor when has cyclic patch
    // - get boundary Index
    std::map<std::string, int> patchNameToIndex;
    forAll(Y[0].boundaryField(), patchi) {
        patchNameToIndex[Y[0].boundaryField()[patchi].patch().name()] = patchi;
    }
    // - get cyclic neighbor
    std::vector<int> cyclicNeighbor(dfDataBase.num_patches, -1);
    forAll(Y[0].boundaryField(), patchi) {
        if (Y[0].boundaryField()[patchi].type() == "cyclic") {
            cyclicNeighbor[patchi] = patchNameToIndex[dynamic_cast<const cyclicFvPatchField<scalar>&>(Y[0].boundaryField()[patchi]).
                    cyclicPatch().cyclicPatch().neighbPatch().name()];
        }
    }
    dfDataBase.setCyclicInfo(cyclicNeighbor);

    // prepare cuda resources
    dfDataBase.prepareCudaResources();

    // setup amgx solvers
    string mode_string = "dDDI";
    string u_setting_path;
    u_setting_path = CanteraTorchProperties.subDict("AmgxSettings").lookupOrDefault("UEqnSettingPath", string(""));
    string p_setting_path;
    p_setting_path = CanteraTorchProperties.subDict("AmgxSettings").lookupOrDefault("pEqnSettingPath", string(""));
    dfDataBase.setAmgxSolvers(mode_string, u_setting_path, p_setting_path);

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
    dfDataBase.setConstantIndexes(&owner[0], &neighbour[0], &procRows[0], &procCols[0], globalOffset);

    // prepare internal and boundary of sf, mag_sf, weights, delta_coeffs, volume
    double *boundary_sf = new double[3 * num_boundary_surfaces];
    double *boundary_mag_sf = new double[num_boundary_surfaces];
    double *boundary_delta_coeffs = new double[num_boundary_surfaces];
    double *boundary_weights = new double[num_boundary_surfaces];
    int *boundary_face_cell = new int[num_boundary_surfaces];
    std::vector<int> patch_type_calculated(num_patches, 5); // default patch type is calculated
    std::vector<int> patch_type_extropolated(num_patches, 8); // default patch type is extrapolated

    int offset = 0;
    forAll(mesh.boundary(), patchi) {
        const fvPatchScalarField& patchY = Y[0].boundaryField()[patchi];
        const vectorField& pSf = mesh.Sf().boundaryField()[patchi];
        const scalarField& pMagSf = mesh.magSf().boundaryField()[patchi];
        const scalarField& pDeltaCoeffs = mesh.deltaCoeffs().boundaryField()[patchi];
        const scalarField& pWeights = mesh.surfaceInterpolation::weights().boundaryField()[patchi];
        const labelUList& pFaceCells = mesh.boundary()[patchi].faceCells();

        int patchsize = pMagSf.size();

        if (patchY.type() == "processor") {
            patch_type_calculated[patchi] = 7; // patchi type is processor
            patch_type_extropolated[patchi] = 7; // patchi type is processor

            memcpy(boundary_sf + 3*offset, &pSf[0][0], 3*patchsize*sizeof(double));
            memcpy(boundary_sf + 3*offset + 3*patchsize, &pSf[0][0], 3*patchsize*sizeof(double));

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
            patch_type_calculated[patchi] = 10; // patchi type is processorCyclic
            patch_type_extropolated[patchi] = 10; // patchi type is processorCyclic

            memcpy(boundary_sf + 3*offset, &pSf[0][0], 3*patchsize*sizeof(double));
            memcpy(boundary_sf + 3*offset + 3*patchsize, &pSf[0][0], 3*patchsize*sizeof(double));

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
            patch_type_calculated[patchi] = 6; // patchi type is cyclic
            patch_type_extropolated[patchi] = 6; // patchi type is cyclic

            memcpy(boundary_sf + 3*offset, &pSf[0][0], 3*patchsize*sizeof(double));
            memcpy(boundary_mag_sf + offset, &pMagSf[0], patchsize*sizeof(double));
            memcpy(boundary_delta_coeffs + offset, &pDeltaCoeffs[0], patchsize*sizeof(double));
            memcpy(boundary_weights + offset, &pWeights[0], patchsize*sizeof(double));
            memcpy(boundary_face_cell + offset, &pFaceCells[0], patchsize * sizeof(int));

            offset += patchsize;
        } else {
            memcpy(boundary_sf + 3*offset, &pSf[0][0], 3*patchsize*sizeof(double));
            memcpy(boundary_mag_sf + offset, &pMagSf[0], patchsize*sizeof(double));
            memcpy(boundary_delta_coeffs + offset, &pDeltaCoeffs[0], patchsize*sizeof(double));
            memcpy(boundary_weights + offset, &pWeights[0], patchsize*sizeof(double));
            memcpy(boundary_face_cell + offset, &pFaceCells[0], patchsize * sizeof(int));

            offset += patchsize;
        }
    }

    dfDataBase.createConstantFieldsInternal();
    dfDataBase.createConstantFieldsBoundary();

    // construct mesh distance for limitedLinear scheme
    vectorField meshDistance = mesh.Sf();
    forAll(meshDistance, facei) {
        label own = owner[facei];
        label nei = neighbour[facei];
        meshDistance[facei] = mesh.C()[nei] - mesh.C()[own];
    }

    dfDataBase.initConstantFieldsInternal(&mesh.Sf()[0][0], &mesh.magSf()[0], &mesh.surfaceInterpolation::weights()[0], 
            &mesh.deltaCoeffs()[0], &mesh.V()[0], &meshDistance[0][0]);
    dfDataBase.initConstantFieldsBoundary(boundary_sf, boundary_mag_sf, boundary_delta_coeffs, boundary_weights, boundary_face_cell, 
            patch_type_calculated, patch_type_extropolated);
    
    dfDataBase.createNonConstantFieldsInternal();
    dfDataBase.createNonConstantFieldsBoundary();

    delete boundary_sf;
    delete boundary_mag_sf;
    delete boundary_delta_coeffs;
    delete boundary_weights;
    delete boundary_face_cell;
}

void createGPURhoEqn(mesh_info_para &mesh_paras, init_data_para &init_data, const volScalarField& rho, const surfaceScalarField& phi) {
    DEBUG_TRACE;
    std::vector<int> patch_type;
    patch_type.resize(mesh_paras.num_patches);

    double h_rho[mesh_paras.num_cells];
    double h_phi[mesh_paras.num_surfaces];
    double h_boundary_phi[mesh_paras.num_boundary_surfaces];
    double h_boundary_rho[mesh_paras.num_boundary_surfaces];
    memcpy(h_phi, &phi[0], mesh_paras.num_surfaces*sizeof(double));
    memcpy(h_rho, &rho[0], mesh_paras.num_cells*sizeof(double));

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

    init_data.rho = &h_rho[0];
    init_data.phi = &h_phi[0];
    init_data.boundary_rho = &h_boundary_rho[0];
    init_data.boundary_phi = &h_boundary_phi[0];

    /* For DeepFlame_Academic */
    const dictionary solverControls = rho.mesh().solverDict(
        rho.select(rho.mesh().data::template lookupOrDefault<bool>("finalIteration", false))); 
 
    createGPUSolverInput(mesh_paras, rho.mesh(), solverControls, patch_type, MATRIX_EQUATION::rhoEqn); 

    initialize_matrix(MATRIX_EQUATION::rhoEqn);
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
    double h_boundary_u[mesh_paras.num_boundary_surfaces*3];
    
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

    double u_newOrder[mesh_paras.num_cells*3];
    double boundary_u_newOrder[mesh_paras.num_boundary_surfaces*3];
    permuteVector(u_newOrder, &U[0][0], mesh_paras.num_cells);
    permuteVector(boundary_u_newOrder, h_boundary_u, mesh_paras.num_boundary_surfaces);

    init_data.u = &u_newOrder[0];
    init_data.boundary_u = &boundary_u_newOrder[0];

    /* For DeepFlame_Academic */
    const dictionary solverControls = U.mesh().solverDict(
        U.select(U.mesh().data::template lookupOrDefault<bool>("finalIteration", false)));
    createGPUSolverInput(mesh_paras, U.mesh(), solverControls, patch_type, MATRIX_EQUATION::UEqn); 
    initialize_matrix(MATRIX_EQUATION::UEqn);    
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
    fprintf(stderr, "num_species: %d\n", init_data.num_species);
    forAll(Y[0].boundaryField(), patchi)
    {
        constructBoundarySelectorPerPatch(&(patch_type[patchi]), Y[0].boundaryField()[patchi].type());
    }
    // set lewis number
    std::vector<double> lewis_number(init_data.num_species, 1.); // unity lewis
    double h_y[init_data.num_species*mesh_paras.num_cells];
    double h_boundary_y[init_data.num_species*mesh_paras.num_boundary_surfaces];

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

    init_data.y = &h_y[0];
    init_data.boundary_y = &h_boundary_y[0];
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
    // EEqn_GPU.setConstantFields(patch_type_he, patch_type_k);

    double h_he[mesh_paras.num_cells];
    double h_boundary_he[mesh_paras.num_boundary_surfaces];
    memcpy(h_he, &he[0], mesh_paras.num_cells*sizeof(double));
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

    init_data.he = &h_he[0];
    init_data.boundary_he = &h_boundary_he[0];
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

    double h_p[mesh_paras.num_cells];
    double h_boundary_p[mesh_paras.num_boundary_surfaces];
    memcpy(h_p, &p[0], mesh_paras.num_cells*sizeof(double));

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
    }
    // pEqn_GPU.setConstantFields(patch_type_U, patch_type_p);

    init_data.p = &h_p[0];
    init_data.boundary_p = &h_boundary_p[0];

    /* For DeepFlame_Academic */
    const dictionary solverControls = p.mesh().solverDict(
        p.select(p.mesh().data::template lookupOrDefault<bool>("finalIteration", false))); 
 
    createGPUSolverInput(mesh_paras, p.mesh(), solverControls, patch_type_p, MATRIX_EQUATION::pEqn); 

    initialize_matrix(MATRIX_EQUATION::pEqn);
}

void createGPUThermo(mesh_info_para &mesh_paras, init_data_para &init_data, const IOdictionary& CanteraTorchProperties, volScalarField& T, volScalarField& he, 
        const volScalarField& psi, const volScalarField& alpha, const volScalarField& mu,
        const volScalarField& K, const volScalarField& dpdt, dfChemistryModel<basicThermo>* chemistry) {
    DEBUG_TRACE;
    // initialize dfThermo
    string mechanismFile;
    mechanismFile = CanteraTorchProperties.lookupOrDefault("CanteraMechanismFile", string(""));

    // thermo_GPU.setConstantValue(mechanismFile, dfDataBase.num_cells, dfDataBase.num_species);
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
    double h_T[mesh_paras.num_cells];
    memcpy(h_T, &T[0], mesh_paras.num_cells*sizeof(double));

    // thermo_GPU.setConstantFields(patch_type_T);

    init_data.T = &h_T[0];
    init_data.thermo_psi = &psi[0];
    init_data.thermo_alpha = &alpha[0];
    init_data.mu = &mu[0];
    init_data.k = &K[0];
    init_data.dpdt = &dpdt[0];
    init_data.thermo_rhoD = &h_thermo_rhoD[0];

    init_data.boundary_T = &h_boundary_T[0];
    init_data.boundary_thermo_psi = &h_boundary_thermo_psi[0];
    init_data.boundary_thermo_alpha = &h_boundary_thermo_alpha[0];
    init_data.boundary_mu = &h_boundary_mu[0];
    init_data.boundary_k = &h_boundary_k[0];
    init_data.boundary_thermo_rhoD = &h_boundary_thermo_rhoD[0];

    delete h_boundary_T;
    delete h_boundary_he;
    delete h_boundary_thermo_psi;
    delete h_boundary_thermo_alpha;
    delete h_boundary_mu;
    delete h_boundary_k;
    delete h_boundary_thermo_rhoD;
    delete h_thermo_rhoD;
}

/*=========================================================================*/
/*                          For DeepFlame_Academic                         */
/*                                                                         */
void createGPUGAMGInput(const lduMesh& mesh, const dictionary solverControls)
{
    std::cout << "========================================================" << std::endl;
    std::cout << "   Set GAMG variables for DeepFlame_Academic " << std::endl;    

    // read control files
    word preconditioner, smoother;
    int nVcycles, nCellsInCoarsestLevel;
    int nPreSweeps, nPostSweeps, nFinestSweeps;
    bool interpolateCorrection, scaleCorrection, solveCoarsest;
    const entry& eP = solverControls.lookupEntry("preconditioner", false, false);
    if (eP.isDict())
    {
        eP.dict().lookup("preconditioner") >> preconditioner;
        eP.dict().lookup("smoother") >> smoother;

        eP.dict().lookup("nVcycles") >> nVcycles;
        eP.dict().lookup("nCellsInCoarsestLevel") >> nCellsInCoarsestLevel;

        eP.dict().lookup("nPreSweeps") >> nPreSweeps;
        eP.dict().lookup("nPostSweeps") >> nPostSweeps;
        eP.dict().lookup("nFinestSweeps") >> nFinestSweeps;

        eP.dict().lookup("interpolateCorrection") >> interpolateCorrection;
        eP.dict().lookup("scaleCorrection") >> scaleCorrection;
        eP.dict().lookup("solveCoarsest") >> solveCoarsest;
    }
    else
    {
        eP.stream() >> preconditioner;
    }
    std::cout << "=== GAMG config preconditioner: " << preconditioner << std::endl;
    std::cout << "=== GAMG config smoother: " << smoother << std::endl;

    std::cout << "=== GAMG config nVcycles: " << nVcycles << std::endl; 
    std::cout << "=== GAMG config nCellsInCoarsestLevel: " << nCellsInCoarsestLevel << std::endl; 
    std::cout << "=== GAMG config nPreSweeps: " << nPreSweeps << std::endl; 
    std::cout << "=== GAMG config nPostSweeps: " << nPostSweeps << std::endl; 
    std::cout << "=== GAMG config nFinestSweeps: " << nFinestSweeps << std::endl; 

    std::cout << "=== GAMG config interpolateCorrection: " << interpolateCorrection << std::endl;  
    std::cout << "=== GAMG config scaleCorrection: " << scaleCorrection << std::endl; 
    std::cout << "=== GAMG config solveCoarsest: " << solveCoarsest << std::endl; 
    
#ifdef iscsr
    #include "CSRGAMGAgglomeration.H"
    const CSRGAMGAgglomeration& agglomeration(CSRGAMGAgglomeration::New(mesh, solverControls));
#else
    #include "ELLGAMGAgglomeration.H"
    const ELLGAMGAgglomeration& agglomeration(ELLGAMGAgglomeration::New(mesh, solverControls));
#endif

    int agglomeration_level = agglomeration.size();
    std::cout << "=== GAMG agglomeration level: " << agglomeration_level << std::endl;

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

        GAMG_maps[leveli].upperAddr = upperAddrVector.data();
        GAMG_maps[leveli].lowerAddr = lowerAddrVector.data();

        GAMG_maps[leveli].restrictMap = restrictMapVector.data();
        GAMG_maps[leveli].faceRestrictMap = faceRestrictMapVector.data();
        GAMG_maps[leveli].faceFlipMap = faceFlipMapVector_int.data();

        std::cout << "  == level: " << leveli << ", cell: " << GAMG_maps[leveli].nCell
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

}

void createGPUSolverInput(mesh_info_para &mesh_paras, const lduMesh& mesh, const dictionary solverControls, std::vector<int> patch_type, MATRIX_EQUATION EqnID)
{
    std::cout << "==========================================================" << std::endl;
    std::cout << "   Set linear solver variables for DeepFlame_Academic: " << EqnID << std::endl;    
    
    matrix_solver_para solver_configs;

    int minIter=-1, maxIter=-1; // set flag
    double tolerance=-1.0, relTol=-1.0;

    solverControls.readIfPresent("minIter",   minIter);
    solverControls.readIfPresent("maxIter",   maxIter);
    solverControls.readIfPresent("tolerance", tolerance);
    solverControls.readIfPresent("relTol",    relTol);

    std::cout << "=== Solver config minIter: "   << minIter   << std::endl;
    std::cout << "=== Solver config maxIter: "   << maxIter   << std::endl;
    std::cout << "=== Solver config tolerance: " << tolerance << std::endl;
    std::cout << "=== Solver config relTol: "    << relTol    << std::endl;

    if (minIter   > 0) solver_configs.minIter   = minIter;
    if (maxIter   > 0) solver_configs.maxIter   = maxIter;
    if (tolerance > 0) solver_configs.tolerance = tolerance;
    if (relTol    > 0) solver_configs.reltol    = relTol;

    solver_configs.patch_type = (int*)malloc(sizeof(int)*mesh_paras.num_patches);
    std::copy( patch_type.begin(), patch_type.end(), solver_configs.patch_type );   

    // Set solver
    const word solver(solverControls.lookup("solver"));
    std::cout << "=== Solver config solver: " << solver << std::endl;

    // read control files
    word preconditioner, smoother;
    if (solver != "diagonal")
    {
        const entry& eP = solverControls.lookupEntry("preconditioner", false, false);
        if (eP.isDict())
        {
            eP.dict().lookup("preconditioner") >> preconditioner;
            eP.dict().lookup("smoother") >> smoother;
        }
        else
        {
            eP.stream() >> preconditioner;
        }
        std::cout << "=== Solver config preconditioner: " << preconditioner << std::endl;
        std::cout << "=== Solver config smoother: " << smoother << std::endl;
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

    // SET MATRIX_PRECONDITIONER
    int matrix_preconditioner = -1;
    if (preconditioner == "GAMG") 
    {
        matrix_preconditioner = 0;
        createGPUGAMGInput(mesh, solverControls);
    }

    // SET MATRIX_SMOOTHER
    int matrix_smoother = -1;
    if (smoother == "Jacobi") matrix_smoother = 0;
    if (smoother == "SRJ")    matrix_smoother = 1;

    // SET MATRIX_SPARSE_FORMAT
    int sparse_format = 0; // CSR

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
    std::cout << "==========================================================" << std::endl;
    std::cout << "   Set mesh info variables for DeepFlame_Academic   " << std::endl;   

    // prepare constant values: num_cells, num_surfaces, num_boundary_surfaces,
    // num_patches, patch_size, num_species, rdelta_t
    const labelUList& owner = mesh.owner();
    const labelUList& neighbour = mesh.neighbour();
    int num_cells = mesh.nCells();
    int num_total_cells = Foam::returnReduce(num_cells, sumOp<label>());
    int num_surfaces = neighbour.size();
    int num_boundary_surfaces = 0;
    int num_patches = 0;
    std::vector<int> patch_size, patch_offset;
    forAll(mesh.boundary(), patchi) 
    {
        patch_offset.push_back(num_boundary_surfaces);

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
    }
    // prepare interface info
    const globalIndex globalNumbering(num_cells);
    const lduInterfacePtrsList interfaces(mesh.interfaces());
    const lduAddressing& lduAddr = mesh.lduAddr();
    labelList globalCells(num_cells);
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
    // TODO: get deltaT fomr time API
    double rDeltaT = 1 / 1e-6;

    // wyr: now there is no other place to prepare nccl info, thus mpi must be initialized at beginning.
    std::vector<int> GPUNeighbProcNo(num_patches, -1);
    std::vector<int> patch_type(num_patches, -1), interfaceFlag(num_patches, -1);
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

    /* For DeepFlame_Academic */
    createMatrixNoDiag();

    // prepare internal and boundary of sf, mag_sf, weights, delta_coeffs, volume
    double *boundary_sf = new double[3 * num_boundary_surfaces];
    double *boundary_mag_sf = new double[num_boundary_surfaces];
    double *boundary_delta_coeffs = new double[num_boundary_surfaces];
    double *boundary_weights = new double[num_boundary_surfaces];
    int *boundary_face_cell = new int[num_boundary_surfaces];
    std::vector<int> patch_type_calculated(num_patches, 5); // default patch type is calculated
    std::vector<int> patch_type_extropolated(num_patches, 8); // default patch type is extrapolated

    int offset = 0;
    forAll(mesh.boundary(), patchi) {
        const fvPatchScalarField& patchY = Y[0].boundaryField()[patchi];
        const vectorField& pSf = mesh.Sf().boundaryField()[patchi];
        const scalarField& pMagSf = mesh.magSf().boundaryField()[patchi];
        const scalarField& pDeltaCoeffs = mesh.deltaCoeffs().boundaryField()[patchi];
        const scalarField& pWeights = mesh.surfaceInterpolation::weights().boundaryField()[patchi];
        const labelUList& pFaceCells = mesh.boundary()[patchi].faceCells();

        int patchsize = pMagSf.size();

        if (patchY.type() == "processor") {
            patch_type[patchi] = 7;
            interfaceFlag[patchi] = 1;
            patch_type_calculated[patchi] = 7; // patchi type is processor
            patch_type_extropolated[patchi] = 7; // patchi type is processor

            memcpy(boundary_sf + 3*offset, &pSf[0][0], 3*patchsize*sizeof(double));
            memcpy(boundary_sf + 3*offset + 3*patchsize, &pSf[0][0], 3*patchsize*sizeof(double));

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
            memcpy(boundary_mag_sf + offset, &pMagSf[0], patchsize*sizeof(double));
            memcpy(boundary_delta_coeffs + offset, &pDeltaCoeffs[0], patchsize*sizeof(double));
            memcpy(boundary_weights + offset, &pWeights[0], patchsize*sizeof(double));
            memcpy(boundary_face_cell + offset, &pFaceCells[0], patchsize * sizeof(int));

            offset += patchsize;
        } else {
            memcpy(boundary_sf + 3*offset, &pSf[0][0], 3*patchsize*sizeof(double));
            memcpy(boundary_mag_sf + offset, &pMagSf[0], patchsize*sizeof(double));
            memcpy(boundary_delta_coeffs + offset, &pDeltaCoeffs[0], patchsize*sizeof(double));
            memcpy(boundary_weights + offset, &pWeights[0], patchsize*sizeof(double));
            memcpy(boundary_face_cell + offset, &pFaceCells[0], patchsize * sizeof(int));

            offset += patchsize;
        }
    }

    // construct mesh distance for limitedLinear scheme
    vectorField meshDistance = mesh.Sf();
    forAll(meshDistance, facei) {
        label own = owner[facei];
        label nei = neighbour[facei];
        meshDistance[facei] = mesh.C()[nei] - mesh.C()[own];
    }

    double meshDistance_newOrder[num_surfaces*3];
    double sf_newOrder[num_surfaces*3];
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
    
    mesh_paras.patch_size     = new int[num_patches]();
    mesh_paras.patch_offset   = new int[num_patches]();
    mesh_paras.neighbProcNo   = new int[num_patches]();
    mesh_paras.cyclicNeighbor = new int[num_patches]();
    mesh_paras.interfaceFlag  = new int[num_patches]();

    std::copy( patch_size.begin(),      patch_size.end(),      mesh_paras.patch_size );
    std::copy( patch_offset.begin(),    patch_offset.end(),    mesh_paras.patch_offset );
    std::copy( GPUNeighbProcNo.begin(), GPUNeighbProcNo.end(), mesh_paras.neighbProcNo );
    std::copy( cyclicNeighbor.begin(),  cyclicNeighbor.end(),  mesh_paras.cyclicNeighbor );   
    std::copy( interfaceFlag.begin(),   interfaceFlag.end(),   mesh_paras.interfaceFlag ); 

    mesh_paras.lowerAddr             = &owner[0];
    mesh_paras.upperAddr             = &neighbour[0];   

    mesh_paras.mesh_dis              = &meshDistance_newOrder[0];
    mesh_paras.sf                    = &sf_newOrder[0];
    mesh_paras.mag_sf                = &mesh.magSf()[0];
    mesh_paras.delta_coeffs          = &mesh.deltaCoeffs()[0];
    mesh_paras.weight                = &mesh.surfaceInterpolation::weights()[0];
    mesh_paras.volume                = &mesh.V()[0];
    
    mesh_paras.boundary_face_cell    = &boundary_face_cell[0];
    mesh_paras.boundary_sf           = &boundary_sf[0];
    mesh_paras.boundary_mag_sf       = &boundary_mag_sf[0];
    mesh_paras.boundary_delta_coeffs = &boundary_delta_coeffs[0];
    mesh_paras.boundary_weight       = &boundary_weights[0];

    delete boundary_sf;
    delete boundary_mag_sf;
    delete boundary_delta_coeffs;
    delete boundary_weights;
    delete boundary_face_cell;
}

void createMatrixNoDiag(){
    std::cout << "==========================================================" << std::endl;
    std::cout << "   Set matrix info variables for DeepFlame_Academic   " << std::endl;

    // TODO: if CSR;
#ifdef iscsr
    set_matrix_format_csr(dfDataBase.d_ldu_to_csr_no_diag, dfDataBase.d_csr_row_index_no_diag, dfDataBase.d_csr_col_index_no_diag);
#else
    set_matrix_format_ell(dfDataBase.ell_row_maxcount, dfDataBase.d_ellCols, dfDataBase.d_ldu2ellIndex);
#endif
}

void createGAMGMatrixNoDiag(){
    std::cout << "========================================================" << std::endl;
    std::cout << "   Set GAMG matrix info variables for DeepFlame_Academic   " << std::endl;

    // TODO: if CSR;
#ifdef iscsr
    set_GAMG_matrix_format_csr(dfDataBase.h_ldu_to_csr_no_diag, dfDataBase.h_csr_row_index_no_diag, dfDataBase.h_csr_col_index_no_diag);
#else
    set_GAMG_matrix_format_ell(dfDataBase.h_ell_row_maxcount, dfDataBase.h_ellCols, dfDataBase.h_ldu2ellIndex);
#endif
}