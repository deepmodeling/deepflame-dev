#include <queue>

ncclUniqueId nccl_id;
ncclComm_t nccl_comm;
int nRanks, myRank, localRank, mpi_init_flag = 0;

dfMatrixDataBase dfDataBase;
dfThermo thermo_GPU(dfDataBase);
dfChemistrySolver chemistrySolver_GPU(dfDataBase);
dfRhoEqn rhoEqn_GPU(dfDataBase);
dfUEqn UEqn_GPU(dfDataBase);
dfYEqn YEqn_GPU(dfDataBase, chemistrySolver_GPU);
dfEEqn EEqn_GPU(dfDataBase, thermo_GPU);
dfpEqn pEqn_GPU(dfDataBase);

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
    int flag_mpi_init;
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

void createGPURhoEqn(const volScalarField& rho, const surfaceScalarField& phi) {
    DEBUG_TRACE;
    std::vector<int> patch_type;
    patch_type.resize(dfDataBase.num_patches);

    double *h_rho = dfDataBase.getFieldPointer("rho", location::cpu, position::internal);
    double *h_phi = dfDataBase.getFieldPointer("phi", location::cpu, position::internal);
    double *h_boundary_phi = dfDataBase.getFieldPointer("phi", location::cpu, position::boundary);
    double *h_boundary_rho = dfDataBase.getFieldPointer("rho", location::cpu, position::boundary);
    memcpy(h_phi, &phi[0], dfDataBase.surface_value_bytes);
    memcpy(h_rho, &rho[0], dfDataBase.cell_value_bytes);

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
    rhoEqn_GPU.setConstantValues();
    rhoEqn_GPU.setConstantFields(patch_type);
    rhoEqn_GPU.initNonConstantFields(h_rho, h_phi, h_boundary_rho, h_boundary_phi);
    rhoEqn_GPU.createNonConstantLduAndCsrFields();
}

void createGPUUEqn(const IOdictionary& CanteraTorchProperties, const volVectorField& U) {
    // TODO need remove amgx solver setting
    // prepare mode_string and setting_path
    string mode_string = "dDDI";
    string settingPath;
    settingPath = CanteraTorchProperties.subDict("AmgxSettings").lookupOrDefault("UEqnSettingPath", string(""));
    UEqn_GPU.setConstantValues(mode_string, settingPath);

    // prepare patch_type
    std::vector<int> patch_type;
    patch_type.resize(dfDataBase.num_patches);

    double *h_u = dfDataBase.getFieldPointer("u", location::cpu, position::internal);
    double *h_boundary_u = dfDataBase.getFieldPointer("u", location::cpu, position::boundary);
    memcpy(h_u, &U[0][0], dfDataBase.cell_value_vec_bytes);
    
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
    UEqn_GPU.setConstantFields(patch_type);

    // prepare internal and boundary of xxx
    UEqn_GPU.createNonConstantFieldsInternal();
    UEqn_GPU.createNonConstantFieldsBoundary();
    UEqn_GPU.createNonConstantLduAndCsrFields();
    // UEqn_GPU has no internal non-constant fields to be init
    UEqn_GPU.initNonConstantFieldsInternal(h_u, h_boundary_u);
    UEqn_GPU.initNonConstantFieldsBoundary();
}

void createGPUYEqn(const IOdictionary& CanteraTorchProperties, PtrList<volScalarField>& Y, const int inertIndex) {
    DEBUG_TRACE;
    // TODO need remove amgx solver setting
    // prepare mode_string and setting_path
    string mode_string = "dDDI";
    string settingPath;
    settingPath = CanteraTorchProperties.subDict("AmgxSettings").lookupOrDefault("UEqnSettingPath", string(""));
    YEqn_GPU.setConstantValues(mode_string, settingPath, inertIndex);

    // prepare patch_type
    std::vector<int> patch_type;
    patch_type.resize(dfDataBase.num_patches);
    fprintf(stderr, "num_species: %d\n", dfDataBase.num_species);
    forAll(Y[0].boundaryField(), patchi)
    {
        constructBoundarySelectorPerPatch(&(patch_type[patchi]), Y[0].boundaryField()[patchi].type());
    }
    // set lewis number
    std::vector<double> lewis_number(dfDataBase.num_species, 1.); // unity lewis
    YEqn_GPU.setConstantFields(patch_type, lewis_number);

    // prepare internal and boundary of xxx
    YEqn_GPU.createNonConstantFieldsInternal();
    YEqn_GPU.createNonConstantFieldsBoundary();
    YEqn_GPU.createNonConstantLduAndCsrFields();

    // prepare internal and boundary of Y
    int offset = 0;
    forAll(Y, speciesI) {
        volScalarField& Yi = Y[speciesI];
        memcpy(dfDataBase.h_y + speciesI * dfDataBase.num_cells, &Yi[0], dfDataBase.num_cells * sizeof(double));
        forAll(Yi.boundaryField(), patchi) {
            const fvPatchScalarField& patchYi = Yi.boundaryField()[patchi];
            int patchsize = patchYi.size();
            if (patchYi.type() == "processor"
                || patchYi.type() == "processorCyclic") {
                scalarField patchYiInternal =
                    dynamic_cast<const processorFvPatchField<scalar>&>(patchYi).patchInternalField()();
                memcpy(dfDataBase.h_boundary_y + offset, &patchYi[0], patchsize * sizeof(double));
                memcpy(dfDataBase.h_boundary_y + offset + patchsize, &patchYiInternal[0], patchsize * sizeof(double));
                offset += patchsize * 2;
            } else {
                memcpy(dfDataBase.h_boundary_y + offset, &patchYi[0], patchsize*sizeof(double));
                offset += patchsize;
            }
        }
    }
    YEqn_GPU.initNonConstantFieldsInternal(dfDataBase.h_y);
    YEqn_GPU.initNonConstantFieldsBoundary(dfDataBase.h_boundary_y);
}

void createGPUEEqn(const IOdictionary& CanteraTorchProperties, volScalarField& he, volScalarField& K) {
    DEBUG_TRACE;
    // TODO need remove amgx solver setting
    // prepare mode_string and setting_path
    string mode_string = "dDDI";
    string settingPath;
    settingPath = CanteraTorchProperties.subDict("AmgxSettings").lookupOrDefault("UEqnSettingPath", string(""));
    EEqn_GPU.setConstantValues(mode_string, settingPath);

    // prepare patch_type
    std::vector<int> patch_type_he(dfDataBase.num_patches), patch_type_k(dfDataBase.num_patches);
    forAll(he.boundaryField(), patchi)
    {
        constructBoundarySelectorPerPatch(&(patch_type_he[patchi]), he.boundaryField()[patchi].type());
        constructBoundarySelectorPerPatch(&(patch_type_k[patchi]), K.boundaryField()[patchi].type());
    }
    EEqn_GPU.setConstantFields(patch_type_he, patch_type_k);

    // prepare internal and boundary of xxx
    EEqn_GPU.createNonConstantFieldsInternal();
    EEqn_GPU.createNonConstantFieldsBoundary();
    EEqn_GPU.createNonConstantLduAndCsrFields();

    double *h_he = dfDataBase.getFieldPointer("he", location::cpu, position::internal);
    double *h_boundary_he = dfDataBase.getFieldPointer("he", location::cpu, position::boundary);
    memcpy(h_he, &he[0], dfDataBase.cell_value_bytes);
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

    EEqn_GPU.initNonConstantFields(h_he, h_boundary_he);
}

void createGPUpEqn(const IOdictionary& CanteraTorchProperties, volScalarField& p, const volVectorField& U) {
    DEBUG_TRACE;
    // TODO need remove amgx solver setting
    // prepare mode_string and setting_path
    string mode_string = "dDDI";
    string settingPath;
    settingPath = CanteraTorchProperties.subDict("AmgxSettings").lookupOrDefault("pEqnSettingPath", string(""));
    pEqn_GPU.setConstantValues(mode_string, settingPath);
    
    // prepare patch_type
    std::vector<int> patch_type_p(dfDataBase.num_patches);
    std::vector<int> patch_type_U(dfDataBase.num_patches);

    double *h_p = dfDataBase.getFieldPointer("p", location::cpu, position::internal);
    double *h_boundary_p = dfDataBase.getFieldPointer("p", location::cpu, position::boundary);
    memcpy(h_p, &p[0], dfDataBase.cell_value_bytes);

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
    pEqn_GPU.setConstantFields(patch_type_U, patch_type_p);
    pEqn_GPU.initNonConstantFields(h_p, h_boundary_p);

    // prepare internal and boundary of xxx
    pEqn_GPU.createNonConstantFieldsInternal();
    pEqn_GPU.createNonConstantFieldsBoundary();
    pEqn_GPU.createNonConstantLduAndCsrFields();
}

void createGPUThermo(const IOdictionary& CanteraTorchProperties, volScalarField& T, volScalarField& he, 
        const volScalarField& psi, const volScalarField& alpha, const volScalarField& mu,
        const volScalarField& K, const volScalarField& dpdt, dfChemistryModel<basicThermo>* chemistry) {
    DEBUG_TRACE;
    // initialize dfThermo
    string mechanismFile;
    mechanismFile = CanteraTorchProperties.lookupOrDefault("CanteraMechanismFile", string(""));

    thermo_GPU.setConstantValue(mechanismFile, dfDataBase.num_cells, dfDataBase.num_species);
    init_const_coeff_ptr(thermo_GPU.nasa_coeffs, thermo_GPU.viscosity_coeffs, thermo_GPU.thermal_conductivity_coeffs, 
            thermo_GPU.binary_diffusion_coeffs, thermo_GPU.molecular_weights);

    // thermal variables in dataBase
    // TODO: note that h_he & h_boundary_he are transfered to GPU in EEqn_GPU, too. We should delete one of them.
    double *h_boundary_T = new double[dfDataBase.num_boundary_surfaces];
    double *h_boundary_he = new double[dfDataBase.num_boundary_surfaces];
    double *h_boundary_thermo_psi = new double[dfDataBase.num_boundary_surfaces];
    double *h_boundary_thermo_alpha = new double[dfDataBase.num_boundary_surfaces];
    double *h_boundary_mu = new double[dfDataBase.num_boundary_surfaces];
    double *h_boundary_k = new double[dfDataBase.num_boundary_surfaces];
    double *h_boundary_thermo_rhoD = new double[dfDataBase.num_boundary_surfaces * dfDataBase.num_species];
    double *h_thermo_rhoD = new double[dfDataBase.num_cells * dfDataBase.num_species];

    // initialize thermo boundary
    std::vector<int> patch_type_T(dfDataBase.num_patches);
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

            for (int i = 0; i < dfDataBase.num_species; i++) {
                const fvPatchScalarField& patchRhoD = chemistry->rhoD(i).boundaryField()[patchi];
                memcpy(h_boundary_thermo_rhoD + i * dfDataBase.num_boundary_surfaces + offset, &patchRhoD[0], patchsize * sizeof(double));
                scalarField patchRhoDInternal = 
                        dynamic_cast<const processorFvPatchField<scalar>&>(patchRhoD).patchInternalField()();
                memcpy(h_boundary_thermo_rhoD + i * dfDataBase.num_boundary_surfaces + offset + patchsize, &patchRhoDInternal[0], patchsize * sizeof(double));
            }

            offset += patchsize * 2;
        } else {
            memcpy(h_boundary_T + offset, &patchT[0], patchsize * sizeof(double));
            memcpy(h_boundary_he + offset, &patchHe[0], patchsize * sizeof(double));
            memcpy(h_boundary_thermo_psi + offset, &patchPsi[0], patchsize * sizeof(double));
            memcpy(h_boundary_thermo_alpha + offset, &patchAlpha[0], patchsize * sizeof(double));
            memcpy(h_boundary_mu + offset, &patchMu[0], patchsize * sizeof(double));
            memcpy(h_boundary_k + offset, &patchK[0], patchsize * sizeof(double));

            for (int i = 0; i < dfDataBase.num_species; i++) {
                const fvPatchScalarField& patchRhoD = chemistry->rhoD(i).boundaryField()[patchi];
                memcpy(h_boundary_thermo_rhoD + i * dfDataBase.num_boundary_surfaces + offset, &patchRhoD[0], patchsize * sizeof(double));
            }
            offset += patchsize;
        }
    }
    for (int i = 0; i < dfDataBase.num_species; i++) {
        memcpy(h_thermo_rhoD + i * dfDataBase.num_cells, &chemistry->rhoD(i)[0], dfDataBase.num_cells * sizeof(double));
    }
    double *h_T = dfDataBase.getFieldPointer("T", location::cpu, position::internal);
    memcpy(h_T, &T[0], dfDataBase.cell_value_bytes);

    thermo_GPU.setConstantFields(patch_type_T);
    thermo_GPU.initNonConstantFields(h_T, &he[0], &psi[0], &alpha[0], &mu[0], &K[0], &dpdt[0], h_thermo_rhoD,
            h_boundary_T, h_boundary_he, h_boundary_thermo_psi, h_boundary_thermo_alpha, h_boundary_mu, h_boundary_k, h_boundary_thermo_rhoD);

    delete h_boundary_T;
    delete h_boundary_he;
    delete h_boundary_thermo_psi;
    delete h_boundary_thermo_alpha;
    delete h_boundary_mu;
    delete h_boundary_k;
    delete h_boundary_thermo_rhoD;
    delete h_thermo_rhoD;
}
