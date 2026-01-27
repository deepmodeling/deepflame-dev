// #TODO Y_NUM
#define Y_NUM 17

template <class ThermoType>
// void Foam::dfChemistryModel<ThermoType>::thermoDNN_blas(volScalarField& he, const volScalarField& p, volScalarField& mixfrac,
//                 volScalarField& rho, volScalarField& T, volScalarField& psi, volScalarField& mu,
//                 volScalarField& alpha, PtrList<volScalarField>& rhoD)
// {
void Foam::dfChemistryModel<ThermoType>::thermoDNN_blas(volScalarField& he, const volScalarField& p, PtrList<volScalarField>& Y,
                volScalarField& rho, volScalarField& T, volScalarField& psi, volScalarField& mu,
                volScalarField& alpha, PtrList<volScalarField>& rhoD)
{
    double init_time = 0.;
    double prepare_time = 0.;
    double inference_time = 0.;
    double update_time = 0.;
    double destory_time= 0.;

    clockTime clock;

    label nCells = he.size();
    label input_dim = 19;
    label offset = 0;
    label output_dim_0 = 5;
    label output_dim_1 = 0;


    Info<<"=== begin solve_thermo with BLASDNN === "<<endl;


#if defined(USE_BLASDNN_HALF)
    __fp16* NNInputs = (__fp16*)aligned_alloc(64, (nCells * input_dim) * sizeof(__fp16));
    memset(NNInputs, '\0', (nCells * input_dim) * sizeof(__fp16));
    __fp16* results0 = (__fp16*)aligned_alloc(64, nCells * output_dim_0 * sizeof(__fp16));
    __fp16* results1 = (__fp16*)aligned_alloc(64, (nCells * output_dim_1) * sizeof(__fp16));
#elif defined(USE_BLASDNN_FLOAT)
    float* NNInputs = (float*)aligned_alloc(64, (nCells * input_dim) * sizeof(float));
    memset(NNInputs, '\0', (nCells * input_dim) * sizeof(float));
    float* results0 = (float*)aligned_alloc(64, nCells * output_dim_0 * sizeof(float));
    float* results1 = (float*)aligned_alloc(64, (nCells * output_dim_1) * sizeof(float));
#elif defined(USE_BLASDNN_DOUBLE)
    double* NNInputs = (double*)aligned_alloc(64, (nCells * input_dim) * sizeof(double));
    memset(NNInputs, '\0', (nCells * input_dim) * sizeof(double));
    double* results0 = (double*)aligned_alloc(64, nCells * output_dim_0 * sizeof(double));
    double* results1 = (double*)aligned_alloc(64, nCells * output_dim_1 * sizeof(double));
#endif

    init_time += clock.timeIncrement();

    const volScalarField::Boundary& pBf = p.boundaryField();

    volScalarField::Boundary& rhoBf = rho.boundaryFieldRef();

    volScalarField::Boundary& TBf = T.boundaryFieldRef();

    volScalarField::Boundary& psiBf = psi.boundaryFieldRef();

    volScalarField::Boundary& muBf = mu.boundaryFieldRef();

    volScalarField::Boundary& alphaBf = alpha.boundaryFieldRef();

    // volScalarField::Boundary mixfracBf = mixfrac.boundaryField();  

    Info<<"=== begin solve_thermo patch with BLASDNN === "<<endl;
    label nFaces = 0;
    volScalarField::Boundary& hBf = he.boundaryFieldRef();
    forAll(T_.boundaryField(), patchi)
    {
        fvPatchScalarField& ph = hBf[patchi];
        nFaces += ph.size();
    }
    Info << "total # of faces is " << nFaces << endl;

    // Prepare input for DNN
#ifdef _OPENMP
#pragma omp parallel for
#endif
    forAll(he, cellI)
    {
        // NNInputs[cellI * input_dim + 0] = ((std::pow(mixfrac[cellI], 0.1) - 1.) * 10. - thermomu_.back()) / thermostd_.back();
        NNInputs[cellI * input_dim + 0] = (he[cellI] - thermomu_[0]) / thermostd_[0];
        NNInputs[cellI * input_dim + 1] = (p[cellI] - thermomu_[1]) / thermostd_[1];
        for(int i = 0; i < Y_NUM; i++){
            NNInputs[cellI * input_dim + 2 + i] = (Y[i][cellI] - thermomu_[2 + i])/thermostd_[2 + i];
        }
    }

    prepare_time += clock.timeIncrement();

    DNNThermo_blas_.Inference(nCells, NNInputs, 
        results0, results1);

    inference_time += clock.timeIncrement();

//     // update thermo properties
// #ifdef _OPENMP
// #pragma omp parallel for
// #endif
//     forAll(he, cellI){
//         size_t offset = cellI * 5;
//         // rho[cellI] = std::pow((results0[cellI] * thermostd_[0] + thermomu_[0]) * 0.1 + 1., 10.) * 1000;
//         rho[cellI] = std::exp(results0[offset] * thermostd_[input_dim] + thermomu_[input_dim]);
//         T[cellI] = results0[offset + 1] * thermostd_[input_dim+1] + thermomu_[input_dim+1];
//         mu[cellI] = results0[offset + 2] * thermostd_[input_dim+2] + thermomu_[input_dim+2];
//         alpha[cellI] = results0[offset + 3] * thermostd_[input_dim+3] + thermomu_[input_dim+3];
//         psi[cellI] = results0[offset + 4] * thermostd_[input_dim+4] + thermomu_[input_dim+4];
//         forAll(rhoD_, i){
//             rhoD_[i][cellI] = alpha[cellI];
//         }
//     }
    update_time += clock.timeIncrement();
    free(NNInputs);
    free(results0);
    free(results1);

    destory_time += clock.timeIncrement();

    /*
    ** boundaryField
    */
// #ifdef _FP16_
#if defined(USE_BLASDNN_HALF)
    __fp16* NNInputs_p = (__fp16*)aligned_alloc(64, (nFaces * input_dim) * sizeof(__fp16));
    memset(NNInputs_p, '\0', (nFaces * input_dim) * sizeof(__fp16));
    __fp16* results0_p = (__fp16*)aligned_alloc(64, nFaces * output_dim_0 * sizeof(__fp16));
    __fp16* results1_p = (__fp16*)aligned_alloc(64, (nFaces * output_dim_1) * sizeof(__fp16));
#elif defined(USE_BLASDNN_FLOAT)
    float* NNInputs_p = (float*)aligned_alloc(64, (nFaces * input_dim) * sizeof(float));
    memset(NNInputs_p, '\0', (nFaces * input_dim) * sizeof(float));
    float* results0_p = (float*)aligned_alloc(64, nFaces * output_dim_0 * sizeof(float));
    float* results1_p = (float*)aligned_alloc(64, (nFaces * output_dim_1) * sizeof(float));
#elif defined(USE_BLASDNN_DOUBLE)
    double* NNInputs_p = (double*)aligned_alloc(64, (nFaces * input_dim) * sizeof(double));
    memset(NNInputs_p, '\0', (nFaces * input_dim) * sizeof(double));
    double* results0_p = (double*)aligned_alloc(64, nFaces * output_dim_0 * sizeof(double));
    double* results1_p = (double*)aligned_alloc(64, (nFaces * output_dim_1) * sizeof(double));
#else
#error "must define one of USE_BLASDNN_HALF, USE_BLASDNN_FLOAT or USE_BLASDNN_DOUBLE"
#endif

    init_time += clock.timeIncrement();

    offset = 0;
    // Prepare input for DNN
    forAll(he.boundaryField(), patchi)
    {
        const fvPatchScalarField& pp = pBf[patchi];
        fvPatchScalarField& ph = hBf[patchi];
        // fvPatchScalarField pmixfrac = mixfracBf[patchi];
#ifdef _OPENMP
#pragma omp parallel for
#endif
        forAll(ph, facei)
        {
//             // NNInputs_p[offset + facei * input_dim + 0] = ((std::pow(pmixfrac[facei], 0.1) - 1.) * 10. - thermomu_.back()) / thermostd_.back();
            NNInputs_p[offset +facei * input_dim + 0] = (ph[facei] - thermomu_[0]) / thermostd_[0];
            NNInputs_p[offset +facei * input_dim + 1] = (pp[facei] - thermomu_[1]) / thermostd_[1];
            for(int i = 0; i < Y_NUM; i++){
                NNInputs_p[offset +facei * input_dim + 2 + i] = (Y[i].boundaryFieldRef()[patchi][facei] - thermomu_[2 + i])/thermostd_[2 + i];
            }
        }
        offset += ph.size()*input_dim;
    }

    prepare_time += clock.timeIncrement();

    // printf("DNNThermo_blas_(boundary) Start\n");
    // TODO: if(pT.fixesValue())
    DNNThermo_blas_.Inference(nFaces, NNInputs_p, 
        results0_p, results1_p);

    inference_time += clock.timeIncrement();

//     offset = 0;
//     // update thermo properties
//     forAll(he.boundaryField(), patchi)
//     {
//         fvPatchScalarField& prho = rhoBf[patchi];
//         fvPatchScalarField& pT = TBf[patchi];
//         fvPatchScalarField& ppsi = psiBf[patchi];
//         fvPatchScalarField& pmu = muBf[patchi];
//         fvPatchScalarField& palpha = alphaBf[patchi];
// #ifdef _OPENMP
// #pragma omp parallel for
// #endif
//         forAll(pT, facei)
//         {
//             // prho[facei] = std::pow((results0_p[(offset + facei) * output_dim_0 + 0] * thermostd_[0] + thermomu_[0]) * 0.1 + 1., 10.) * 1000;
//             prho[facei] = std::exp((results0_p[(offset + facei) * output_dim_0 + 0] * thermostd_[input_dim+0] + thermomu_[input_dim+0]));
//             pT[facei] = results0_p[(offset + facei) * output_dim_0 + 1] * thermostd_[input_dim+1] + thermomu_[input_dim+1];
//             ppsi[facei] = results0_p[(offset + facei) * output_dim_0 + 2] * thermostd_[input_dim+2] + thermomu_[input_dim+2];
//             pmu[facei] = results0_p[(offset + facei) * output_dim_0 + 3] * thermostd_[input_dim+3] + thermomu_[input_dim+3];
//             palpha[facei] = results0_p[(offset + facei) * output_dim_0 + 4] * thermostd_[input_dim+4] + thermomu_[input_dim+4];
//             forAll(rhoD_, i)
//             {
//                 rhoD_[i].boundaryFieldRef()[patchi][facei] = palpha[facei];
//             }
//         }
//         offset += pT.size();
//     }

    update_time += clock.timeIncrement();
    
    free(NNInputs_p);
    free(results0_p);
    free(results1_p);

    destory_time += clock.timeIncrement();

    double total_time = clock.elapsedTime();

    Info << "=== end solve_thermo with BLASDNN === " << endl;

    Info << "solve_DNN_blas_thermo profiling --------------------------------------------------" << endl;
    Info << "solve_DNN_blas_thermo total time: " << total_time << endl;
    Info << "init time: " << init_time << endl;
    Info << "prepare time: " << prepare_time << endl;
    Info << "inference time: " << inference_time << endl;
    Info << "update time: " << update_time << endl;
    Info << "destory time: " << destory_time << endl;
    Info << "other time: " << total_time - init_time - prepare_time - inference_time - update_time - destory_time  << endl;
    Info << "----------------------------------------------------------------------------------" << endl;
}
