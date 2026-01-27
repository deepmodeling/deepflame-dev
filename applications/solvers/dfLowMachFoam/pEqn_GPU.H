thermo_GPU.updateRho();

// Thermodynamic density needs to be updated by psi*d(p) after the
// pressure solution
thermo_GPU.psip0();

UEqn_GPU.getHbyA();
pEqn_GPU.process();
UEqn_GPU.sync();

#if defined DEBUG_
    if (!pimple.simpleRho())
    {
        rho = thermo.rho();
    }

    const volScalarField psip0(psi*p);

    volScalarField rAU(1/UEqn.A());
    surfaceScalarField rhorAUf("rhorAUf", fvc::interpolate(rho*rAU));
    volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U, p));

    double *h_boundary_rAU = new double[dfDataBase.num_boundary_surfaces];
    double *h_boundary_rhorAUf = (double*)calloc(dfDataBase.num_boundary_surfaces, sizeof(double));
    double *h_boundary_HbyA = new double[3 * dfDataBase.num_boundary_surfaces];
    offset = 0;
    forAll(rAU.boundaryField(), patchi)
    {
        const fvPatchScalarField& patchrAU = rAU.boundaryField()[patchi];
        const fvPatchVectorField& patchHbyA = HbyA.boundaryField()[patchi];
        const fvsPatchScalarField& patchrhorAUf = rhorAUf.boundaryField()[patchi];
        int patchSize = patchrAU.size();

        if (patchrAU.type() == "processor"
            || patchrAU.type() == "processorCyclic") {
            memcpy(h_boundary_rAU + offset, &patchrAU[0], patchSize*sizeof(double));
            scalarField patchrAUInternal = 
                    dynamic_cast<const processorFvPatchField<scalar>&>(patchrAU).patchInternalField()();
            memcpy(h_boundary_rAU + offset + patchSize, &patchrAUInternal[0], patchSize*sizeof(double));

            memcpy(h_boundary_rhorAUf + offset, &patchrhorAUf[0], patchSize*sizeof(double));

            memcpy(h_boundary_HbyA + offset * 3, &patchHbyA[0][0], patchSize*3*sizeof(double));
            vectorField patchHbyAInternal = 
                    dynamic_cast<const processorFvPatchField<vector>&>(patchHbyA).patchInternalField()();
            memcpy(h_boundary_HbyA + offset * 3 + patchSize * 3, &patchHbyAInternal[0][0], patchSize*3*sizeof(double));

            offset += patchSize * 2;
        } else {
            memcpy(h_boundary_rAU + offset, &patchrAU[0], patchSize*sizeof(double));
            memcpy(h_boundary_rhorAUf + offset, &patchrhorAUf[0], patchSize*sizeof(double));
            memcpy(h_boundary_HbyA + offset * 3, &patchHbyA[0][0], patchSize*3*sizeof(double));
            offset += patchSize;
        }
    }
    if (!mpi_init_flag || rank == 0) {
        // UEqn_GPU.compareHbyA(&HbyA[0][0], h_boundary_HbyA, false);
        // UEqn_GPU.comparerAU(&rAU[0], h_boundary_rAU, false);
        // pEqn_GPU.comparerhorAUf(&rhorAUf[0], h_boundary_rhorAUf, false);
    }

    delete h_boundary_rAU;
    delete h_boundary_rhorAUf;
    delete h_boundary_HbyA;
#endif


#if defined DEBUG_
    surfaceScalarField phiHbyA
    (
        "phiHbyA",
        fvc::interpolate(rho)*fvc::flux(HbyA)
    + rhorAUf*fvc::ddtCorr(rho, U, phi, rhoUf)
    );

    double *h_boundary_phiHbyA = (double*)calloc(dfDataBase.num_boundary_surfaces, sizeof(double));
    offset = 0;
    forAll(phiHbyA.boundaryField(), patchi)
    {
        const fvsPatchScalarField& patchphiHbyA = phiHbyA.boundaryField()[patchi];
        int patchSize = patchphiHbyA.size();
        if (patchphiHbyA.type() == "processor"
            || patchphiHbyA.type() == "processorCyclic") {
            memcpy(h_boundary_phiHbyA + offset, &patchphiHbyA[0], patchSize*sizeof(double));
            offset += 2 * patchSize;
        } else {
            memcpy(h_boundary_phiHbyA + offset, &patchphiHbyA[0], patchSize*sizeof(double));
            offset += patchSize;
        }
    }
    if (!mpi_init_flag || rank == 0) {
        // pEqn_GPU.comparephiHbyA(&phiHbyA[0], h_boundary_phiHbyA, false);
    }
    delete h_boundary_phiHbyA;
#endif

#if defined DEBUG_
    fvScalarMatrix pDDtEqn
    (
        fvc::ddt(rho) + 
        psi*correction(fvm::ddt(p))
        + fvc::div(phiHbyA)
    );
    while (pimple.correctNonOrthogonal())
    {
        fvScalarMatrix pEqn(pDDtEqn - fvm::laplacian(rhorAUf, p));
        pEqn.solve();
        if (pimple.finalNonOrthogonalIter())
        {
            phi = phiHbyA + pEqn.flux();
        }
        thermo.correctRho(psi*p - psip0);
        // compare pEqn
        std::vector<double> h_internal_coeffs(dfDataBase.num_boundary_surfaces);
        std::vector<double> h_boundary_coeffs(dfDataBase.num_boundary_surfaces);

        offset = 0;
        for (int patchi = 0; patchi < dfDataBase.num_patches; patchi++)
        {
            const fvPatchScalarField& patchP = p.boundaryField()[patchi];
            int patchsize = dfDataBase.patch_size[patchi];
            const double* internal_coeff_ptr = &pEqn.internalCoeffs()[patchi][0];
            const double* boundary_coeff_ptr = &pEqn.boundaryCoeffs()[patchi][0];
            memcpy(h_internal_coeffs.data() + offset, internal_coeff_ptr, patchsize * sizeof(double));
            memcpy(h_boundary_coeffs.data() + offset, boundary_coeff_ptr, patchsize * sizeof(double));
            if (patchP.type() == "processor" || patchP.type() == "processorCyclic") offset += 2 * patchsize;
            else offset += patchsize;
        }
        if (!mpi_init_flag || rank == 0) {
            // pEqn_GPU.compareResult(&pEqn.lower()[0], &pEqn.upper()[0], &pEqn.diag()[0], &pEqn.source()[0], 
            //         h_internal_coeffs.data(), h_boundary_coeffs.data(), false);
        }
    }

    // compare p
    double *h_boundary_p = new double[dfDataBase.num_boundary_surfaces];
    offset = 0;
    forAll(p.boundaryField(), patchi)
    {
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
    // pEqn_GPU.correctP(&p[0], h_boundary_p);
    if (!mpi_init_flag || rank == 0) {
        //pEqn_GPU.comparep(&p[0], h_boundary_p, false);
    }
    delete h_boundary_p;

    // compare phi
    double *h_boundary_phi = new double[dfDataBase.num_boundary_surfaces];
    offset = 0;
    forAll(phi.boundaryField(), patchi)
    {
        const fvsPatchScalarField& patchFlux = phi.boundaryField()[patchi];
        int patchSize = patchFlux.size();
        memcpy(h_boundary_phi + offset, &patchFlux[0], patchSize*sizeof(double));
        if (patchFlux.type() == "processor"
            || patchFlux.type() == "processorCyclic") {
            memset(h_boundary_phi + offset + patchSize, 0, patchSize*sizeof(double));
            offset += 2 * patchSize;
        } else {
            offset += patchSize;
        }
    }
    if (!mpi_init_flag || rank == 0) {
        // pEqn_GPU.comparephi(&phi[0], h_boundary_phi, false);
    }
    delete h_boundary_phi;
#endif

thermo_GPU.correctPsipRho();

#include "rhoEqn.H"
// #include "compressibleContinuityErrs.H" // TODO: implement this func in future

#if defined DEBUG_

    U = HbyA - rAU*fvc::grad(p);
    U.correctBoundaryConditions();
    K = 0.5*magSqr(U);

    // check U
    double *h_boundary_u_tmp_inp = new double[dfDataBase.num_boundary_surfaces * 3];
    offset = 0;
    forAll(U.boundaryField(), patchi)
    {
        const fvPatchVectorField& patchU = U.boundaryField()[patchi];
        int patchSize = patchU.size();

        if (patchU.type() == "processor"
            || patchU.type() == "processorCyclic") {
            memcpy(h_boundary_u_tmp_inp + 3*offset, &patchU[0][0], patchSize*sizeof(double)*3);
            vectorField patchUInternal = 
                    dynamic_cast<const processorFvPatchField<vector>&>(patchU).patchInternalField()();
            memcpy(h_boundary_u_tmp_inp + 3*offset + 3*patchSize, &patchUInternal[0][0], patchSize*sizeof(double)*3);
            offset += patchSize * 2;
        } else {
            memcpy(h_boundary_u_tmp_inp + 3*offset, &patchU[0][0], patchSize*sizeof(double)*3);
            offset += patchSize;
        }
    }
    if (!mpi_init_flag || rank == 0) {
        // pEqn_GPU.compareU(&U[0][0], h_boundary_u_tmp_inp, false);
    }
    delete h_boundary_u_tmp_inp;
#endif

#if defined DEBUG_ 
    dpdt = fvc::ddt(p);
    // pEqn_GPU.comparedpdt(&dpdt[0], false);
#endif

// #undef CPUSolver_
// #define GPUSolverNew_
