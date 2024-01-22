double p_build_opt_start, p_build_opt_end, p_build_opt_time = 0.;
double p_build_org_start, p_build_org_end, p_build_org_time = 0.;

double p_pre_time = 0.;
double p_post_time = 0.;
double p_build_time = 0.;
double p_convert_time = 0.;
double p_solve_time = 0.;
double p_pimple_time = 0.;

clockTime clock;

if (!pimple.simpleRho())
{
    rho = thermo.rho();
}

// Thermodynamic density needs to be updated by psi*d(p) after the
// pressure solution
const volScalarField psip0(psi*p);

#ifdef OPT_GenMatrix_p
    tmp<volScalarField> trAU = rAUConstructor(UEqn);
    volScalarField& rAU = trAU.ref();
#else
    volScalarField rAU(1.0/UEqn.A());
#endif

tmp<volScalarField> tmp_rhoRau = rho*rAU;

// #ifdef OPT_GenMatrix_p // FIXME: write intepolation introduce small err
//     tmp<surfaceScalarField> trhorAUf = rhorAUfConstructor(tmp_rhoRau, linear_weights);
//     surfaceScalarField rhorAUf = trhorAUf();
// #else
surfaceScalarField rhorAUf("rhorAUf", fvc::interpolate(tmp_rhoRau));
// #endif

#ifdef OPT_GenMatrix_p
    tmp<volVectorField> tUEqnH = UEqn_H(UEqn);
#else
    tmp<volVectorField> tUEqnH = UEqn.H();
#endif

volVectorField HbyA(constrainHbyA(rAU*tUEqnH, U, p));

if (pimple.nCorrPiso() <= 1)
{
    tUEqn.clear();
}

#ifdef OPT_GenMatrix_p
    tmp<surfaceScalarField> tddtCorr = EulerDdtSchemeFvcDdtCorr(rho, U, phi, rhoUf);
    surfaceScalarField ddtCorr = tddtCorr();
#else
    tmp<surfaceScalarField> tddtCorr = fvc::ddtCorr(rho, U, phi, rhoUf);
    surfaceScalarField ddtCorr = tddtCorr();
#endif

// check_field_equal(ddtCorrTest, ddtCorrRef);

// #ifdef OPT_GenMatrix_p
//     tmp<surfaceScalarField> tphiHby_test = phiHbyAConstructor(rho, HbyA, rhorAUf, tddtCorr, linear_weights);
//     surfaceScalarField phiHby_test = tphiHby_test.ref();
// #else
surfaceScalarField phiHbyA
(
    "phiHbyA",
    fvc::interpolate(rho)*fvc::flux(HbyA)
    + rhorAUf*tddtCorr
);
// #endif
// construct phiHbyA directly

fvc::makeRelative(phiHbyA, rho, U);

// Update the pressure BCs to ensure flux consistency
constrainPressure(p, rho, U, phiHbyA, rhorAUf);

p_pre_time += clock.timeIncrement();

if (pimple.transonic())
{
    p_pimple_time += clock.timeIncrement();

    surfaceScalarField phid
    (
        "phid",
        (fvc::interpolate(psi)/fvc::interpolate(rho))*phiHbyA
    );

    phiHbyA -= fvc::interpolate(psi*p)*phiHbyA/fvc::interpolate(rho);
    
    fvScalarMatrix pDDtEqn
    (
        fvc::ddt(rho) + psi * correction(fvm::ddt(p))
      + fvc::div(phiHbyA) + fvm::div(phid, p)
    );
    p_build_time += clock.timeIncrement();

    while (pimple.correctNonOrthogonal())
    {
        p_pimple_time += clock.timeIncrement();

        fvScalarMatrix pEqn(pDDtEqn - fvm::laplacian(rhorAUf, p));
        // Relax the pressure equation to ensure diagonal-dominance
        pEqn.relax();

        p_build_time += clock.timeIncrement();

#ifdef _CSR_
        csrMatrix csr(pEqn);
#endif
#ifdef _ELL_
        ellMatrix ell(pEqn);
#endif
#ifdef _DIV_
        // divMatrix div(pEqn);
        div.copy_value_from_fvMatrix(pEqn);
#endif
#ifdef _LDU_
        LDUMatrix LDU(pEqn);
#endif
        p_convert_time += clock.timeIncrement();

#ifdef _CSR_
        auto& psi = const_cast<GeometricField<scalar, fvPatchField, volMesh>&>(pEqn.psi());
        auto& source = pEqn.source();
        auto& internalCoeffs = pEqn.internalCoeffs();
        auto& boundaryCoeffs = pEqn.boundaryCoeffs();
        csr.solve(psi, source, internalCoeffs, boundaryCoeffs);
#endif
#ifdef _ELL_
        auto& psi = const_cast<GeometricField<scalar, fvPatchField, volMesh>&>(pEqn.psi());
        auto& source = pEqn.source();
        auto& internalCoeffs = pEqn.internalCoeffs();
        auto& boundaryCoeffs = pEqn.boundaryCoeffs();
        ell.solve(psi, source, internalCoeffs, boundaryCoeffs);
#endif
#ifdef _DIV_
        auto& psi = const_cast<GeometricField<scalar, fvPatchField, volMesh>&>(pEqn.psi());
        auto& source = pEqn.source();
        auto& internalCoeffs = pEqn.internalCoeffs();
        auto& boundaryCoeffs = pEqn.boundaryCoeffs();
        div.solve(psi, source, internalCoeffs, boundaryCoeffs);
#endif
#ifdef _LDU_
        auto& psi = const_cast<GeometricField<scalar, fvPatchField, volMesh>&>(pEqn.psi());
        auto& source = pEqn.source();
        auto& internalCoeffs = pEqn.internalCoeffs();
        auto& boundaryCoeffs = pEqn.boundaryCoeffs();
        LDU.solve(psi, source, internalCoeffs, boundaryCoeffs);
#endif

#if !defined(_CSR_) && !defined(_ELL_) && !defined(_LDU_) && !defined(_DIV_)
        pEqn.solve();
#endif
        p_solve_time += clock.timeIncrement();

        if (pimple.finalNonOrthogonalIter())
        {
            phi = phiHbyA + pEqn.flux();
        }
        
        p_post_time += clock.timeIncrement();

    }
}
else
{
    p_pimple_time += clock.timeIncrement();

#ifdef OPT_GenMatrix_p
    p_build_opt_start = MPI_Wtime();
    tmp<fvScalarMatrix> tpDDtEqn = GenMatrix_p(rho, p, phiHbyA, rhorAUf, psi);
    fvScalarMatrix pDDtEqn = tpDDtEqn.ref();
    p_build_opt_end = MPI_Wtime();
    p_build_opt_time += p_build_opt_end - p_build_opt_start;
    Info << "p_build_opt_time : " << p_build_opt_time << endl;
#else
    p_build_org_start = MPI_Wtime();
    fvScalarMatrix pDDtEqn
    (
        fvc::ddt(rho) + psi * correction(fvm::ddt(p))
        // fvm::ddt(p)
      + fvc::div(phiHbyA) - fvm::laplacian(rhorAUf, p)
    );
    p_build_org_end = MPI_Wtime();
    p_build_org_time += p_build_org_end - p_build_org_start;
    Info << "p_build_org_time : " << p_build_org_time << endl;
#endif

#ifdef OPT_GenMatrix_p_check
    p_build_org_start = MPI_Wtime();
    fvScalarMatrix pDDtEqn_answer
    (
        fvc::ddt(rho) + psi * correction(fvm::ddt(p))
      + fvc::div(phiHbyA)
      - fvm::laplacian(rhorAUf, p)
    );
    p_build_org_end = MPI_Wtime();
    p_build_org_time += p_build_org_end - p_build_org_start;
    Info << "p_build_org_time : " << p_build_org_time << endl;
    check_fvmatrix_equal(pDDtEqn, pDDtEqn_answer);
#endif

    p_build_time += clock.timeIncrement();

    while (pimple.correctNonOrthogonal())
    {
        p_pimple_time += clock.timeIncrement();

        fvScalarMatrix pEqn(pDDtEqn);

        p_build_time += clock.timeIncrement();

#ifdef _CSR_
        csrMatrix csr(pEqn);
#endif
#ifdef _ELL_
        ellMatrix ell(pEqn);
#endif
#ifdef _DIV_
        // divMatrix div(pEqn);
        div.copy_value_from_fvMatrix(pEqn);
#endif
#ifdef _LDU_
        LDUMatrix LDU(pEqn);
#endif
        p_convert_time += clock.timeIncrement();

#ifdef _CSR_
        auto& psi = const_cast<GeometricField<scalar, fvPatchField, volMesh>&>(pEqn.psi());
        auto& source = pEqn.source();
        auto& internalCoeffs = pEqn.internalCoeffs();
        auto& boundaryCoeffs = pEqn.boundaryCoeffs();
        csr.solve(psi, source, internalCoeffs, boundaryCoeffs);
#endif
#ifdef _ELL_
        auto& psi = const_cast<GeometricField<scalar, fvPatchField, volMesh>&>(pEqn.psi());
        auto& source = pEqn.source();
        auto& internalCoeffs = pEqn.internalCoeffs();
        auto& boundaryCoeffs = pEqn.boundaryCoeffs();
        ell.solve(psi, source, internalCoeffs, boundaryCoeffs);
#endif
#ifdef _DIV_
        auto& psi = const_cast<GeometricField<scalar, fvPatchField, volMesh>&>(pEqn.psi());
        auto& source = pEqn.source();
        auto& internalCoeffs = pEqn.internalCoeffs();
        auto& boundaryCoeffs = pEqn.boundaryCoeffs();
        div.solve(psi, source, internalCoeffs, boundaryCoeffs);
#endif
#ifdef _LDU_
        auto& psi = const_cast<GeometricField<scalar, fvPatchField, volMesh>&>(pEqn.psi());
        auto& source = pEqn.source();
        auto& internalCoeffs = pEqn.internalCoeffs();
        auto& boundaryCoeffs = pEqn.boundaryCoeffs();
        LDU.solve(psi, source, internalCoeffs, boundaryCoeffs);
#endif

#if !defined(_CSR_) && !defined(_ELL_) && !defined(_LDU_) && !defined(_DIV_)
        pEqn.solve();
#endif
        p_solve_time += clock.timeIncrement();

        if (pimple.finalNonOrthogonalIter())
        {
            phi = phiHbyA + pEqn.flux();
        }
        p_post_time += clock.timeIncrement();
    }
}

bool limitedp = pressureControl.limit(p);

// Thermodynamic density update
thermo.correctRho(psi*p - psip0);

if (limitedp)
{
    rho = thermo.rho();
}

#include "rhoEqn.H"
#include "compressibleContinuityErrs.H"

// Explicitly relax pressure for momentum corrector
p.relax();

U = HbyA - rAU*fvc::grad(p);
U.correctBoundaryConditions();
K = 0.5*magSqr(U);

if (pimple.simpleRho())
{
    rho = thermo.rho();
}

// Correct rhoUf if the mesh is moving
fvc::correctRhoUf(rhoUf, rho, U, phi);

if (thermo.dpdt())
{
    dpdt = fvc::ddt(p);

    if (mesh.moving())
    {
        dpdt -= fvc::div(fvc::meshPhi(rho, U), p);
    }
}

p_post_time += clock.timeIncrement();

double p_total_time = clock.elapsedTime();

Info << "p total Time : " << p_total_time << endl;
Info << "p pre Time : " << p_pre_time << endl;
Info << "p build Time : " << p_build_time << endl;
Info << "p convert Time : " << p_convert_time << endl;
Info << "p solve Time : " << p_solve_time << endl;
Info << "p post Time : " << p_post_time << endl;
Info << "p p_pimple_time Time : " << p_pimple_time << endl;
Info << "p other Time : " << p_total_time - p_pre_time - p_build_time - p_convert_time - p_solve_time - p_post_time - p_pimple_time << endl;

