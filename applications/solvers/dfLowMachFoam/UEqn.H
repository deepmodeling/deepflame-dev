// Solve the Momentum equation
double U_build_start, U_build_end, U_build_time = 0.;
double U_convert_start, U_convert_end, U_convert_time = 0.;
double U_solve_start, U_solve_end, U_solve_time = 0.;

double U_build_relax_start, U_build_relax_end, U_build_relax_time = 0.;

double U_build_org_start, U_build_org_end, U_build_org_time = 0.;
double U_build_opt_start, U_build_opt_end, U_build_opt_time = 0.;
double U_total_time = 0.;

double U_total_start = MPI_Wtime();

U_build_start = MPI_Wtime();

#ifndef OPT_GenMatrix_U
U_build_org_start = MPI_Wtime();
tmp<fvVectorMatrix> tUEqn
(
    fvm::ddt(rho, U) + fvm::div(phi, U)
    + turbulence->divDevRhoReff(U)
    == 
    -fvc::grad(p)
);
fvVectorMatrix& UEqn = tUEqn.ref();
U_build_org_end = MPI_Wtime();
U_build_org_time += U_build_org_end - U_build_org_start;
#else

U_build_opt_start = MPI_Wtime();
tmp<fvVectorMatrix> tUEqn = GenMatrix_U(rho, U, phi, p, turbulence());
fvVectorMatrix& UEqn = tUEqn.ref();
U_build_opt_end = MPI_Wtime();
U_build_opt_time = U_build_opt_end - U_build_opt_start;

#endif

U_build_end = MPI_Wtime();
U_build_time += U_build_end - U_build_start;

#ifdef OPT_GenMatrix_U_check
tmp<fvVectorMatrix> tUEqn_answer
(
    fvm::ddt(rho, U) + fvm::div(phi, U)
    + turbulence->divDevRhoReff(U)
    == 
    -fvc::grad(p)
);
fvVectorMatrix& UEqn_answer = tUEqn_answer.ref();
check_fvmatrix_equal(UEqn_answer, UEqn, "UEqn");
#endif

U_build_start = MPI_Wtime();
UEqn.relax();
U_build_end = MPI_Wtime();
U_build_time += U_build_end - U_build_start;

/*
fvVectorMatrix turb_source
(
    turbulence->divDevRhoReff(U)
);*/

assert(pimple.momentumPredictor());

if (pimple.momentumPredictor())
{
    // solve(UEqn_new);

#ifdef _DIV_
    U_convert_start = MPI_Wtime();
    div.copy_value_from_fvMatrix(UEqn);
    U_convert_end = MPI_Wtime();
    U_convert_time += U_convert_end - U_convert_start;
#endif
#ifdef _CSR_
    U_convert_start = MPI_Wtime();
    csrMatrix csr(UEqn);
    U_convert_end = MPI_Wtime();
    U_convert_time += U_convert_end - U_convert_start;
#endif
    U_solve_start = MPI_Wtime();
#ifdef _DIV_
    auto& psi = const_cast<volVectorField&>(UEqn.psi());
    auto& source = UEqn.source();
    auto& internalCoeffs = UEqn.internalCoeffs();
    auto& boundaryCoeffs = UEqn.boundaryCoeffs();
    div.solve(psi, source, internalCoeffs, boundaryCoeffs);
#endif
#ifdef _CSR_
    auto& psi = const_cast<volVectorField&>(UEqn.psi());
    auto& source = UEqn.source();
    auto& internalCoeffs = UEqn.internalCoeffs();
    auto& boundaryCoeffs = UEqn.boundaryCoeffs();
    csr.solve(psi, source, internalCoeffs, boundaryCoeffs);
#endif

#if !defined(_CSR_) && !defined(_ELL_) && !defined(_LDU_) && !defined(_DIV_)
    UEqn.solve();
#endif

    U_solve_end = MPI_Wtime();
    U_solve_time = U_solve_end - U_solve_start;

    K = 0.5*magSqr(U);
}

double U_total_end = MPI_Wtime();

U_total_time += U_total_end - U_total_start;

Info << "U total Time : " << U_total_time << endl;
Info << "U build Time : " << U_build_time << endl;
Info << "U build opt time : " << U_build_opt_time << endl;
Info << "U build org time : " << U_build_org_time << endl;
Info << "U convert Time : " << U_convert_time << endl;
Info << "U solve Time : " << U_solve_time << endl;
Info << "U other Time : " << U_total_time - U_build_time - U_convert_time - U_solve_time << endl;
