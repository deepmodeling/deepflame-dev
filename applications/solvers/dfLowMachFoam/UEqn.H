// Solve the Momentum equation
double Uconss = std::clock();
tmp<fvVectorMatrix> tUEqn
(
    fvm::ddt(rho, U) + fvm::div(phi, U)
  + turbulence->divDevRhoReff(U)
);
double Uconse = std::clock();
U_construct += double(Uconse -Uconss)/double(CLOCKS_PER_SEC);

MPI_Barrier(PstreamGlobals::MPI_COMM_FOAM);

fvVectorMatrix& UEqn = tUEqn.ref();
double Urelaxs = std::clock();
UEqn.relax();
double Urelaxe = std::clock();
U_relax += double(Urelaxe - Urelaxs)/double(CLOCKS_PER_SEC);

MPI_Barrier(PstreamGlobals::MPI_COMM_FOAM);

double Usolve_start= std::clock();
if (pimple.momentumPredictor())
{
    solve(UEqn == -fvc::grad(p));

    K = 0.5*magSqr(U);
}
//std::cout<< typeid(UEqn).name() << std::endl;
//double time_amgx_solve = UEqn.solve().solver().time_amgx();
double Usolve_end = std::clock();
U_solve += double(Usolve_end - Usolve_start)/double(CLOCKS_PER_SEC);