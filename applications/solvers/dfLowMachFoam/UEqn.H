Info << "Enter UEqn" << endl;
assert(pimple.momentumPredictor());

clockTime UEqnClock;

// Solve the Momentum equation
#ifndef OPT_GenMatrix_U
tmp<fvVectorMatrix> tUEqn
(
    fvm::ddt(rho, U) + fvm::div(phi, U)
    + turbulence->divDevRhoReff(U) 
    ==
    -fvc::grad(p)
);
fvVectorMatrix& UEqn = tUEqn.ref();
#else

tmp<fvVectorMatrix> tUEqn = GenMatrix_U(rho, U, phi, p, turbulence());
fvVectorMatrix& UEqn = tUEqn.ref();

#endif

time_monitor_UEqn_pre += UEqnClock.timeIncrement();

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

time_monitor_UEqn_pre_check += UEqnClock.timeIncrement();

{
    time_monitor_UEqn_pre += UEqnClock.timeIncrement();

#ifdef USE_DF_MATRIX
    // dfMatrix df_UEqnSolve(UEqn.ref(), regionPtr);
    auto& psi = const_cast<volVectorField&>(UEqn.psi());
    auto& source = UEqn.source();
    auto& internalCoeffs = UEqn.internalCoeffs();
    auto& boundaryCoeffs = UEqn.boundaryCoeffs();

    matrixPtr->valueCopy(UEqn);

    time_monitor_UEqn_convert += UEqnClock.timeIncrement();

    // df_UEqnSolve.solve(psi, source, internalCoeffs, boundaryCoeffs);
    matrixPtr->solve(psi, source, internalCoeffs, boundaryCoeffs);
#else
    // UEqnSolver.solve();
    UEqn.solve();
#endif

    time_monitor_UEqn_solve += UEqnClock.timeIncrement(); 

    K.oldTime();
    K = 0.5*magSqr(U);

}

time_monitor_UEqn_post += UEqnClock.timeIncrement();

time_monitor_UEqn += UEqnClock.elapsedTime();

Info << "Exit UEqn" << endl;
