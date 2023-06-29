start1 = std::clock();
tmp<fvVectorMatrix> tUEqn
(
    fvm::ddt(rho, U) + fvm::div(phi, U)
  + turbulence->divDevRhoReff(U) 
);
fvVectorMatrix& UEqn = tUEqn.ref();

end1 = std::clock();
time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
time_monitor_UEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);

UEqn.relax();
start1 = std::clock();
if (pimple.momentumPredictor())
{
    solve(UEqn == -fvc::grad(p));

    K = 0.5*magSqr(U);
}
end1 = std::clock();
time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
time_monitor_UEqn_solve += double(end1 - start1) / double(CLOCKS_PER_SEC);

