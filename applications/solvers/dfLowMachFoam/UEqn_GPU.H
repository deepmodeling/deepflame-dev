// Solve the Momentum equation
start1 = std::clock();
int offset = 0;
const tmp<volScalarField> nuEff_tmp(turbulence->nuEff());
const volScalarField& nuEff = nuEff_tmp();
forAll(U.boundaryField(), patchi)
{
    const scalarField& patchP = p.boundaryField()[patchi];
    const vectorField& patchU = U.boundaryField()[patchi];
    const scalarField& patchRho = rho.boundaryField()[patchi];
    const scalarField& patchNuEff = nuEff.boundaryField()[patchi];

    int patchSize = patchP.size();

    // boundary pressure
    memcpy(boundary_pressure_init+offset, &patchP[0], patchSize*sizeof(double));
    // boundary velocity
    memcpy(boundary_velocity_init+3*offset, &patchU[0][0], 3*patchSize*sizeof(double));
    // boundary nuEff
    memcpy(boundary_nuEff_init+offset, &patchNuEff[0], patchSize*sizeof(double));
    // boundary rho
    memcpy(boundary_rho_init+offset, &patchRho[0], patchSize*sizeof(double));
    offset += patchSize;
}
end1 = std::clock();
time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
time_monitor_UEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);
time_monitor_UEqn_mtxAssembly_CPU_prepare += double(end1 - start1) / double(CLOCKS_PER_SEC);

start1 = std::clock();
UEqn_GPU.initializeTimeStep();
U.oldTime();
UEqn_GPU.fvm_ddt(&U.oldTime()[0][0]);
UEqn_GPU.fvm_div(boundary_pressure_init, boundary_velocity_init, boundary_nuEff_init, boundary_rho_init);
UEqn_GPU.fvc_grad(&p[0]);
UEqn_GPU.fvc_grad_vector();
UEqn_GPU.dev2T();
UEqn_GPU.fvc_div_tensor(&nuEff[0]);
UEqn_GPU.fvm_laplacian();
UEqn_GPU.sync();
end1 = std::clock();
time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
time_monitor_UEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);
time_monitor_UEqn_mtxAssembly_GPU_run += double(end1 - start1) / double(CLOCKS_PER_SEC);

// start2 = std::clock();
// fvVectorMatrix turb_source
// (
//     turbulence->divDevRhoReff(U)
// );
// end2 = std::clock();
// time_monitor_CPU += double(end2 - start2) / double(CLOCKS_PER_SEC);

// UEqn_GPU.add_fvMatrix(&turb_source.lower()[0], &turb_source.diag()[0], &turb_source.upper()[0], &turb_source.source()[0][0]);
// end1 = std::clock();
// time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
// time_monitor_UEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);

// check value
// U.oldTime();
// tmp<fvVectorMatrix> tUEqn
// (
    // fvm::ddt(rho, U) 
    // + 
    // fvm::div(phi, U)
    // + 
    // turbulence->divDevRhoReff(U) 
    // == -fvc::grad(p)
// );
// fvVectorMatrix& UEqn = tUEqn.ref();
// printf("b_cpu = %e\n", UEqn.source()[1][1]);
// forAll(U.boundaryField(), patchi){
    // labelUList sub_boundary = mesh.boundary()[patchi].faceCells();
    // forAll(sub_boundary, i){
    //     if (sub_boundary[i] == 1){
    //         printf("b_cpu_bou = %e\n", UEqn.boundaryCoeffs()[patchi][i][1]);
    //         printf("patchi = %d, i = %d\n", patchi, i);
    //     }
    // }
// }
// if (pimple.momentumPredictor())
// {
//     solve(UEqn);
//     Info << "U_CPU\n" << U << endl;
//     K = 0.5*magSqr(U);
// }
// UEqn_GPU.checkValue(true);

// start1 = std::clock();
// // // std::thread t(&dfMatrix::solve, &UEqn_GPU);
// UEqn_GPU.solve();
// end1 = std::clock();
// time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
// time_monitor_UEqn_solve += double(end1 - start1) / double(CLOCKS_PER_SEC);

// start1 = std::clock();
// // // t.join();
// // UEqn_GPU.updatePsi(&U[0][0]);
// K = 0.5*magSqr(U);
// end1 = std::clock();
// time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
// time_monitor_UEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);
// time_monitor_CPU += double(end1 - start1) / double(CLOCKS_PER_SEC);
// // Info << "U_amgx = " << U << endl;

