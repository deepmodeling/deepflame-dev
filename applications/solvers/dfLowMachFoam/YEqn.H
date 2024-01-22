double Y_total_start, Y_total_end, Y_total_time = 0.; 
double Y_build_start, Y_build_end, Y_build_time = 0.; 
double Y_convert_start, Y_convert_end, Y_convert_time = 0.; 
double Y_update_start, Y_update_end, Y_update_time = 0.; 
double Y_solve_start, Y_solve_end, Y_solve_time = 0.;

double Y_build_old_start, Y_build_old_end, Y_build_old_time = 0.; 
double Y_build_new_start, Y_build_new_end, Y_build_new_time = 0.; 

hDiffCorrFlux = Zero;
diffAlphaD = Zero;
sumYDiffError = Zero;

start = MPI_Wtime();
Y_total_start = MPI_Wtime();
Y_build_start = MPI_Wtime();

#ifdef OPT_GenMatrix_Y
preProcess_Y(Y, phiUc, sumYDiffError, chemistry, upwindWeights, thermo.alpha(), phi);
#else
forAll(Y, i)
{
    sumYDiffError += chemistry->rhoD(i) * fvc::grad(Y[i]);;
}
phiUc = (linearInterpolate(sumYDiffError) & mesh.Sf()).ref();
#endif

#ifdef OPT_GenMatrix_Y_check
volVectorField sumYDiffErrorRef = sumYDiffError;
sumYDiffErrorRef = Zero;
forAll(Y, i)
{
    sumYDiffErrorRef += chemistry->rhoD(i) * fvc::grad(Y[i]);;
}
check_field_boundary_equal(sumYDiffErrorRef, sumYDiffError, "sumYDiffError");
const surfaceScalarField phiUcRef = (linearInterpolate(sumYDiffErrorRef) & mesh.Sf()).ref();
check_field_boundary_equal(phiUcRef, phiUc, "phiUc");
#endif

Y_build_end = MPI_Wtime();
Y_total_end = MPI_Wtime();
Y_build_time += Y_build_end - Y_build_start;
Y_total_time += Y_total_end - Y_total_start;
end = MPI_Wtime();
time_monitor_Y += end - start;

int flag_mpi_init;
MPI_Initialized(&flag_mpi_init);
if(flag_mpi_init) MPI_Barrier(PstreamGlobals::MPI_COMM_FOAM);

if (!splitting)
{
    start = MPI_Wtime();
    combustion->correct();
    if(flag_mpi_init) MPI_Barrier(PstreamGlobals::MPI_COMM_FOAM);
    end = MPI_Wtime();
    time_monitor_chem += end - start;
}

start = MPI_Wtime();
Y_total_start = MPI_Wtime();

volScalarField Yt(0.0*Y[0]);

forAll(Y, i)
{
    volScalarField& Yi = Y[i];

    if (i != inertIndex)
    {

        Y_build_start = MPI_Wtime();
#ifndef OPT_GenMatrix_Y
        Y_build_old_start = MPI_Wtime();
        tmp<fv::convectionScheme<scalar>> mvConvection
        (
            fv::convectionScheme<scalar>::New
            (
                mesh,
                fields,
                phi,
                mesh.divScheme("div(phi,Yi_h)")
            )
        );

        tmp<volScalarField> DEff = chemistry->rhoD(i) + turbulence->mut()/Sct;
        fvScalarMatrix YiEqn
        (
            fvm::ddt(rho, Yi)
            + mvConvection->fvmDiv(phi, Yi)
            + mvConvection->fvmDiv(phiUc, Yi)
            ==
            (
                splitting
            ?   fvm::laplacian(DEff(), Yi)
            :  (fvm::laplacian(DEff(), Yi) + combustion->R(Yi))
            )
        );
        Y_build_old_end = MPI_Wtime();
        Y_build_old_time += Y_build_old_end - Y_build_old_start;
#else
        Y_build_new_start = MPI_Wtime();
        tmp<fvScalarMatrix> tYiEqn = GenMatrix_Y(rho, Yi, phi, phiUc, chemistry->rhoD(i), turbulence->mut(), splitting, Sct, combustion(), upwindWeights);
        fvScalarMatrix& YiEqn = tYiEqn.ref();
        Y_build_new_end = MPI_Wtime();
        Y_build_new_time += Y_build_new_end - Y_build_new_start;
#endif
        YiEqn.relax();
        Y_build_end = MPI_Wtime();
        Y_build_time += Y_build_end - Y_build_start;

#ifdef OPT_GenMatrix_Y_check
        tmp<fv::convectionScheme<scalar>> mvConvection
        (
            fv::convectionScheme<scalar>::New
            (
                mesh,
                fields,
                phi,
                mesh.divScheme("div(phi,Yi_h)")
            )
        );
        Y_build_old_start = MPI_Wtime();
        tmp<volScalarField> DEff = chemistry->rhoD(i);
        fvScalarMatrix YiEqn_answer
        (
            fvm::ddt(rho, Yi)
            + mvConvection->fvmDiv(phi, Yi)
            + mvConvection->fvmDiv(phiUc, Yi)
            ==
            (
                splitting
            ?   fvm::laplacian(DEff(), Yi)
            :  (fvm::laplacian(DEff(), Yi) + combustion->R(Yi))
            )
        );
        Y_build_old_end = MPI_Wtime();
        Y_build_old_time += Y_build_old_end - Y_build_old_start;
        check_fvmatrix_equal(YiEqn_answer, YiEqn, "YiEqn");
#endif
#ifdef _DIV_
            Y_convert_start = MPI_Wtime();
            div.copy_value_from_fvMatrix(YiEqn);
            Y_convert_end = MPI_Wtime();
            Y_convert_time += Y_convert_end - Y_convert_start;
#endif
        Y_solve_start = MPI_Wtime();
#ifdef _DIV_
            auto& psi = const_cast<GeometricField<double, fvPatchField, volMesh>&>(YiEqn.psi());
            auto& source = YiEqn.source();
            auto& internalCoeffs = YiEqn.internalCoeffs();
            auto& boundaryCoeffs = YiEqn.boundaryCoeffs();
            div.solve(psi, source, internalCoeffs, boundaryCoeffs, "Yi");
#else
        YiEqn.solve("Yi");
#endif

        Y_solve_end = MPI_Wtime();
        Y_solve_time += Y_solve_end - Y_solve_start;

        Y_update_start = MPI_Wtime();
        Yi.max(0.0);
        Yt += Yi;
        Y_update_end = MPI_Wtime();
        Y_update_time += Y_update_end - Y_update_start;
    }
}

Y_update_start = MPI_Wtime();
Y[inertIndex] = scalar(1) - Yt;
Y[inertIndex].max(0.0);
Y_update_end = MPI_Wtime();
Y_update_time += Y_update_end - Y_update_start;

Y_total_end = MPI_Wtime();
Y_total_time += Y_total_end - Y_total_start;

end = MPI_Wtime();
time_monitor_Y += end - start;

Info << "Y total Time : " << Y_total_time << endl;
Info << "Y build Time : " << Y_build_time << endl;
Info << "Y build new Time : " << Y_build_new_time << endl;
Info << "Y build old Time : " << Y_build_old_time << endl;
Info << "Y convert Time : " << Y_convert_time << endl;
Info << "Y update Time : " << Y_update_time << endl;
Info << "Y solve Time : " << Y_solve_time << endl;
Info << "Y other Time : " << Y_total_time - Y_build_time - Y_convert_time - Y_solve_time - Y_update_time << endl;
