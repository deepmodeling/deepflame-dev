start = std::clock();

combustion->correct();

label flag_mpi_init;
MPI_Initialized(&flag_mpi_init);
if(flag_mpi_init) MPI_Barrier(PstreamGlobals::MPI_COMM_FOAM);
end = std::clock();
time_monitor_chem += double(end - start) / double(CLOCKS_PER_SEC);

volScalarField Yt(0.0*Y[0]);

start = std::clock();
forAll(Y, i)
{
    volScalarField& Yi = Y[i];

    if (i != inertIndex)
    {          
        rhoYi[i].ref() += chemistry->RR(i)*runTime.deltaT();
        Info <<"max reaction rate "<< Yi.name() << " is " << max(chemistry->RR(i)).value() << endl;

        Yi = rhoYi[i]/rho;
        Yi.max(0.0);

        Yi.correctBoundaryConditions();
        rhoYi[i] = rho*Yi;
        Yt += Yi;
    }
}

Y[inertIndex] = scalar(1) - Yt;
Y[inertIndex].max(0.0);
rhoYi[inertIndex] = rho*Y[inertIndex];

end = std::clock();
time_monitor_Y += double(end - start) / double(CLOCKS_PER_SEC);

chemistry->correctThermo();
Info<< "min/max(T) = "
        << min(T).value() << ", " << max(T).value() << endl;

p.ref() = rho()/psi();
p.correctBoundaryConditions();
rho.boundaryFieldRef() == psi.boundaryField()*p.boundaryField();


Info << " finish calculate Reaction" << nl << endl;

