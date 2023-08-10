if (!(timeIndex % 2))
{
    volScalarField Yt(0.0*Y[0]);

    scalar dtSave = runTime.deltaT().value();
    runTime.setDeltaT(dtSave * 2);

    start = std::clock();
    combustion->correct();

    label flag_mpi_init;
    MPI_Initialized(&flag_mpi_init);
    if(flag_mpi_init) MPI_Barrier(MPI_COMM_WORLD);
    end = std::clock();
    time_monitor_chem += double(end - start) / double(CLOCKS_PER_SEC);

    forAll(Y, i)
    {
        volScalarField& Yi = Y[i];

        if (i != inertIndex)
        {
            volScalarField& Yi = Y[i];
            fvScalarMatrix YiEqn
            (
                fvm::ddt(rho, Yi)
                ==
                combustion->R(Yi)
            );

            YiEqn.relax();

            YiEqn.solve(mesh.solver("Yi"));

            Yi.max(0.0);
            Yt += Yi;
        }
    }
    Y[inertIndex] = scalar(1) - Yt;
    Y[inertIndex].max(0.0);

    forAll (Y, i)
    {
        volScalarField& tYi = Y[i].oldTime();

        forAll(tYi, celli)
        {
            tYi[celli] = Y[i][celli];
        }
        volScalarField::Boundary& Bf = tYi.boundaryFieldRef();
        forAll(Bf, patchi)
        {
            forAll(Bf[patchi], facei)
            {
                Bf[patchi][facei] = Y[i].boundaryField()[patchi][facei];
            }
        }
    }

    runTime.setDeltaT(dtSave);
}