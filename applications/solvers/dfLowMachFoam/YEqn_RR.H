Info << "Enter YEqn_RR" << endl;

if (!(timeIndex % 2))
{
    volScalarField Yt(0.0*Y[0]);

    scalar dtSave = runTime.deltaT().value();
    runTime.setDeltaT(dtSave * 2);

    start = MPI_Wtime();
    combustion->correct();

    int flag_mpi_init;
    MPI_Initialized(&flag_mpi_init);
    if(flag_mpi_init) MPI_Barrier(PstreamGlobals::MPI_COMM_FOAM);
    end = MPI_Wtime();
    time_monitor_chem += double(end - start);

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

            YiEqn.solve("Yi");

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

Info << "Exit YEqn_RR" << endl;