/*---------------------------------------------------------------------------*\
    lowMach/YEqn_RR.H — ODE-splitting reaction source step for lowMach region.

    Mirrors dfLowMachFoam/YEqn_RR.H: solves species equations with only the
    reaction source term every other timestep, using 2*dt.

    Requires `timeIndex` to be defined in the enclosing scope (dfHybridFoam.C).
\*---------------------------------------------------------------------------*/

if (!(timeIndex % 2))
{
    volScalarField Yt(0.0*Y[0]);

    scalar dtSave = runTime.deltaT().value();
    runTime.setDeltaT(dtSave * 2);

    combustion->correct();

    forAll(Y, i)
    {
        volScalarField& Yi = Y[i];

        if (i != inertIndex)
        {
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

    forAll(Y, i)
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
