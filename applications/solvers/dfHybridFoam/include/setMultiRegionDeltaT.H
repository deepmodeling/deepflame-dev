/*---------------------------------------------------------------------------*\
    include/setMultiRegionDeltaT.H — Adjust deltaT from global Courant number
\*---------------------------------------------------------------------------*/

if (adjustTimeStep)
{
    if (CoNum == -great)
    {
        CoNum = small;
    }

    scalar maxDeltaTFactor = maxCo/(CoNum + small);

    scalar deltaTFactor =
        min
        (
            min(maxDeltaTFactor, 1.0 + 0.1*maxDeltaTFactor),
            1.2
        );

    runTime.setDeltaT
    (
        min
        (
            deltaTFactor*runTime.deltaT().value(),
            maxDeltaT
        )
    );

    Info<< "deltaT = " << runTime.deltaT().value() << endl;
}
