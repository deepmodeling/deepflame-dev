/*---------------------------------------------------------------------------*\
    include/setInitialMultiRegionDeltaT.H
\*---------------------------------------------------------------------------*/

if (adjustTimeStep)
{
    if ((runTime.timeIndex() == 0) && (CoNum > small))
    {
        runTime.setDeltaT
        (
            min
            (
                maxCo/CoNum*runTime.deltaT().value(),
                min(runTime.deltaTValue(), maxDeltaT)
            )
        );
        Info<< "Initial deltaT = " << runTime.deltaT().value() << endl;
    }
}
