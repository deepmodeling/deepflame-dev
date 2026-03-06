/*---------------------------------------------------------------------------*\
    include/readMultiRegionTimeControls.H — Read adjustable time step controls
\*---------------------------------------------------------------------------*/

const bool adjustTimeStep =
    runTime.controlDict().lookupOrDefault("adjustTimeStep", false);

scalar maxCo =
    runTime.controlDict().lookupOrDefault<scalar>("maxCo", 1.0);

scalar maxDeltaT =
    runTime.controlDict().lookupOrDefault<scalar>("maxDeltaT", great);
