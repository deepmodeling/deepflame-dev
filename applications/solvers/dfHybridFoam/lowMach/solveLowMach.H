/*---------------------------------------------------------------------------*\
    solveLowMach.H — Pressure-based PIMPLE solve for a single lowMach region.

    This mirrors the time-loop body of dfLowMachFoam.C but wrapped for the
    multi-region context. The equation include files (UEqn.H, YEqn.H, EEqn.H,
    pEqn.H, rhoEqn.H) are reused from dfLowMachFoam or duplicated locally
    with minor adaptations.

    Called inside forAll(lowMachRegions, i) after setRegionLowMachFields.H.
\*---------------------------------------------------------------------------*/
{
    // --- PIMPLE corrector loop for this lowMach region ---
    while (pimple.loop())
    {
        if (pimple.firstPimpleIter() && !pimple.simpleRho())
        {
            #include "lowMach/rhoEqn.H"
        }

        #include "lowMach/UEqn.H"

        if (combModelName != "ESF"
         && combModelName != "flareFGM"
         && combModelName != "DeePFGM"
         && combModelName != "FSD")
        {
            #include "lowMach/YEqn.H"
            #include "lowMach/EEqn.H"

            chemistry->correctThermo();
        }
        else
        {
            combustion->correct();
        }

        Info<< "min/max(T) = " << min(T).value() << ", "
            << max(T).value() << endl;

        // --- Pressure corrector loop
        while (pimple.correct())
        {
            #include "lowMach/pEqn.H"
        }

        if (pimple.turbCorr())
        {
            turbulence.correct();
        }
    }

    rho = thermo.rho();
}
