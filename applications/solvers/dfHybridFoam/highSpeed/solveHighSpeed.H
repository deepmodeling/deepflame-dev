/*---------------------------------------------------------------------------*\
    solveHighSpeed.H — Density-based central-upwind solve for a single
    highSpeed region (Euler time-stepping only, no RK in multi-region context).

    Called inside forAll(highSpeedRegions, i) after setRegionHighSpeedFields.H.

    This is a simplified version of dfHighSpeedFoam.C's time-loop body for
    the Euler time-stepping branch. RK support can be added later.
\*---------------------------------------------------------------------------*/
{
    // Update diagnostic fields
    gradP = mag(fvc::grad(p));
    forAll(pMax, celli)
    {
        if (p[celli] > pMax[celli])
        {
            pMax[celli] = p[celli];
        }
    }

    // --- Calculate flux using central-upwind scheme ---
    volScalarField rPsi("rPsi", 1.0/psi);
    volScalarField c("c", sqrt(thermo.Cp()/thermo.Cv()*rPsi));

    fluxSchemeFields.update
    (
        rho, rhoYi, nspecies, U, ea, p, c,
        phi, rhoPhi, rhoPhiYi, rhoUPhi, rhoEPhi
    );

    // --- Central-upwind face interpolation weights for sigmaDotU ---
    surfaceScalarField pos
    (
        IOobject("pos", runTime.timeName(), mesh),
        mesh, dimensionedScalar(dimless, 1.0)
    );
    surfaceScalarField neg
    (
        IOobject("neg", runTime.timeName(), mesh),
        mesh, dimensionedScalar(dimless, -1.0)
    );

    surfaceVectorField U_pos(interpolate(U, pos, U.name()));
    surfaceVectorField U_neg(interpolate(U, neg, U.name()));

    surfaceScalarField phiv_pos("phiv_pos", U_pos & mesh.Sf());
    surfaceScalarField phiv_neg("phiv_neg", U_neg & mesh.Sf());

    surfaceScalarField cSf_pos
    (
        "cSf_pos",
        interpolate(c, pos, T.name())*mesh.magSf()
    );
    surfaceScalarField cSf_neg
    (
        "cSf_neg",
        interpolate(c, neg, T.name())*mesh.magSf()
    );

    surfaceScalarField ap
    (
        "ap",
        max(max(phiv_pos + cSf_pos, phiv_neg + cSf_neg), v_zero)
    );
    surfaceScalarField am
    (
        "am",
        min(min(phiv_pos - cSf_pos, phiv_neg - cSf_neg), v_zero)
    );

    surfaceScalarField a_pos("a_pos", ap/(ap - am));

    word fluxSchemeName("Kurganov");
    mesh.schemesDict().readIfPresent("fluxScheme", fluxSchemeName);
    if (fluxSchemeName == "Tadmor")
    {
        a_pos = 0.5;
    }

    surfaceScalarField a_neg("a_neg", 1.0 - a_pos);

    volScalarField muEff("muEff", turbulence.muEff());
    volTensorField tauMC("tauMC", muEff*dev2(Foam::T(fvc::grad(U))));

    // --- Solve density ---
    #include "highSpeed/rhoEqn.H"

    // --- Solve momentum ---
    #include "highSpeed/rhoUEqn.H"

    // --- Solve species ---
    #include "highSpeed/rhoYEqn.H"

    // --- Solve energy ---
    #include "highSpeed/rhoEEqn.H"

    turbulence.correct();
}
