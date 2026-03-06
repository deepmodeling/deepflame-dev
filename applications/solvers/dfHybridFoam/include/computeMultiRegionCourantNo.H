/*---------------------------------------------------------------------------*\
    include/computeMultiRegionCourantNo.H
    
    Compute overall Courant number from all regions.
    - For lowMach: standard compressible Courant number
    - For highSpeed: central-upwind Courant number (wave-speed based)

    Sets global CoNum and meanCoNum as the maximum across all regions.
\*---------------------------------------------------------------------------*/

scalar CoNum = -great;
scalar meanCoNum = 0;

// ---- lowMach regions (compressible Courant number) ----
forAll(lowMachRegions, i)
{
    const fvMesh& mesh = lowMachRegions[i];
    const surfaceScalarField& phi = phiLowMach[i];
    const volScalarField& rho = rhoLowMach[i];

    scalar regionCoNum = 0;
    scalar regionMeanCoNum = 0;

    scalarField sumPhi
    (
        fvc::surfaceSum(mag(phi))().primitiveField()
    );

    regionCoNum = 0.5*gMax(sumPhi/(rho.primitiveField()*mesh.V().field()))
        * runTime.deltaTValue();

    regionMeanCoNum = 0.5*(gSum(sumPhi)
        / gSum(rho.primitiveField()*mesh.V().field()))
        * runTime.deltaTValue();

    Info<< "lowMach region " << mesh.name()
        << " Courant Number mean: " << regionMeanCoNum
        << " max: " << regionCoNum << endl;

    CoNum = max(CoNum, regionCoNum);
    meanCoNum = max(meanCoNum, regionMeanCoNum);
}

// ---- highSpeed regions (central-upwind wave-speed Courant) ----
forAll(highSpeedRegions, i)
{
    const fvMesh& mesh = highSpeedRegions[i];
    const volScalarField& rho = rhoHighSpeed[i];
    const volVectorField& U = UHighSpeed[i];
    const volVectorField& rhoU = rhoUHighSpeed[i];
    const rhoThermo& thermo = thermoHighSpeed[i];
    const volScalarField& psi = thermo.psi();

    scalar regionCoNum = 0;
    scalar regionMeanCoNum = 0;

    dimensionedScalar v_zero("v_zero", dimVolume/dimTime, 0.0);

    // --- Interpolation for central-upwind wave speeds ---
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

    volScalarField rPsi("rPsi", 1.0/psi);
    volScalarField c("c", sqrt(thermo.Cp()/thermo.Cv()*rPsi));

    surfaceScalarField cSf_pos
    (
        "cSf_pos",
        interpolate(c, pos, thermo.T().name())*mesh.magSf()
    );
    surfaceScalarField cSf_neg
    (
        "cSf_neg",
        interpolate(c, neg, thermo.T().name())*mesh.magSf()
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

    surfaceScalarField amaxSf("amaxSf", max(mag(am), mag(ap)));

    if (mesh.nInternalFaces())
    {
        scalarField sumAmaxSf(fvc::surfaceSum(amaxSf)().primitiveField());

        regionCoNum = 0.5*gMax(sumAmaxSf/mesh.V().field())
            * runTime.deltaTValue();

        regionMeanCoNum = 0.5*(gSum(sumAmaxSf)/gSum(mesh.V().field()))
            * runTime.deltaTValue();
    }

    Info<< "highSpeed region " << mesh.name()
        << " Courant Number mean: " << regionMeanCoNum
        << " max: " << regionCoNum << endl;

    CoNum = max(CoNum, regionCoNum);
    meanCoNum = max(meanCoNum, regionMeanCoNum);
}

Info<< "Overall Courant Number max: " << CoNum << endl;
