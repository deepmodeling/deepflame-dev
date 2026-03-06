/*---------------------------------------------------------------------------*\
    createLowMachFields.H — Create PtrLists of fields for lowMach regions.

    Follows the chtMultiRegionFoam pattern: allocate PtrList for each variable,
    then populate in a forAll loop over lowMach region meshes.
\*---------------------------------------------------------------------------*/

// ---- Pointer lists for lowMach region fields ----
PtrList<fluidThermo>                                    thermoLowMach(lowMachRegions.size());
PtrList<volScalarField>                                 rhoLowMach(lowMachRegions.size());
PtrList<volVectorField>                                 ULowMach(lowMachRegions.size());
PtrList<surfaceScalarField>                             phiLowMach(lowMachRegions.size());
PtrList<compressible::turbulenceModel>                  turbulenceLowMach(lowMachRegions.size());
PtrList<CombustionModel<basicThermo>>                   combustionLowMach(lowMachRegions.size());
PtrList<dfChemistryModel<basicThermo>>                  chemistryLowMach(lowMachRegions.size());
PtrList<volScalarField>                                 KLowMach(lowMachRegions.size());
PtrList<volScalarField>                                 dpdtLowMach(lowMachRegions.size());
PtrList<volScalarField>                                 p_rghLowMach(lowMachRegions.size());
PtrList<pressureControl>                                pressureControlLowMach(lowMachRegions.size());
PtrList<pimpleControl>                                  pimpleLowMach(lowMachRegions.size());
PtrList<SLGThermo>                                      slgThermoLowMach(lowMachRegions.size());
PtrList<basicSprayCloud>                                parcelsLowMach(lowMachRegions.size());

PtrList<multivariateSurfaceInterpolationScheme<scalar>::fieldTable>
    fieldsLowMach(lowMachRegions.size());

PtrList<volScalarField>                                 diffAlphaDLowMach(lowMachRegions.size());
PtrList<volVectorField>                                 hDiffCorrFluxLowMach(lowMachRegions.size());
PtrList<volVectorField>                                 sumYDiffErrorLowMach(lowMachRegions.size());
PtrList<IOdictionary>                                   CanteraTorchPropertiesLowMach(lowMachRegions.size());

List<scalar>                                            initialMassLowMach(lowMachRegions.size());
List<label>                                             inertIndexLowMach(lowMachRegions.size());
List<word>                                              combModelNameLowMach(lowMachRegions.size());
List<Switch>                                            splittingLowMach(lowMachRegions.size());
List<scalar>                                            SctLowMach(lowMachRegions.size());

List<scalar> cumulativeContErrsLowMach(lowMachRegions.size(), scalar(0));

// ---- Populate lowMach fields ----
forAll(lowMachRegions, i)
{
    fvMesh& mesh = lowMachRegions[i];

    Info<< "*** Reading lowMach thermophysical properties for region "
        << mesh.name() << nl << endl;

    CanteraMixture::setEnergyName("hs");

    Info<< "    Adding to thermoLowMach\n" << endl;
    thermoLowMach.set
    (
        i,
        new heRhoThermo<rhoThermo, CanteraMixture>(mesh, word::null)
    );
    fluidThermo& thermo = thermoLowMach[i];

    Info<< "    Adding to slgThermoLowMach\n" << endl;
    slgThermoLowMach.set(i, new SLGThermo(mesh, thermo));

    Info<< "    Adding to rhoLowMach\n" << endl;
    rhoLowMach.set
    (
        i,
        new volScalarField
        (
            IOobject
            (
                "rho",
                runTime.timeName(),
                mesh,
                IOobject::READ_IF_PRESENT,
                IOobject::AUTO_WRITE
            ),
            thermo.rho()
        )
    );

    Info<< "    Adding to ULowMach\n" << endl;
    ULowMach.set
    (
        i,
        new volVectorField
        (
            IOobject
            (
                "U",
                runTime.timeName(),
                mesh,
                IOobject::MUST_READ,
                IOobject::AUTO_WRITE
            ),
            mesh
        )
    );

    Info<< "    Adding to phiLowMach\n" << endl;
    phiLowMach.set
    (
        i,
        new surfaceScalarField
        (
            IOobject
            (
                "phi",
                runTime.timeName(),
                mesh,
                IOobject::READ_IF_PRESENT,
                IOobject::AUTO_WRITE
            ),
            linearInterpolate(rhoLowMach[i]*ULowMach[i])
                & mesh.Sf()
        )
    );

    Info<< "    Adding to pimpleLowMach\n" << endl;
    pimpleLowMach.set(i, new pimpleControl(mesh));

    Info<< "    Adding to turbulenceLowMach\n" << endl;
    turbulenceLowMach.set
    (
        i,
        compressible::turbulenceModel::New
        (
            rhoLowMach[i],
            ULowMach[i],
            phiLowMach[i],
            thermo
        ).ptr()
    );

    Info<< "    Adding to combustionLowMach\n" << endl;
    combustionLowMach.set
    (
        i,
        CombustionModel<basicThermo>::New(thermo, turbulenceLowMach[i])
    );

    combModelNameLowMach[i] = word
    (
        mesh.objectRegistry::lookupObject<IOdictionary>("combustionProperties")
            .lookup("combustionModel")
    );
    Info<< "    Combustion Model: " << combModelNameLowMach[i] << endl;

    chemistryLowMach.set(i, combustionLowMach[i].chemistry());

    dfChemistryModel<basicThermo>& chemistry = chemistryLowMach[i];
    PtrList<volScalarField>& Y = chemistry.Y();
    forAll(Y, j)
    {
        Y[j].correctBoundaryConditions();
    }
    const word inertSpecie(chemistry.lookup("inertSpecie"));
    inertIndexLowMach[i] = chemistry.species()[inertSpecie];
    chemistry.updateEnergy();

    SctLowMach[i] = chemistry.lookupOrDefault("Sct", 1.);

    if (combModelNameLowMach[i] != "ESF"
     && combModelNameLowMach[i] != "flareFGM"
     && combModelNameLowMach[i] != "DeePFGM"
     && combModelNameLowMach[i] != "FSD")
    {
        chemistry.correctThermo();
        Info<< "    At init, min/max(T) = "
            << min(thermo.T()).value() << ", "
            << max(thermo.T()).value() << endl;
    }

    rhoLowMach[i] = thermo.rho();

    Info<< "    Adding to pressureControlLowMach\n" << endl;
    pressureControlLowMach.set
    (
        i,
        new pressureControl(thermo.p(), rhoLowMach[i], pimpleLowMach[i].dict(), false)
    );

    mesh.setFluxRequired(thermo.p().name());

    Info<< "    Adding to dpdtLowMach\n" << endl;
    dpdtLowMach.set
    (
        i,
        new volScalarField
        (
            IOobject
            (
                "dpdt",
                runTime.timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            mesh,
            dimensionedScalar("dpdt", thermo.p().dimensions()/dimTime, 0)
        )
    );

    Info<< "    Adding to KLowMach\n" << endl;
    KLowMach.set
    (
        i,
        new volScalarField("K", 0.5*magSqr(ULowMach[i]))
    );

    initialMassLowMach[i] = fvc::domainIntegrate(rhoLowMach[i]).value();

    Info<< "    Adding to fieldsLowMach\n" << endl;
    fieldsLowMach.set
    (
        i,
        new multivariateSurfaceInterpolationScheme<scalar>::fieldTable
    );
    if (combModelNameLowMach[i] != "ESF"
     && combModelNameLowMach[i] != "flareFGM"
     && combModelNameLowMach[i] != "DeePFGM"
     && combModelNameLowMach[i] != "FSD")
    {
        forAll(Y, j)
        {
            fieldsLowMach[i].add(Y[j]);
        }
        fieldsLowMach[i].add(thermo.he());
    }

    Info<< "    Adding to diffAlphaDLowMach\n" << endl;
    diffAlphaDLowMach.set
    (
        i,
        new volScalarField
        (
            IOobject
            (
                "diffAlphaD",
                runTime.timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            mesh,
            dimensionedScalar(dimEnergy/dimTime/dimVolume, 0)
        )
    );

    Info<< "    Adding to hDiffCorrFluxLowMach\n" << endl;
    hDiffCorrFluxLowMach.set
    (
        i,
        new volVectorField
        (
            IOobject
            (
                "hDiffCorrFlux",
                runTime.timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            mesh,
            dimensionedVector(dimensionSet(1,0,-3,0,0,0,0), Zero)
        )
    );

    Info<< "    Adding to sumYDiffErrorLowMach\n" << endl;
    sumYDiffErrorLowMach.set
    (
        i,
        new volVectorField
        (
            IOobject
            (
                "sumYDiffError",
                runTime.timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            mesh,
            dimensionedVector("sumYDiffError", dimDynamicViscosity/dimLength, Zero)
        )
    );

    Info<< "    Adding to CanteraTorchPropertiesLowMach\n" << endl;
    CanteraTorchPropertiesLowMach.set
    (
        i,
        new IOdictionary
        (
            IOobject
            (
                "CanteraTorchProperties",
                runTime.constant(),
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE
            )
        )
    );

    splittingLowMach[i] = CanteraTorchPropertiesLowMach[i].lookupOrDefault
    (
        "splittingStrategy", false
    );

    turbulenceLowMach[i].validate();
}
