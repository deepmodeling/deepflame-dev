/*---------------------------------------------------------------------------*\
    createHighSpeedFields.H — Create PtrLists of fields for highSpeed regions.

    Follows the chtMultiRegionFoam pattern: allocate PtrList for each variable,
    then populate in a forAll loop over highSpeed region meshes.
\*---------------------------------------------------------------------------*/

// ---- Pointer lists for highSpeed region fields ----
PtrList<rhoThermo>                                      thermoHighSpeed(highSpeedRegions.size());
PtrList<volScalarField>                                 rhoHighSpeed(highSpeedRegions.size());
PtrList<volVectorField>                                 UHighSpeed(highSpeedRegions.size());
PtrList<volVectorField>                                 rhoUHighSpeed(highSpeedRegions.size());
PtrList<surfaceScalarField>                             phiHighSpeed(highSpeedRegions.size());
PtrList<surfaceScalarField>                             rhoPhiHighSpeed(highSpeedRegions.size());
PtrList<surfaceVectorField>                             rhoUPhiHighSpeed(highSpeedRegions.size());
PtrList<surfaceScalarField>                             rhoEPhiHighSpeed(highSpeedRegions.size());
PtrList<compressible::turbulenceModel>                  turbulenceHighSpeed(highSpeedRegions.size());
PtrList<CombustionModel<basicThermo>>                   combustionHighSpeed(highSpeedRegions.size());
PtrList<dfChemistryModel<basicThermo>>                  chemistryHighSpeed(highSpeedRegions.size());
PtrList<volScalarField>                                 rhoEHighSpeed(highSpeedRegions.size());
PtrList<SLGThermo>                                      slgThermoHighSpeed(highSpeedRegions.size());
PtrList<basicSprayCloud>                                parcelsHighSpeed(highSpeedRegions.size());
PtrList<Foam::fluxScheme>                               fluxSchemeHighSpeed(highSpeedRegions.size());
PtrList<IOMRFZoneList>                                  MRFHighSpeed(highSpeedRegions.size());

PtrList<multivariateSurfaceInterpolationScheme<scalar>::fieldTable>
    fieldsHighSpeed(highSpeedRegions.size());

PtrList<volScalarField>                                 diffAlphaDHighSpeed(highSpeedRegions.size());
PtrList<volVectorField>                                 hDiffCorrFluxHighSpeed(highSpeedRegions.size());
PtrList<volVectorField>                                 sumYDiffErrorHighSpeed(highSpeedRegions.size());
PtrList<volScalarField>                                 gradPHighSpeed(highSpeedRegions.size());
PtrList<volScalarField>                                 pMaxHighSpeed(highSpeedRegions.size());

// Per-region species conserved variables
PtrList<PtrList<volScalarField>>                        rhoYiHighSpeed(highSpeedRegions.size());
PtrList<PtrList<surfaceScalarField>>                    rhoPhiYiHighSpeed(highSpeedRegions.size());

// Field saves for RK time-stepping
PtrList<volScalarField>                                 rho_saveHighSpeed(highSpeedRegions.size());
PtrList<volVectorField>                                 rhoU_saveHighSpeed(highSpeedRegions.size());
PtrList<volScalarField>                                 rhoE_saveHighSpeed(highSpeedRegions.size());
PtrList<PtrList<volScalarField>>                        rhoYi_saveHighSpeed(highSpeedRegions.size());

List<label>                                             inertIndexHighSpeed(highSpeedRegions.size());
List<word>                                              ddtSchemesHighSpeed(highSpeedRegions.size());
List<word>                                              chemSchemesHighSpeed(highSpeedRegions.size());
List<bool>                                              inviscidHighSpeed(highSpeedRegions.size());
List<scalar>                                            SctHighSpeed(highSpeedRegions.size());

// ---- Populate highSpeed fields ----
forAll(highSpeedRegions, i)
{
    fvMesh& mesh = highSpeedRegions[i];

    Info<< "*** Reading highSpeed thermophysical properties for region "
        << mesh.name() << nl << endl;

    // Read time scheme
    ddtSchemesHighSpeed[i] = "Euler";
    mesh.schemesDict().readIfPresent("timeScheme", ddtSchemesHighSpeed[i]);
    Info<< "    ddtSchemes: " << ddtSchemesHighSpeed[i] << endl;

    chemSchemesHighSpeed[i] = "ode";
    mesh.schemesDict().readIfPresent("chemScheme", chemSchemesHighSpeed[i]);

    CanteraMixture::setEnergyName("ea");

    Info<< "    Adding to thermoHighSpeed\n" << endl;
    thermoHighSpeed.set
    (
        i,
        new heRhoThermo<rhoThermo, CanteraMixture>(mesh, word::null)
    );
    rhoThermo& thermo = thermoHighSpeed[i];

    slgThermoHighSpeed.set(i, new SLGThermo(mesh, thermo));

    // Read inviscid option
    dictionary thermoDict
    (
        IOdictionary
        (
            IOobject
            (
                "thermophysicalProperties",
                runTime.constant(),
                mesh,
                IOobject::MUST_READ_IF_MODIFIED,
                IOobject::NO_WRITE
            )
        )
    );
    inviscidHighSpeed[i] = thermoDict.lookupOrDefault("inviscid", false);

    volScalarField& p = thermo.p();
    const volScalarField& T = thermo.T();

    Info<< "    Adding to UHighSpeed\n" << endl;
    UHighSpeed.set
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

    Info<< "    Adding to rhoHighSpeed\n" << endl;
    rhoHighSpeed.set
    (
        i,
        new volScalarField
        (
            IOobject
            (
                "rho",
                runTime.timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            thermo.rho()
        )
    );

    Info<< "    Adding to rhoUHighSpeed\n" << endl;
    rhoUHighSpeed.set
    (
        i,
        new volVectorField
        (
            IOobject
            (
                "rhoU",
                runTime.timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            rhoHighSpeed[i]*UHighSpeed[i]
        )
    );

    Info<< "    Adding to phiHighSpeed\n" << endl;
    phiHighSpeed.set
    (
        i,
        new surfaceScalarField
        (
            IOobject
            (
                "phi",
                mesh.time().timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            mesh,
            dimensionedScalar("0", dimVelocity*dimArea, 0.0)
        )
    );

    Info<< "    Adding to rhoPhiHighSpeed\n" << endl;
    rhoPhiHighSpeed.set
    (
        i,
        new surfaceScalarField
        (
            IOobject
            (
                "rhoPhi",
                mesh.time().timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            mesh,
            dimensionedScalar("0", dimDensity*dimVelocity*dimArea, 0.0)
        )
    );

    Info<< "    Adding to rhoUPhiHighSpeed\n" << endl;
    rhoUPhiHighSpeed.set
    (
        i,
        new surfaceVectorField
        (
            IOobject
            (
                "rhoUPhi",
                mesh.time().timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            mesh,
            dimensionedVector("0", dimDensity*sqr(dimVelocity)*dimArea, Zero)
        )
    );

    Info<< "    Adding to rhoEPhiHighSpeed\n" << endl;
    rhoEPhiHighSpeed.set
    (
        i,
        new surfaceScalarField
        (
            IOobject
            (
                "rhoEPhi",
                mesh.time().timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            mesh,
            dimensionedScalar("0", dimDensity*pow3(dimVelocity)*dimArea, 0.0)
        )
    );

    Info<< "    Adding to fluxSchemeHighSpeed\n" << endl;
    fluxSchemeHighSpeed.set(i, Foam::fluxScheme::New(mesh).ptr());

    Info<< "    Adding MRFHighSpeed\n" << endl;
    MRFHighSpeed.set(i, new IOMRFZoneList(mesh));

    Info<< "    Adding to turbulenceHighSpeed\n" << endl;
    turbulenceHighSpeed.set
    (
        i,
        compressible::turbulenceModel::New
        (
            rhoHighSpeed[i],
            UHighSpeed[i],
            phiHighSpeed[i],
            thermo
        ).ptr()
    );

    Info<< "    Adding to combustionHighSpeed\n" << endl;
    combustionHighSpeed.set
    (
        i,
        CombustionModel<basicThermo>::New(thermo, turbulenceHighSpeed[i])
    );

    chemistryHighSpeed.set(i, combustionHighSpeed[i].chemistry());

    dfChemistryModel<basicThermo>& chemistry = chemistryHighSpeed[i];
    PtrList<volScalarField>& Y = chemistry.Y();
    const word inertSpecie(chemistry.lookup("inertSpecie"));
    inertIndexHighSpeed[i] = chemistry.species()[inertSpecie];
    chemistry.updateEnergy();

    SctHighSpeed[i] = chemistry.lookupOrDefault("Sct", 1.);

    volScalarField& ea = thermo.he();

    Info<< "    Adding to rhoEHighSpeed\n" << endl;
    rhoEHighSpeed.set
    (
        i,
        new volScalarField
        (
            IOobject
            (
                "rhoE",
                runTime.timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            rhoHighSpeed[i]*(ea + 0.5*magSqr(UHighSpeed[i]))
        )
    );

    chemistry.correctThermo();
    Info<< "    At init, min/max(T) = "
        << min(T).value() << ", " << max(T).value() << endl;

    Info<< "    Adding to fieldsHighSpeed\n" << endl;
    fieldsHighSpeed.set
    (
        i,
        new multivariateSurfaceInterpolationScheme<scalar>::fieldTable
    );
    forAll(Y, j)
    {
        fieldsHighSpeed[i].add(Y[j]);
    }
    fieldsHighSpeed[i].add(thermo.he());

    // Create rhoYi and rhoPhiYi
    const label nspecies = chemistry.species().size();
    rhoYiHighSpeed.set(i, new PtrList<volScalarField>(nspecies));
    rhoPhiYiHighSpeed.set(i, new PtrList<surfaceScalarField>(nspecies));

    forAll(Y, j)
    {
        rhoYiHighSpeed[i].set
        (
            j,
            new volScalarField
            (
                IOobject
                (
                    "rhoYi" + Y[j].name(),
                    runTime.timeName(),
                    mesh,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE
                ),
                rhoHighSpeed[i]*Y[j]
            )
        );

        rhoPhiYiHighSpeed[i].set
        (
            j,
            new surfaceScalarField
            (
                IOobject
                (
                    "rhoPhi" + Y[j].name(),
                    runTime.timeName(),
                    mesh,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE
                ),
                mesh,
                dimensionedScalar("0", dimDensity*dimVelocity*dimArea, 0.0)
            )
        );
    }

    // Field saves for RK
    rho_saveHighSpeed.set
    (
        i,
        new volScalarField("rho_save", rhoHighSpeed[i])
    );
    rhoU_saveHighSpeed.set
    (
        i,
        new volVectorField("rhoU_save", rhoHighSpeed[i]*UHighSpeed[i])
    );
    rhoE_saveHighSpeed.set
    (
        i,
        new volScalarField("rhoE_save", rhoEHighSpeed[i])
    );
    rhoYi_saveHighSpeed.set(i, new PtrList<volScalarField>(nspecies));
    forAll(Y, j)
    {
        rhoYi_saveHighSpeed[i].set
        (
            j,
            new volScalarField
            (
                IOobject
                (
                    "rhoYi_save" + Y[j].name(),
                    runTime.timeName(),
                    mesh,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE
                ),
                rhoHighSpeed[i]*Y[j]
            )
        );
    }

    // Gradient and pMax for diagnostics
    gradPHighSpeed.set
    (
        i,
        new volScalarField
        (
            IOobject
            (
                "gradP",
                runTime.timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            mag(fvc::grad(p))
        )
    );

    pMaxHighSpeed.set
    (
        i,
        new volScalarField
        (
            IOobject
            (
                "pMax",
                runTime.timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            p
        )
    );

    // Diffusion correction
    diffAlphaDHighSpeed.set
    (
        i,
        new volScalarField
        (
            IOobject("diffAlphaD", runTime.timeName(), mesh,
                IOobject::NO_READ, IOobject::NO_WRITE),
            mesh,
            dimensionedScalar(dimEnergy/dimTime/dimVolume, 0)
        )
    );
    hDiffCorrFluxHighSpeed.set
    (
        i,
        new volVectorField
        (
            IOobject("hDiffCorrFlux", runTime.timeName(), mesh,
                IOobject::NO_READ, IOobject::NO_WRITE),
            mesh,
            dimensionedVector(dimensionSet(1,0,-3,0,0,0,0), Zero)
        )
    );
    sumYDiffErrorHighSpeed.set
    (
        i,
        new volVectorField
        (
            IOobject("sumYDiffError", runTime.timeName(), mesh,
                IOobject::NO_READ, IOobject::NO_WRITE),
            mesh,
            dimensionedVector("sumYDiffError", dimDynamicViscosity/dimLength, Zero)
        )
    );

    turbulenceHighSpeed[i].validate();
}
