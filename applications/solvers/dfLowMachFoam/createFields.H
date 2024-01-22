clockTime createFieldsClock;

#include "createRDeltaT.H"

Info << "createFields time 0 : " << createFieldsClock.timeIncrement() << endl;

IOdictionary CanteraTorchProperties
(
    IOobject
    (
        "CanteraTorchProperties",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ,
        IOobject::NO_WRITE
    )
);

Info << "createFields time 1 : " << createFieldsClock.timeIncrement() << endl;


Info<< "Reading thermophysical properties\n" << endl;

fluidThermo* pThermo = new hePsiThermo<psiThermo, CanteraMixture>(mesh, word::null);
fluidThermo& thermo = *pThermo;

Info << "createFields time 2 : " << createFieldsClock.timeIncrement() << endl;


thermo.validate(args.executable(), "ha");

Info << "createFields time 3 : " << createFieldsClock.timeIncrement() << endl;

volScalarField& T = thermo.T();

const volScalarField& psi = thermo.psi();
volScalarField& p = thermo.p();

Info << "createFields time 4 : " << createFieldsClock.timeIncrement() << endl;

volScalarField rho
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
);

Info << "createFields time 5 : " << createFieldsClock.timeIncrement() << endl;


Info<< "Reading field U\n" << endl;
volVectorField U
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
);

Info << "createFields time 6 : " << createFieldsClock.timeIncrement() << endl;

#include "compressibleCreatePhi.H"

Info << "createFields time 7 : " << createFieldsClock.timeIncrement() << endl;

pressureControl pressureControl(p, rho, pimple.dict(), false);

Info << "createFields time 8 : " << createFieldsClock.timeIncrement() << endl;

mesh.setFluxRequired(p.name());

Info << "createFields time 9 : " << createFieldsClock.timeIncrement() << endl;

Info<< "Creating turbulence model\n" << endl;
autoPtr<compressible::turbulenceModel> turbulence
(
    compressible::turbulenceModel::New
    (
        rho,
        U,
        phi,
        thermo
    )
);

Info << "createFields time 10 : " << createFieldsClock.timeIncrement() << endl;

Info<< "Creating field dpdt\n" << endl;
volScalarField dpdt
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
    dimensionedScalar("dpdt",p.dimensions()/dimTime, 0)
);

Info << "createFields time 11 : " << createFieldsClock.timeIncrement() << endl;

Info<< "Creating reaction model\n" << endl;
autoPtr<CombustionModel<basicThermo>> combustion
(
    CombustionModel<basicThermo>::New(thermo, turbulence())
);
Info<< "end Creating reaction model\n" << endl;

Info << "createFields time 12 : " << createFieldsClock.timeIncrement() << endl;

//const word combModelName(combustion->type());
const word combModelName(mesh.objectRegistry::lookupObject<IOdictionary>("combustionProperties").lookup("combustionModel"));
Info << "Combustion Model Name is confirmed as "<< combModelName << endl;

Info << "createFields time 13 : " << createFieldsClock.timeIncrement() << endl;

dfChemistryModel<basicThermo>* chemistry = combustion->chemistry();
PtrList<volScalarField>& Y = chemistry->Y();

Info << "createFields time 14 : " << createFieldsClock.timeIncrement() << endl;

const word inertSpecie(chemistry->lookup("inertSpecie"));
const label inertIndex(chemistry->species()[inertSpecie]);

Info << "createFields time 15 : " << createFieldsClock.timeIncrement() << endl;

// Info<< "At initial time, min/max(T) = " << min(T).value() << ", " << max(T).value() << endl;

//for dpdt

Info<< "Creating field kinetic energy K\n" << endl;
volScalarField K("K", 0.5*magSqr(U));

Info << "createFields time 16 : " << createFieldsClock.timeIncrement() << endl;

multivariateSurfaceInterpolationScheme<scalar>::fieldTable fields;

if(combModelName!="flareFGM")
{
forAll(Y, i)
{
    fields.add(Y[i]);
}
fields.add(thermo.he());
}

Info << "createFields time 17 : " << createFieldsClock.timeIncrement() << endl;

const scalar Sct = chemistry->lookupOrDefault("Sct", 1.);
volScalarField diffAlphaD
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
);

Info << "createFields time 18 : " << createFieldsClock.timeIncrement() << endl;

volVectorField hDiffCorrFlux
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
);

Info << "createFields time 19 : " << createFieldsClock.timeIncrement() << endl;

volVectorField sumYDiffError
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
);

Info << "createFildes_time_20 : " << createFieldsClock.timeIncrement() << endl;

const Switch splitting = CanteraTorchProperties.lookupOrDefault("splittingStrategy", false);
// switch for field initialization
bool initialized = false;
bool renumberMesh = false;
int refineLevel = CanteraTorchProperties.subDict("RefineSettings").lookupOrDefault("refineLevel", 0);

Info << "createFildes_time_21 : " << createFieldsClock.timeIncrement() << endl;

const surfaceScalarField linear_weights = mesh.surfaceInterpolation::weights(); // interpolation weight (linear)

#ifdef USE_PYTORCH
    const Switch log_ = CanteraTorchProperties.subDict("TorchSettings").lookupOrDefault("log", false);
    const Switch torch_ = CanteraTorchProperties.subDict("TorchSettings").lookupOrDefault("torch", false);
#endif
#ifdef USE_LIBTORCH
    const Switch log_ = CanteraTorchProperties.subDict("TorchSettings").lookupOrDefault("log", false);
    const Switch torch_ = CanteraTorchProperties.subDict("TorchSettings").lookupOrDefault("torch", false);
#endif
// 

Info << "createFildes_time_22 : " << createFieldsClock.timeIncrement() << endl;
