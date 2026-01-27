#include "readGravitationalAcceleration.H"

Info<< "Reading thermophysical properties\n" << endl;
// fluidThermo* pThermo = new hePsiThermo<psiThermo, CanteraMixture>(mesh, word::null);
fluidThermo* pThermo = new heRhoThermo<rhoThermo, CanteraMixture>(mesh, word::null);
fluidThermo& thermo = *pThermo;
// thermo.validate(args.executable(), "ha");

SLGThermo slgThermo(mesh, thermo);

const volScalarField& psi = thermo.psi();
volScalarField& p = thermo.p();
volScalarField& T = thermo.T();
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

Info<< "\nReading field U\n" << endl;
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

#include "compressibleCreatePhi.H"

mesh.setFluxRequired(p.name());

dimensionedScalar rhoMax
(
    dimensionedScalar::lookupOrDefault
    (
        "rhoMax",
        pimple.dict(),
        dimDensity,
        great
    )
);

dimensionedScalar rhoMin
(
    dimensionedScalar::lookupOrDefault
    (
        "rhoMin",
        pimple.dict(),
        dimDensity,
        0
    )
);

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

const word turbName(mesh.objectRegistry::lookupObject<IOdictionary>("turbulenceProperties").lookup("simulationType"));

Info<< "Creating field dpdt\n" << endl;
volScalarField dpdt
(
    IOobject
    (
        "dpdt",
        runTime.timeName(),
        mesh
    ),
    mesh,
    dimensionedScalar(p.dimensions()/dimTime, 0)
);

Info<< "Creating reaction model\n" << endl;
autoPtr<CombustionModel<basicThermo>> combustion
(
    CombustionModel<basicThermo>::New(thermo, turbulence())
);

const word combModelName(mesh.objectRegistry::lookupObject<IOdictionary>
("combustionProperties").lookup("combustionModel"));
Info << "Combustion Model Name is confirmed as "<< combModelName << endl;


Info<< "end Creating reaction model\n" << endl;
dfChemistryModel<basicThermo>* chemistry = combustion->chemistry();
PtrList<volScalarField>& Y = chemistry->Y();
const word inertSpecie(chemistry->lookup("inertSpecie"));
const label inertIndex(chemistry->species()[inertSpecie]);
chemistry->setEnergyName("ha");
chemistry->updateEnergy();

if (combModelName != "flareFGM" && combModelName != "DeePFGM")
{
    chemistry->correctThermo();
    Info<< "At initial time, min/max(T) = " << min(T).value() << ", " << max(T).value() << endl;
}


Info<< "Creating field kinetic energy K\n" << endl;
volScalarField K("K", 0.5*magSqr(U));
multivariateSurfaceInterpolationScheme<scalar>::fieldTable fields;

forAll(Y, i)
{
    fields.add(Y[i]);
}
fields.add(thermo.he());

#include "createMRF.H"
#include "createClouds.H"
//#include "createRadiationModel.H"
//#include "createFvOptions.H"

// for diffusion velocity correction

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
