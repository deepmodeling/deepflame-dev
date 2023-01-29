#include "createRDeltaT.H"

Info<< "Reading thermophysical properties\n" << endl;

fluidThermo* pThermo = new hePsiThermo<psiThermo, CanteraMixture>(mesh, word::null);
fluidThermo& thermo = *pThermo;
thermo.validate(args.executable(), "ha");

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

#include "compressibleCreatePhi.H"

pressureControl pressureControl(p, rho, pimple.dict(), false);

mesh.setFluxRequired(p.name());

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


Info<< "Creating reaction model\n" << endl;
autoPtr<CombustionModel<basicThermo>> combustion
(
    CombustionModel<basicThermo>::New(thermo, turbulence())
);
Info<< "end Creating reaction model\n" << endl;


//const word combModelName(combustion->type());
const word combModelName(mesh.objectRegistry::lookupObject<IOdictionary>("combustionProperties").lookup("combustionModel"));
Info << "Combustion Model Name is confirmed as "<< combModelName << endl;

dfChemistryModel<basicThermo>* chemistry = combustion->chemistry();
PtrList<volScalarField>& Y = chemistry->Y();
const word inertSpecie(chemistry->lookup("inertSpecie"));
const label inertIndex(chemistry->species()[inertSpecie]);


chemistry->correctThermo();
Info<< "At initial time, min/max(T) = " << min(T).value() << ", " << max(T).value() << endl;

//for dpdt

Info<< "Creating field kinetic energy K\n" << endl;
volScalarField K("K", 0.5*magSqr(U));

multivariateSurfaceInterpolationScheme<scalar>::fieldTable fields;

if(combModelName!="flareFGM")
{
forAll(Y, i)
{
    fields.add(Y[i]);
}
fields.add(thermo.he());
}


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
const Switch splitting = CanteraTorchProperties.lookupOrDefault("splittingStrategy", false);
#ifdef USE_PYTORCH
    const Switch log_ = CanteraTorchProperties.subDict("TorchSettings").lookupOrDefault("log", false);
    const Switch torch_ = CanteraTorchProperties.subDict("TorchSettings").lookupOrDefault("torch", false);
#endif
#ifdef USE_LIBTORCH
    const Switch log_ = CanteraTorchProperties.subDict("TorchSettings").lookupOrDefault("log", false);
    const Switch torch_ = CanteraTorchProperties.subDict("TorchSettings").lookupOrDefault("torch", false);
#endif
