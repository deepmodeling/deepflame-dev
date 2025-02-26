#include "createRDeltaT.H"
#include "readGravitationalAcceleration.H"

Info<< "Reading thermophysical properties\n" << endl;

CanteraMixture::setEnergyName("hs");
// fluidThermo* pThermo = new hePsiThermo<psiThermo, CanteraMixture>(mesh, word::null);
fluidThermo* pThermo = new heRhoThermo<rhoThermo, CanteraMixture>(mesh, word::null);
fluidThermo& thermo = *pThermo;
Info<< "Readed thermophysical properties\n" << endl;
SLGThermo slgThermo(mesh, thermo);

const volScalarField& psi = thermo.psi();
volScalarField& p = thermo.p();
volScalarField& T = thermo.T();
T.correctBoundaryConditions();
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

U.correctBoundaryConditions();
p.correctBoundaryConditions();

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


const word combModelName(mesh.objectRegistry::lookupObject<IOdictionary>("combustionProperties").lookup("combustionModel"));
Info << "Combustion Model Name is confirmed as "<< combModelName << endl;

const word turbName(mesh.objectRegistry::lookupObject<IOdictionary>("turbulenceProperties").lookup("simulationType"));

dfChemistryModel<basicThermo>* chemistry = combustion->chemistry();
PtrList<volScalarField>& Y = chemistry->Y();
forAll(Y, i)
{
    Y[i].correctBoundaryConditions();
}
const word inertSpecie(chemistry->lookup("inertSpecie"));
const label inertIndex(chemistry->species()[inertSpecie]);
chemistry->updateEnergy();


if(combModelName!="ESF" && combModelName!="flareFGM"  && combModelName!="DeePFGM" && combModelName!="FSD")
{
    chemistry->correctThermo();
    Info<< "At initial time, min/max(T) = " << min(T).value() << ", " << max(T).value() << endl;
}
rho = thermo.rho();

//for dpdt

Info<< "Creating field kinetic energy K\n" << endl;
volScalarField K("K", 0.5*magSqr(U));

multivariateSurfaceInterpolationScheme<scalar>::fieldTable fields;

#include "createMRF.H"
#include "createClouds.H"

if(combModelName!="ESF" && combModelName!="flareFGM"  && combModelName!="DeePFGM" && combModelName!="FSD")
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

volScalarField UEqn_A
(
    IOobject
    (
        "A("+U.name()+')',
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar(dimensionSet(1,-3,-1,0,0,0,0), Zero),
    extrapolatedCalculatedFvPatchScalarField::typeName
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
