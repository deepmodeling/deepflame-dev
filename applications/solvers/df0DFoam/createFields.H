#include "createRDeltaT.H"

Info<< "Reading thermophysical properties\n" << endl;

fluidThermo* pThermo = new hePsiThermo<psiThermo, CanteraMixture>(mesh, word::null);
fluidThermo& thermo = *pThermo;

thermo.validate(args.executable(), "ha");

volScalarField& p = thermo.p();

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
        mesh
    ),
    mesh,
    dimensionedScalar(p.dimensions()/dimTime, 0)
);

Info<< "Creating field kinetic energy K\n" << endl;
volScalarField K("K", 0.5*magSqr(U));

multivariateSurfaceInterpolationScheme<scalar>::fieldTable fields;

#include "createMRF.H"
//#include "createFvOptions.H"



dfChemistryModel<basicThermo> chemistry(thermo);
PtrList<volScalarField>& Y = chemistry.Y();
const word inertSpecie(chemistry.lookup("inertSpecie"));
const label inertIndex(chemistry.species()[inertSpecie]);

forAll(Y, i)
{
    fields.add(Y[i]);
}
fields.add(thermo.he());

const word constProp = CanteraTorchProperties.subDict("zeroDReactor").lookup("constantProperty");

if
(
   (constProp != "pressure")
&& (constProp != "volume")
)
{
    FatalError
        << "in initialConditions, unknown constantProperty type "
        << constProp << nl
        << "    Valid types are: pressure volume."
        << exit(FatalError);
}

volScalarField& he = thermo.he();
scalar u0 = he[0] - p[0]/rho[0];

#ifdef USE_PYTORCH
    const Switch log_ = CanteraTorchProperties.subDict("TorchSettings").lookupOrDefault("log", false);
    const Switch torch_ = CanteraTorchProperties.subDict("TorchSettings").lookupOrDefault("torch", false);
#endif
#ifdef USE_LIBTORCH
    const Switch log_ = CanteraTorchProperties.subDict("TorchSettings").lookupOrDefault("log", false);
    const Switch torch_ = CanteraTorchProperties.subDict("TorchSettings").lookupOrDefault("torch", false);
#endif