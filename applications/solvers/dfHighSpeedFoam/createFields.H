#include "readGravitationalAcceleration.H"
#include "createRDeltaT.H"

word ddtSchemes("Euler");
if (mesh.schemesDict().readIfPresent("timeScheme", ddtSchemes))
{
    if(ddtSchemes == "RK2SSP" || ddtSchemes == "RK3SSP" || ddtSchemes == "Euler")
    {
        Info<< "ddtSchemes: " << ddtSchemes << endl;
        if(ddtSchemes == "RK2SSP" || ddtSchemes == "RK3SSP")
        {
            Info<< "!! Note: RK2SSP and RK3SSP are not available for two-phase flow simulaiton. "
                << "If you want to simulate two-phase flows, please change time scheme to 'Euler' ."
                << endl;
        }
    }
    else
    {
        FatalErrorInFunction
            << "This timeScheme is not a valid choice. "
            << "Please use Euler, RK2SSP or RK3SSP scheme."
            << abort(FatalError);
    }
}

word chemScheme("ode");
mesh.schemesDict().readIfPresent("chemScheme", chemScheme);

if ((chemScheme == "direct") || (chemScheme == "ode"))
{
    Info<< "chemScheme: " << chemScheme << endl;
}
else
{
    FatalErrorInFunction
        << "chemScheme: " << chemScheme
        << " is not a valid choice. "
        << "Options are: 'ode' | 'direct'"
        << abort(FatalError);
}


if((ddtSchemes != "RK2SSP") && (ddtSchemes != "RK3SSP"))
{
    if(chemScheme == "direct")
    {
        FatalErrorInFunction
            << "This combination is not a valid choice. "
            << "If you want to use direct integrate for chemistry, please use RK2SSP or RK3SSP scheme."
            << abort(FatalError);
    }
}

Info<< "Reading thermophysical properties\n" << endl;

// psiThermo* pThermo = new hePsiThermo<psiThermo, CanteraMixture>(mesh, word::null);
// psiThermo& thermo = *pThermo;
rhoThermo* pThermo = new heRhoThermo<rhoThermo, CanteraMixture>(mesh, word::null);
rhoThermo& thermo = *pThermo;

SLGThermo slgThermo(mesh, thermo);

//move from creatFieldRefs.H to createFields.H
//p needed to be created before e
volScalarField& p = thermo.p();
const volScalarField& T = thermo.T();
const volScalarField& psi = thermo.psi();
const volScalarField& mu = thermo.mu();

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

bool inviscid(thermoDict.lookupOrDefault("inviscid",false));

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

volScalarField rho
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
);

volVectorField rhoU
(
    IOobject
    (
        "rhoU",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    rho*U
);

// surfaceScalarField phi("phi", fvc::flux(rhoU));
surfaceScalarField phi
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
);

Info<< "Creating flux properties\n" << endl;
surfaceScalarField rhoPhi
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
);

surfaceVectorField rhoUPhi
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
);

surfaceScalarField rhoEPhi
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
);

surfaceScalarField pos
(
    IOobject
    (
        "pos",
        runTime.timeName(),
        mesh
    ),
    mesh,
    dimensionedScalar(dimless, 1.0)
);

surfaceScalarField neg
(
    IOobject
    (
        "neg",
        runTime.timeName(),
        mesh
    ),
    mesh,
    dimensionedScalar(dimless, -1.0)
);

autoPtr<Foam::fluxScheme> fluxSchemeFields(Foam::fluxScheme::New(mesh));

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

multivariateSurfaceInterpolationScheme<scalar>::fieldTable fields;

Info<< "Creating reaction model\n" << endl;
autoPtr<CombustionModel<basicThermo>> combustion
(
    CombustionModel<basicThermo>::New(thermo, turbulence())
);
Info<< "end Creating reaction model\n" << endl;
dfChemistryModel<basicThermo>* chemistry = combustion->chemistry();
PtrList<volScalarField>& Y = chemistry->Y();
const word inertSpecie(chemistry->lookup("inertSpecie"));
const label inertIndex(chemistry->species()[inertSpecie]);
chemistry->setEnergyName("ea");
chemistry->updateEnergy();

volScalarField& ea = thermo.he();
volScalarField ha = ea + p/rho;

volScalarField rhoE
(
    IOobject
    (
        "rhoE",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    rho*(ea + 0.5*magSqr(U))
);

chemistry->correctThermo();
Info<< "At initial time, min/max(T) = " << min(T).value() << ", " << max(T).value() << endl;

forAll(Y, i)
{
    fields.add(Y[i]);
}
fields.add(thermo.he());

const label nspecies(chemistry->species().size());
PtrList<volScalarField> rhoYi(nspecies);
forAll(rhoYi,i)
{
    rhoYi.set
    (
        i,
        new volScalarField
        (
            IOobject
            (
                "rhoYi" + Y[i].name(),
                runTime.timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            rho*Y[i]
        )
    );
}

PtrList<surfaceScalarField> rhoPhiYi(nspecies);
forAll(rhoPhiYi,i)
{
    rhoPhiYi.set
    (
        i,
        new surfaceScalarField
        (
            IOobject
            (
                "rhoPhi" + Y[i].name(),
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

#include "createMRF.H"
#include "createClouds.H"

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

volScalarField gradP
(
    IOobject
    (
        "gradP",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mag(fvc::grad(thermo.p()))
);

volScalarField pMax
(
    IOobject
    (
        "pMax",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    thermo.p()
);
