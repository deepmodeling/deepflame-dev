Info<< "Reading thermophysical properties\n" << endl;
//autoPtr<psiReactionThermo> pThermo(psiReactionThermo::New(mesh));
//psiReactionThermo& thermo = pThermo();
//thermo.validate(args.executable(), "h", "e");

CanteraMixture::setEnergyName("hs");
// fluidThermo* pThermo = new heRhoThermo<rhoThermo, CanteraMixture>(mesh, word::null);
fluidThermo*   pThermo = new heRhoThermo<rhoThermo, dfSingleStepReactingMixture>(mesh, word::null);
fluidThermo& thermo = *pThermo;
SLGThermo slgThermo(mesh, thermo);

// basicSpecieMixture& composition = thermo.composition();
// PtrList<volScalarField>& Y = composition.Y();

// const word inertSpecie(thermo.lookup("inertSpecie"));
// if (!composition.species().found(inertSpecie))
// {
//     FatalIOErrorIn(args.executable().c_str(), thermo)
//         << "Inert specie " << inertSpecie << " not found in available species "
//         << composition.species()
//         << exit(FatalIOError);
// }

Info<< "Creating field rho\n" << endl;
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

volScalarField& p = thermo.p();

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

#include "createMRF.H"


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

Info<< "Creating combustion model\n" << endl;
autoPtr<CombustionModel<basicThermo>> combustion
(
    CombustionModel<basicThermo>::New(thermo, turbulence())
);

dfChemistryModel<basicThermo>* chemistry = combustion->chemistry();
PtrList<volScalarField>& Y = chemistry->Y();

#include "readGravitationalAcceleration.H"
#include "readhRef.H"
#include "gh.H"
#include "readpRef.H"

volScalarField p_rgh
(
    IOobject
    (
        "p_rgh",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

mesh.setFluxRequired(p_rgh.name());

#include "phrghEqn.H"


multivariateSurfaceInterpolationScheme<scalar>::fieldTable fields;

forAll(Y, i)
{
    fields.add(Y[i]);
}
fields.add(thermo.he());

volScalarField HRR
(
    IOobject
    (
        "HRR",
        runTime.timeName(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar(dimEnergy/dimVolume/dimTime, 0)
);

IOdictionary additionalControlsDict
(
    IOobject
    (
        "additionalControls",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ_IF_MODIFIED,
        IOobject::NO_WRITE
    )
);

Switch solvePrimaryRegion
(
    additionalControlsDict.lookup("solvePrimaryRegion")
);

Switch solvePyrolysisRegion
(
    additionalControlsDict.lookupOrDefault<bool>("solvePyrolysisRegion", true)
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

#include "createClouds.H"
#include "createSurfaceFilmModel.H"
// #include "createPyrolysisModel.H"
#include "createRadiationModel.H"
// #include "createFvOptions.H"
