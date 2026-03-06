/*---------------------------------------------------------------------------*\
    setRegionHighSpeedFields.H — Extract per-region references from PtrLists.

    This file is included inside the `forAll(highSpeedRegions, i)` loop.
    It creates local references identical to what dfHighSpeedFoam uses.
\*---------------------------------------------------------------------------*/

// Restore global energy name for highSpeed region (lowMach sets it to "hs")
CanteraMixture::setEnergyName("ea");

fvMesh& mesh = highSpeedRegions[i];

rhoThermo& thermo = thermoHighSpeed[i];
volScalarField& p = thermo.p();
const volScalarField& T = thermo.T();
const volScalarField& psi = thermo.psi();
const volScalarField& mu = thermo.mu();

volScalarField& rho = rhoHighSpeed[i];
volVectorField& U = UHighSpeed[i];
volVectorField& rhoU = rhoUHighSpeed[i];
surfaceScalarField& phi = phiHighSpeed[i];
surfaceScalarField& rhoPhi = rhoPhiHighSpeed[i];
surfaceVectorField& rhoUPhi = rhoUPhiHighSpeed[i];
surfaceScalarField& rhoEPhi = rhoEPhiHighSpeed[i];

compressible::turbulenceModel& turbulence = turbulenceHighSpeed[i];
CombustionModel<basicThermo>* combustion = &combustionHighSpeed[i];
dfChemistryModel<basicThermo>* chemistry = &chemistryHighSpeed[i];
IOMRFZoneList& MRF = MRFHighSpeed[i];
PtrList<volScalarField>& Y = chemistry->Y();
const label inertIndex = inertIndexHighSpeed[i];
const label nspecies = chemistry->species().size();

volScalarField& ea = thermo.he();
volScalarField& rhoE = rhoEHighSpeed[i];

PtrList<volScalarField>& rhoYi = rhoYiHighSpeed[i];
PtrList<surfaceScalarField>& rhoPhiYi = rhoPhiYiHighSpeed[i];

volScalarField& rho_save = rho_saveHighSpeed[i];
volVectorField& rhoU_save = rhoU_saveHighSpeed[i];
volScalarField& rhoE_save = rhoE_saveHighSpeed[i];
PtrList<volScalarField>& rhoYi_save = rhoYi_saveHighSpeed[i];

Foam::fluxScheme& fluxSchemeFields = fluxSchemeHighSpeed[i];

multivariateSurfaceInterpolationScheme<scalar>::fieldTable& fields =
    fieldsHighSpeed[i];

volScalarField& diffAlphaD = diffAlphaDHighSpeed[i];
volVectorField& hDiffCorrFlux = hDiffCorrFluxHighSpeed[i];
volVectorField& sumYDiffError = sumYDiffErrorHighSpeed[i];
volScalarField& gradP = gradPHighSpeed[i];
volScalarField& pMax = pMaxHighSpeed[i];

const word& ddtSchemes = ddtSchemesHighSpeed[i];
const word& chemScheme = chemSchemesHighSpeed[i];
const bool inviscid = inviscidHighSpeed[i];
const scalar Sct = SctHighSpeed[i];

const word turbName
(
    mesh.objectRegistry::lookupObject<IOdictionary>("turbulenceProperties")
        .lookup("simulationType")
);

dimensionedScalar v_zero("v_zero", dimVolume/dimTime, 0.0);
