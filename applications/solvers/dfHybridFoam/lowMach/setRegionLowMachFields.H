/*---------------------------------------------------------------------------*\
    setRegionLowMachFields.H — Extract per-region references from PtrLists.

    This file is included inside the `forAll(lowMachRegions, i)` loop.
    It creates local references identical to what dfLowMachFoam uses,
    so the equation include files (UEqn.H, YEqn.H, etc.) can be reused
    without modification.
\*---------------------------------------------------------------------------*/

// Restore global energy name for lowMach region (highSpeed sets it to "ea")
CanteraMixture::setEnergyName("hs");

fvMesh& mesh = lowMachRegions[i];

fluidThermo& thermo = thermoLowMach[i];
const volScalarField& psi = thermo.psi();
volScalarField& p = thermo.p();
volScalarField& T = thermo.T();

volScalarField& rho = rhoLowMach[i];
volVectorField& U = ULowMach[i];
surfaceScalarField& phi = phiLowMach[i];

compressible::turbulenceModel& turbulence = turbulenceLowMach[i];
CombustionModel<basicThermo>* combustion = &combustionLowMach[i];
dfChemistryModel<basicThermo>* chemistry = &chemistryLowMach[i];
PtrList<volScalarField>& Y = chemistry->Y();
const label inertIndex = inertIndexLowMach[i];

volScalarField& K = KLowMach[i];
volScalarField& dpdt = dpdtLowMach[i];

multivariateSurfaceInterpolationScheme<scalar>::fieldTable& fields =
    fieldsLowMach[i];

pressureControl& presControl = pressureControlLowMach[i];
pimpleControl& pimple = pimpleLowMach[i];

volScalarField& diffAlphaD = diffAlphaDLowMach[i];
volVectorField& hDiffCorrFlux = hDiffCorrFluxLowMach[i];
volVectorField& sumYDiffError = sumYDiffErrorLowMach[i];

const word& combModelName = combModelNameLowMach[i];
const Switch& splitting = splittingLowMach[i];
const scalar Sct = SctLowMach[i];

const word turbName
(
    mesh.objectRegistry::lookupObject<IOdictionary>("turbulenceProperties")
        .lookup("simulationType")
);

scalar& cumulativeContErr = cumulativeContErrsLowMach[i];

// TODO: parcels support — uncomment when spray coupling is needed
// basicSprayCloud& parcels = parcelsLowMach[i];
