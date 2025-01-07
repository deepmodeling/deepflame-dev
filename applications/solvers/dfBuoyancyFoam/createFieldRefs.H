const volScalarField& psi = thermo.psi();
const volScalarField& T = thermo.T();
// regionModels::surfaceFilmModel& surfaceFilm = tsurfaceFilm();
const word inertSpecie(chemistry->lookup("inertSpecie"));
const label inertIndex(chemistry->species()[inertSpecie]);
