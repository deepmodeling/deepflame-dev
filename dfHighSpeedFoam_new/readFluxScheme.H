
typedef void (*FnPtr)
(
    const surfaceScalarField&,
    const surfaceScalarField&,
    const surfaceVectorField&,
    const surfaceScalarField&,
    const surfaceScalarField&,
    const surfaceVectorField&,
    const surfaceVectorField&,
    const surfaceScalarField&,
    const surfaceScalarField&,
    const surfaceScalarField&,
    const surfaceScalarField&,
    const surfaceScalarField&,
    const surfaceScalarField&,
    const surfaceScalarField&,
    const surfaceScalarField&,
    surfaceVectorField&,
    surfaceScalarField&,
    surfaceVectorField&,
    surfaceScalarField&
);

std::map<word, FnPtr> fluxMap;
fluxMap["AUSMDV"] = AUSMDVFlux;
fluxMap["Kurganov"] = KNPFlux;
//fluxMap["RoeC"] = RoeCFlux;

word fluxScheme("Kurganov");
mesh.schemesDict().readIfPresent("fluxScheme", fluxScheme);
if ((fluxScheme == "AUSMDV") || (fluxScheme == "Kurganov"))
{
    Info<< "fluxScheme: " << fluxScheme << endl;
}
else
{
    FatalErrorInFunction
        << "fluxScheme: " << fluxScheme
        << " is not a valid choice. "
        << "Options are: Tadmor, Kurganov"
        << abort(FatalError);
}

