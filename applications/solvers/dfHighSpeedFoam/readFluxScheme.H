
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
    surfaceScalarField&,
    surfaceVectorField&,
    surfaceScalarField& 
);

std::map<word, FnPtr> fluxMap;
fluxMap["AUSMDV"] = AUSMDVFlux;
fluxMap["Kurganov"] = KNPFlux;
fluxMap["RoeC"] = RoeCFlux;

word fluxScheme("Kurganov");
mesh.schemesDict().readIfPresent("fluxScheme", fluxScheme);
