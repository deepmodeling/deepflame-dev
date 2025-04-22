Info<< "\nConstructing reacting cloud" << endl;
basicSprayCloud parcels
(
    "sprayCloud",
    rho,
    U,
    g,
    slgThermo
);

parcels.UTrans().writeOpt() = IOobject::NO_WRITE;
parcels.UCoeff().writeOpt() = IOobject::NO_WRITE;
parcels.hsTrans().writeOpt() = IOobject::NO_WRITE;
parcels.hsCoeff().writeOpt() = IOobject::NO_WRITE;

forAll(Y, i)
{
    parcels.rhoTrans(i).writeOpt() = IOobject::NO_WRITE;
}
