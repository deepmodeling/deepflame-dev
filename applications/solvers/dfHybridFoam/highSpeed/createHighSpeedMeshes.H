/*---------------------------------------------------------------------------*\
    createHighSpeedMeshes.H — Create fvMesh list for highSpeed regions
\*---------------------------------------------------------------------------*/

const wordList highSpeedNames(rp["highSpeed"]);

PtrList<fvMesh> highSpeedRegions(highSpeedNames.size());

forAll(highSpeedNames, i)
{
    Info<< "Create highSpeed mesh for region " << highSpeedNames[i]
        << " for time = " << runTime.timeName() << nl << endl;

    highSpeedRegions.set
    (
        i,
        new fvMesh
        (
            IOobject
            (
                highSpeedNames[i],
                runTime.timeName(),
                runTime,
                IOobject::MUST_READ
            )
        )
    );
}
