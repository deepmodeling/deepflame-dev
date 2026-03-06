/*---------------------------------------------------------------------------*\
    createLowMachMeshes.H — Create fvMesh list for lowMach regions
\*---------------------------------------------------------------------------*/

const wordList lowMachNames(rp["lowMach"]);

PtrList<fvMesh> lowMachRegions(lowMachNames.size());

forAll(lowMachNames, i)
{
    Info<< "Create lowMach mesh for region " << lowMachNames[i]
        << " for time = " << runTime.timeName() << nl << endl;

    lowMachRegions.set
    (
        i,
        new fvMesh
        (
            IOobject
            (
                lowMachNames[i],
                runTime.timeName(),
                runTime,
                IOobject::MUST_READ
            )
        )
    );
}
