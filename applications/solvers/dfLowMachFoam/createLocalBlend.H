/*---------------------------------------------------------------------------*\
This function is designed to read the UBlendingFactor for localBlended  
schemes used by the divSchemes, i.e.:
```
divSchemes
{
    div(phi,U)        Gauss localBlended linear upwind;
}
```
\*---------------------------------------------------------------------------*/

// Find and parse the divSchemes sub-dictionary
const dictionary& divSchemesDict = mesh.schemesDict().subDict("divSchemes");

Switch foundLocalBlended = false;

// Parse the divSchemes tokens
wordList keys = divSchemesDict.toc();
forAll(keys, i)
{
    const word& key = keys[i];
    // Convert to string
    ITstream& is = divSchemesDict.lookup(key);
    OStringStream os;
    os << is;
    word schemeStr = os.str();

    if (schemeStr.find("localBlended") != word::npos)
    {
        foundLocalBlended = true;
    }
}

// Create the localBlendingFactor field only if localBlended scheme is found
if (foundLocalBlended)
{
    Info << " Creating UBlendingFactor field for localBlended schemes " << endl;

    // Read the scalarField UBlend 
    volScalarField UBlend
    (
        IOobject
        (
            "UBlend",
            mesh.time().timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        ),
        mesh
    )

    // Construct the surfaceScalarField UBlendingFactor by interpolating UBlend to cell faces
    surfaceScalarField UBlendingFactor
    (
        IOobject
        (
            "UBlendingFactor",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        fvc::interpolate(UBlend)
    );
}
