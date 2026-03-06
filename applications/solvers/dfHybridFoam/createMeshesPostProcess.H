#include "createMeshes.H"

if (!lowMachRegions.size() && !highSpeedRegions.size())
{
    FatalErrorIn(args.executable())
        << "No region meshes present" << exit(FatalError);
}

fvMesh& mesh = lowMachRegions.size() ? lowMachRegions[0] : highSpeedRegions[0];
