Info << "refine\n" << endl;
Info << "before refine cell number = " << U.size() << endl;

// Store divrhoU from the previous mesh so that it can be mapped
// and used in correctPhi to ensure the corrected phi has the
// same divergence
autoPtr<volScalarField> divrhoU;
if (correctPhi)
{
    divrhoU = new volScalarField
    (
        "divrhoU",
        fvc::div(fvc::absolute(phi, rho, U))
    );
}

 
// Store momentum to set rhoUf for introduced faces.
autoPtr<volVectorField> rhoU;
if (rhoUf.valid())
{
    rhoU = new volVectorField("rhoU", rho*U);
}

// Do any mesh changes
mesh.update();

bool changing = mesh.changing();

if (changing)
{
    if (correctPhi)
    {
        // Calculate absolute flux from the mapped surface velocity
        phi = mesh.Sf() & rhoUf();
        
        #include "correctPhi.H"

        // Make the fluxes relative to the mesh-motion
        fvc::makeRelative(phi, rho, U);
    }

    if (checkMeshCourantNo)
    {
        #include "meshCourantNo.H"
    }
}

refineLevel --;

Info << "after refine cell number = " << U.size() << endl;

