Info << "refine\n" << endl;
Info << "before refine cell number = " << U.size() << endl;

clockTime clock;

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

Info << "Refine.H time 0 : " << clock.timeIncrement() << endl;
 
// Store momentum to set rhoUf for introduced faces.
autoPtr<volVectorField> rhoU;
if (rhoUf.valid())
{
    rhoU = new volVectorField("rhoU", rho*U);
}

Info << "Refine.H time 1 : " << clock.timeIncrement() << endl;

// Do any mesh changes
mesh.update();

Info << "Refine.H time 2 : " << clock.timeIncrement() << endl;

bool changing = mesh.changing();

Info << "Refine.H time 3 : " << clock.timeIncrement() << endl;
Info << "Refine.H changing : " << changing << endl;
Info << "Refine.H correctPhi : " << changing << endl;

if (changing)
{
    if (correctPhi)
    {
        // Calculate absolute flux from the mapped surface velocity
        phi = mesh.Sf() & rhoUf();
        
        Info << "Refine.H time 4 : " << clock.timeIncrement() << endl;

        #include "correctPhi.H"

        Info << "Refine.H time 5 : " << clock.timeIncrement() << endl;

        // Make the fluxes relative to the mesh-motion
        fvc::makeRelative(phi, rho, U);

        Info << "Refine.H time 6 : " << clock.timeIncrement() << endl;
    }

    if (checkMeshCourantNo)
    {
        #include "meshCourantNo.H"
    }
    Info << "Refine.H time 7 : " << clock.timeIncrement() << endl;
}

refineLevel --;

Info << "after refine cell number = " << U.size() << endl;

