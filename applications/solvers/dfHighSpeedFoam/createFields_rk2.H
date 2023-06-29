volScalarField rho_save("rho_save",thermo.rho()); 

volVectorField rhoU_save("rhoU_save",rho*U);

volScalarField rhoE_save("rhoE_save",rho*(ea + 0.5*magSqr(U))); 

PtrList<volScalarField> rhoYi_save(nspecies);
forAll(rhoYi_save,i)
{
    rhoYi_save.set
    (
        i,
        new volScalarField
        (
            IOobject
            (
                "rhoYi_save" + Y[i].name(),
                runTime.timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            rho*Y[i]
        )
    );
}

