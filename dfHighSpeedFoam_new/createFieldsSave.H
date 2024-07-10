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

List<scalar> rkcoe1(3); 
List<scalar> rkcoe2(3);
List<scalar> rkcoe3(3);
scalar rk=2;
label nrk=0;

if(ddtSchemes == "RK2SSP")
{
    rkcoe1[0]=1.0; rkcoe2[0]=0.0; rkcoe3[0]=1.0;
    rkcoe1[1]=0.5; rkcoe2[1]=0.5; rkcoe3[1]=0.5;
    rkcoe1[2]=0.0; rkcoe2[2]=0.0; rkcoe3[2]=0.0;
}
else if(ddtSchemes == "RK3SSP")
{
    rkcoe1[0]=1.0; rkcoe2[0]=0.0; rkcoe3[0]=1.0;
    rkcoe1[1]=0.75; rkcoe2[1]=0.25; rkcoe3[1]=0.25;
    rkcoe1[2]=1.0/3.0; rkcoe2[2]=2.0/3.0; rkcoe3[2]=2.0/3.0;
    rk=3;
}
