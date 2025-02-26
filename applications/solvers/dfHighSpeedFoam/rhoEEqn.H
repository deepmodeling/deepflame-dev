surfaceScalarField sigmaDotU
(
    "sigmaDotU",
    (
        fvc::interpolate(turbulence->muEff())*mesh.magSf()*fvc::snGrad(U)
      + (mesh.Sf() & fvc::interpolate(tauMC))
    )
    & (a_pos*U_pos + a_neg*U_neg)
);

if((ddtSchemes == "RK2SSP") || (ddtSchemes == "RK3SSP"))
{
    rhoE_rhs = -fvc::div(rhoEPhi)+fvc::div(sigmaDotU);

    rhoE =  rkcoe1[nrk]*rhoE_save
          + rkcoe2[nrk]*rhoE
          + rkcoe3[nrk]*rhoE_rhs*runTime.deltaT();
}
else
{
    tmp<fvScalarMatrix> thcSource(new fvScalarMatrix(rhoE, dimEnergy/dimTime));
    fvScalarMatrix& hcSource = thcSource.ref();
    forAll(Y, i)
    {
        hcSource.source() -= parcels.rhoTrans(i)*chemistry->mixture().Hc(i)/runTime.deltaT();
    }

    solve
    (
            fvm::ddt(rhoE)
          + fvc::div(rhoEPhi)
          - fvc::div(sigmaDotU)
        ==
            parcels.Sh(rhoE)
          + hcSource
    );
}

ea = rhoE/rho - 0.5*magSqr(U);
ea.correctBoundaryConditions();

// ha = ea + p/rho;
chemistry->correctThermo(); // before this, we must update ha = e + p/rho

rhoE.boundaryFieldRef() == rho.boundaryField()*(ea.boundaryField() + 0.5*magSqr(U.boundaryField()));

if (!inviscid)
{
    if((ddtSchemes == "RK2SSP") || (ddtSchemes == "RK3SSP"))
    {
        rhoE_rhs =   fvc::laplacian(turbulence->alphaEff()*thermo.gamma(), ea)
                  - diffAlphaD
                  + fvc::div(hDiffCorrFlux);

        rhoE +=  rkcoe3[nrk]*rhoE_rhs*runTime.deltaT();

        ea = rhoE/rho - 0.5*magSqr(U);
        ea.correctBoundaryConditions();

        // ha = ea + p/rho;
        chemistry->correctThermo(); // before this, we must update ha = e + p/rho
    }
    else
    {
        fvScalarMatrix eEqn
        (
              fvm::ddt(rho, ea) - fvc::ddt(rho, ea)
            ==
                (
                    turbName == "laminar"
                    ?
                    (
                        // alpha in deepflame is considered to calculate h by default (kappa/Cp), so multiply gamma to correct alpha
                        fvm::laplacian(turbulence->alphaEff()*thermo.gamma(), ea)
                    -   diffAlphaD
                    +   fvc::div(hDiffCorrFlux)
                    )
                    :
                    (
                        fvm::laplacian(turbulence->alphaEff()*thermo.gamma(), ea)
                    )
                )
        );

        eEqn.solve("ea");

        // ha = ea + p/rho;
        chemistry->correctThermo();
        rhoE = rho*(ea + 0.5*magSqr(U));
        rhoE.boundaryFieldRef() == rho.boundaryField()*(ea.boundaryField() + 0.5*magSqr(U.boundaryField()));
    }

}

Info<< "min/max(T) = "
        << min(T).value() << ", " << max(T).value() << endl;

p.ref() = rho()/psi();
p.correctBoundaryConditions();
rho.boundaryFieldRef() == psi.boundaryField()*p.boundaryField();
