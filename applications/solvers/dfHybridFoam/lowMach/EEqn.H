/*---------------------------------------------------------------------------*\
    lowMach/EEqn.H — Enthalpy equation for lowMach region

    Mirrors dfLowMachFoam/EEqn.H for the multi-region context.
\*---------------------------------------------------------------------------*/
{
    volScalarField& he = thermo.he();

    fvScalarMatrix EEqn
    (
        fvm::ddt(rho, he) + mvConvection->fvmDiv(phi, he)
      + fvc::ddt(rho, K) + fvc::div(phi, K)
      - dpdt
      ==
        (
            turbName == "laminar"
            ? (
                  fvm::laplacian(turbulence.alpha(), he)
                - diffAlphaD
                + fvc::div(hDiffCorrFlux)
              )
            : fvm::laplacian(turbulence.alphaEff(), he)
        )
    );

    if (CanteraMixture::getEnergyName() == word("hs"))
    {
        EEqn -= combustion->Qdot();
    }
    else if (CanteraMixture::getEnergyName() == word("ha"))
    {
        // For absolute enthalpy, Qdot is not used.
        // Spray hcSource would go here when parcels are enabled.
    }
    else
    {
        FatalErrorInFunction
            << "Wrong CanteraMixture energy name: "
            << CanteraMixture::getEnergyName()
            << abort(FatalError);
    }

    EEqn.solve("h");
}
