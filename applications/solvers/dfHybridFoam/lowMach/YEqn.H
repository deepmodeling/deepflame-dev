/*---------------------------------------------------------------------------*\
    lowMach/YEqn.H — Species transport for lowMach region
\*---------------------------------------------------------------------------*/

// Reset diffusion correction fields
hDiffCorrFlux = Zero;
diffAlphaD = Zero;
sumYDiffError = Zero;

tmp<fv::convectionScheme<scalar>> mvConvection
(
    fv::convectionScheme<scalar>::New
    (
        mesh,
        fields,
        phi,
        mesh.divScheme("div(phi,Yi_h)")
    )
);

forAll(Y, specI)
{
    sumYDiffError += chemistry->rhoD(specI)*fvc::grad(Y[specI]);
}
const surfaceScalarField phiUc = linearInterpolate(sumYDiffError) & mesh.Sf();

{
    if (!splitting)
    {
        combustion->correct();
    }

    volScalarField Yt(0.0*Y[0]);
    forAll(Y, specI)
    {
        volScalarField& Yi = Y[specI];
        hDiffCorrFlux += chemistry->hei(specI)
          * (chemistry->rhoD(specI)*fvc::grad(Yi) - Yi*sumYDiffError);
        diffAlphaD += fvc::laplacian(thermo.alpha()*chemistry->hei(specI), Yi);

        if (specI != inertIndex)
        {
            tmp<volScalarField> DEff = chemistry->rhoD(specI)
              + turbulence.mut()/Sct;

            fvScalarMatrix YiEqn
            (
                fvm::ddt(rho, Yi)
              + (
                    turbName == "laminar"
                    ? (mvConvection->fvmDiv(phi, Yi)
                     + mvConvection->fvmDiv(phiUc, Yi))
                    : mvConvection->fvmDiv(phi, Yi)
                )
              ==
                (
                    splitting
                    ? fvm::laplacian(DEff(), Yi)
                    : (fvm::laplacian(DEff(), Yi) + combustion->R(Yi))
                )
            );

            YiEqn.solve("Yi");
            Yi.max(0.0);
            Yt += Yi;
        }
    }

    Y[inertIndex] = scalar(1) - Yt;
    Y[inertIndex].max(0.0);
}
