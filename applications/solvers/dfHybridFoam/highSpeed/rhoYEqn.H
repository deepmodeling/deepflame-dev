/*---------------------------------------------------------------------------*\
    highSpeed/rhoYEqn.H — Species transport for highSpeed region (Euler branch)
\*---------------------------------------------------------------------------*/

if (!inviscid)
{
    hDiffCorrFlux = Zero;
    diffAlphaD = Zero;
    sumYDiffError = Zero;

    forAll(Y, specI)
    {
        sumYDiffError += chemistry->rhoD(specI)*fvc::grad(Y[specI]);
    }
}

tmp<fv::convectionScheme<scalar>> mvConvection
(
    fv::convectionScheme<scalar>::New
    (
        mesh,
        fields,
        rhoPhi,
        mesh.divScheme("div(phi,Yi_h)")
    )
);

{
    combustion->correct();

    volScalarField Yt(0.0*Y[0]);

    forAll(Y, specI)
    {
        volScalarField& Yi = Y[specI];

        if (!inviscid)
        {
            hDiffCorrFlux += chemistry->hei(specI)
              * (chemistry->rhoD(specI)*fvc::grad(Yi) - Yi*sumYDiffError);
            diffAlphaD += fvc::laplacian(thermo.alpha()*chemistry->hei(specI), Yi);
        }

        {
            solve
            (
                fvm::ddt(rhoYi[specI])
              + fvc::div(rhoPhiYi[specI])
              ==
                chemistry->RR(specI)
            );

            Yi = rhoYi[specI]/rho;
            Yi.max(0.0);

            if (!inviscid)
            {
                const surfaceScalarField phiUc =
                    linearInterpolate(sumYDiffError) & mesh.Sf();
                tmp<volScalarField> DEff = chemistry->rhoD(specI)
                  + turbulence.mut()/Sct;

                fvScalarMatrix YiEqn
                (
                    fvm::ddt(rho, Yi) - fvc::ddt(rho, Yi)
                  - (
                        turbName == "laminar"
                        ? (fvm::laplacian(DEff(), Yi)
                         - mvConvection->fvmDiv(phiUc, Yi))
                        : fvm::laplacian(DEff(), Yi)
                    )
                );

                YiEqn.relax();
                YiEqn.solve("Yi");
                Yi.max(0.0);
            }

            rhoYi[specI] = rho*Yi;
            Yt += Yi;
        }
    }

    // Normalise mass fractions
    forAll(Y, specI)
    {
        Y[specI] = Y[specI]/Yt;
        Y[specI].max(0.0);
        rhoYi[specI] = rho*Y[specI];
    }
}
