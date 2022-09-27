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

forAll(Y, i)
{
    sumYDiffError += chemistry->rhoD(i)*fvc::grad(Y[i]);
}
const surfaceScalarField phiUc = linearInterpolate(sumYDiffError) & mesh.Sf();

{
    combustion->correct();
    volScalarField Yt(0.0*Y[0]);

    forAll(Y, i)
    {
        volScalarField& Yi = Y[i];
        hDiffCorrFlux += chemistry->hai(i)*(chemistry->rhoD(i)*fvc::grad(Yi) - Yi*sumYDiffError);
        diffAlphaD += fvc::laplacian(thermo.alpha()*chemistry->hai(i), Yi);

        if (i != inertIndex)
        {
            tmp<volScalarField> DEff = chemistry->rhoD(i) + turbulence->mut()/Sct;
            fvScalarMatrix YEqn
            (
                fvm::ddt(rho, Yi)
              + mvConvection->fvmDiv(phi, Yi)
              - fvm::laplacian(DEff(), Yi)
              + mvConvection->fvmDiv(phiUc, Yi)
             ==
                parcels.SYi(i, Yi)
              + combustion->R(Yi)
              //+ fvOptions(rho, Yi)
            );

            YEqn.relax();

            //fvOptions.constrain(YEqn);

            YEqn.solve("Yi");

            //fvOptions.correct(Yi);

            Yi.max(0.0);
            Yt += Yi;
        }
    }

    Y[inertIndex] = scalar(1) - Yt;
    Y[inertIndex].max(0.0);
}
