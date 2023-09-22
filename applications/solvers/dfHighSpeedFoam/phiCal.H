

surfaceVectorField  phiUp
(
    IOobject
    (
        "phiUp",
        mesh.time().timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedVector("0", sqr(dimVelocity)*dimDensity*dimArea, Zero)
);

surfaceScalarField  phiEp
(
    IOobject
    (
        "phiEp",
        mesh.time().timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("0", pow3(dimVelocity)*dimDensity*dimArea, 0)
);


surfaceScalarField  meshPhi
(
    IOobject
    (
        "meshPhi",
        mesh.time().timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("0", (dimVelocity)*dimArea, 0)
);
if(mesh.moving())
{
  meshPhi = meshPhi + mesh.phi();
}

//-- Computing flux
fluxMap[fluxScheme](
    meshPhi,
    mesh.magSf(),
    normal,
    rho_pos,rho_neg,
    U_pos,U_neg,
    H_pos,H_neg,
    p_pos,p_neg,
    gamma_pos,gamma_neg,
    a_pos,a_neg,
    phi,
    phiUp,
    phiEp
);


PtrList<surfaceScalarField> phiYi(nspecies);
forAll(phiYi,i)
{
  phiYi.set
  (
    i,
    new surfaceScalarField
    (
      IOobject
      (
        "phiYi_" + Y[i].name(),
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
      ),
      phi*fvc::interpolate(Y[i], "Yi").ref() 
    )
  );
}

volScalarField muEff("muEff", turbulence->muEff());
volTensorField tauMC("tauMC", muEff*dev2(Foam::T(fvc::grad(U))));
