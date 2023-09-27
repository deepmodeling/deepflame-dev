phi = aphiv_pos*rho_pos + aphiv_neg*rho_neg;

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
      aphiv_pos*rhoYi_pos[i] + aphiv_neg*rhoYi_neg[i]
    )
  );
}

surfaceVectorField phiUp
(
  (aphiv_pos*rhoU_pos + aphiv_neg*rhoU_neg)
    + (a_pos*p_pos + a_neg*p_neg)*mesh.Sf()
);

surfaceScalarField phiEp
(
    "phiEp",
    aphiv_pos*(rho_pos*(ea_pos + 0.5*magSqr(U_pos)) + p_pos)
  + aphiv_neg*(rho_neg*(ea_neg + 0.5*magSqr(U_neg)) + p_neg)
  + aSf*p_pos - aSf*p_neg
);

// Make flux for pressure-work absolute
if (mesh.moving())
{
  phiEp += mesh.phi()*(a_pos*p_pos + a_neg*p_neg);
}

volScalarField muEff("muEff", turbulence->muEff());
volTensorField tauMC("tauMC", muEff*dev2(Foam::T(fvc::grad(U))));
