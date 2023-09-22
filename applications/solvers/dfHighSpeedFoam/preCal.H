surfaceVectorField normal(mesh.Sf()/mesh.magSf());

// --- Directed interpolation of primitive fields onto faces
surfaceScalarField rho_pos(interpolate(rho, pos));
surfaceScalarField rho_neg(interpolate(rho, neg));

PtrList<surfaceScalarField> rhoYi_pos(nspecies);
PtrList<surfaceScalarField> rhoYi_neg(nspecies);
forAll(rhoYi_pos,i)
{
  rhoYi_pos.set
  (
    i,
    new surfaceScalarField
    (
      IOobject
      (
        "rhoYi_pos" + Y[i].name(),
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
      ),
      interpolate(rhoYi[i], pos,"Yi")
    )
  );
}

forAll(rhoYi_neg,i)
{
  rhoYi_neg.set
  (
    i,
    new surfaceScalarField
    (
      IOobject
      (
        "rhoYi_neg" + Y[i].name(),
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
      ),
      interpolate(rhoYi[i], neg,"Yi")
    )
  );
}

surfaceVectorField rhoU_pos(interpolate(rhoU, pos, U.name()));
surfaceVectorField rhoU_neg(interpolate(rhoU, neg, U.name()));

surfaceScalarField rhoE_pos(interpolate(rhoE, pos, T.name()));
surfaceScalarField rhoE_neg(interpolate(rhoE, neg, T.name()));

volScalarField rPsi("rPsi", 1.0/psi);
surfaceScalarField rPsi_pos(interpolate(rPsi, pos, T.name()));
surfaceScalarField rPsi_neg(interpolate(rPsi, neg, T.name()));

surfaceScalarField ea_pos(interpolate(ea, pos, T.name()));
surfaceScalarField ea_neg(interpolate(ea, neg, T.name()));

surfaceVectorField U_pos("U_pos", rhoU_pos/rho_pos);
surfaceVectorField U_neg("U_neg", rhoU_neg/rho_neg);

surfaceScalarField p_pos("p_pos", rho_pos*rPsi_pos);
surfaceScalarField p_neg("p_neg", rho_neg*rPsi_neg);

surfaceScalarField E_pos("E_pos", ea_pos + 0.5*magSqr(U_pos));//rhoE_pos/rho_pos);
surfaceScalarField E_neg("E_neg", ea_neg + 0.5*magSqr(U_neg));//rhoE_neg/rho_neg);

//volScalarField H("H",  p/(chemistry.Cp()/chemistry.Cv() - 1.0)/rho+ rPsi);
surfaceScalarField H_pos("H_pos", (rhoE_pos + p_pos)/rho_pos);
surfaceScalarField H_neg("H_neg", (rhoE_neg + p_neg)/rho_neg);

volScalarField gamma_("gamma_", thermo.Cp()/thermo.Cv());
surfaceScalarField gamma_pos
(
    "gamma_pos",
    interpolate(gamma_, pos, T.name())
);
surfaceScalarField gamma_neg
(
    "gamma_neg",
    interpolate(gamma_, neg, T.name())
);

volScalarField c("c", sqrt(thermo.Cp()/thermo.Cv()*rPsi));
surfaceScalarField a_pos
(
    "a_pos",
    interpolate(c, pos, T.name())
);
surfaceScalarField a_neg
(
    "a_neg",
    interpolate(c, neg, T.name())
);

// Maximum eigenvalue
surfaceScalarField amaxSf("amaxSf",
    0.5 * mag((U_pos + U_neg) & mesh.Sf()) - mesh.phi()
    + 0.5 * (a_pos + a_neg) * mesh.magSf()
);
