        surfaceVectorField normal(mesh.Sf()/mesh.magSf());

        //-- reconstruct variable to cell interface

        surfaceScalarField rho_pos(interpolate(rho, pos));
        surfaceScalarField rho_neg(interpolate(rho, neg));

        surfaceVectorField rhoU_pos(interpolate(rhoU, pos, U.name()));
        surfaceVectorField rhoU_neg(interpolate(rhoU, neg, U.name()));

        surfaceScalarField rhoE_pos(interpolate(rhoE, pos, T.name()));
        surfaceScalarField rhoE_neg(interpolate(rhoE, neg, T.name()));

        surfaceVectorField U_pos(interpolate(U, pos, U.name()));
        surfaceVectorField U_neg(interpolate(U, neg, U.name()));

        volScalarField rPsi("rPsi", 1.0/psi);
        surfaceScalarField rPsi_pos(interpolate(rPsi, pos, T.name()));
        surfaceScalarField rPsi_neg(interpolate(rPsi, neg, T.name()));
        surfaceScalarField p_pos("p_pos", rho_pos*rPsi_pos);
        surfaceScalarField p_neg("p_neg", rho_neg*rPsi_neg);

        surfaceScalarField e_pos(interpolate(e, pos, T.name()));
        surfaceScalarField e_neg(interpolate(e, neg, T.name()));

        surfaceScalarField E_pos("E_pos", e_pos + 0.5*magSqr(U_pos));//rhoE_pos/rho_pos);
        surfaceScalarField E_neg("E_neg", e_neg + 0.5*magSqr(U_neg));//rhoE_neg/rho_neg);

        // surfaceVectorField U_pos(interpolate(U, pos, U.name()));
        // surfaceVectorField U_neg(interpolate(U, neg, U.name()));

        //volScalarField H("H",  p/(chemistry.Cp()/chemistry.Cv() - 1.0)/rho+ rPsi);
        surfaceScalarField H_pos("H_pos", (rhoE_pos + p_pos)/rho_pos);
        surfaceScalarField H_neg("H_neg", (rhoE_neg + p_neg)/rho_neg);
       
        volScalarField gamma_("gamma_", chemistry.Cp()/chemistry.Cv());
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

        volScalarField c("c", sqrt(chemistry.Cp()/chemistry.Cv()*rPsi));
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
            0.5 * mag((U_pos + U_neg) & mesh.Sf())
            + 0.5 * (a_pos + a_neg) * mesh.magSf()
        );
    
        // Compute cfl number and time step
        #include "centralCourantNo.H"

        //-- Computing flux
        fluxMap[fluxScheme](
            mesh.magSf(),
            normal,
            rho_pos,rho_neg,
            U_pos,U_neg,
            H_pos,H_neg,
            p_pos,p_neg,
            gamma_pos,gamma_neg,
            a_pos,a_neg,
            massFlux,
            momentumFlux,
            energyFlux
        );

        phi = massFlux;
        