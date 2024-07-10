rho_save = rho;

rhoU_save = rho*U;

rhoE_save = rho*(ea + 0.5*magSqr(U));

forAll(rhoYi_save,i)
{
    rhoYi_save[i] = rho*Y[i];
}

Info <<"finish update saved fields"<< endl;