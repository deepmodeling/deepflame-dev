void KNPFlux(
    const surfaceScalarField& meshPhi,
    const surfaceScalarField& magSf,
    const surfaceVectorField& normal,
    const surfaceScalarField& rhoOwn,
    const surfaceScalarField& rhoNei,
    const surfaceVectorField& UOwn,
    const surfaceVectorField& UNei,
    const surfaceScalarField& HOwn,
    const surfaceScalarField& HNei,
    const surfaceScalarField& pOwn,
    const surfaceScalarField& pNei,
    const surfaceScalarField& gammaOwn,
    const surfaceScalarField& gammaNei,
    const surfaceScalarField& aOwn,
    const surfaceScalarField& aNei,
    surfaceScalarField& massFlux,
    surfaceVectorField& momentumFlux,
    surfaceScalarField& energyFlux
)
{
    surfaceScalarField UvOwn((UOwn & normal) - meshPhi/magSf);

    surfaceScalarField UvNei((UNei & normal) - meshPhi/magSf);

    surfaceScalarField EOwn("EOwn", HOwn - pOwn/rhoOwn);
    surfaceScalarField ENei("ENei", HNei - pNei/rhoNei);

    surfaceScalarField phiv_pos("phiv_pos", UvOwn*magSf);  //
    surfaceScalarField phiv_neg("phiv_neg", UvNei*magSf);

    
    surfaceScalarField cSf_pos("cSf_pos",aOwn*magSf);
    surfaceScalarField cSf_neg("cSf_neg",aNei*magSf);
    dimensionedScalar v_zero("v_zero", dimVolume/dimTime, 0.0);
    surfaceScalarField ap("ap",max(max(phiv_pos + cSf_pos, phiv_neg + cSf_neg), v_zero));
    surfaceScalarField am("am",min(min(phiv_pos - cSf_pos, phiv_neg - cSf_neg), v_zero));

    surfaceScalarField a_pos("a_pos", ap/(ap - am));

    surfaceScalarField amaxSf("amaxSf", max(mag(am), mag(ap)));

    surfaceScalarField aSf("aSf", am*a_pos);

    surfaceScalarField a_neg("a_neg", 1.0 - a_pos);

    phiv_pos *= a_pos;
    phiv_neg *= a_neg;

    surfaceScalarField aphiv_pos("aphiv_pos", phiv_pos - aSf);
    surfaceScalarField aphiv_neg("aphiv_neg", phiv_neg + aSf);

    // Reuse amaxSf for the maximum positive and negative fluxes
    // estimated by the central scheme
    amaxSf = max(mag(aphiv_pos), mag(aphiv_neg));

    massFlux = aphiv_pos*rhoOwn + aphiv_neg*rhoNei;

    momentumFlux = (aphiv_pos*rhoOwn*UOwn + aphiv_neg*rhoNei*UNei)
        + (a_pos*pOwn + a_neg*pNei)*magSf*normal;

    energyFlux = aphiv_pos*(rhoOwn*EOwn + pOwn)
        + aphiv_neg*(rhoNei*ENei + pNei)
        + aSf*pOwn - aSf*pNei
        + meshPhi*(a_pos*pOwn + a_neg*pNei);

}
