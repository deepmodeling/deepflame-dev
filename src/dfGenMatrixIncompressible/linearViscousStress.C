// #include "linearViscousStress.H"
// #include "fvc.H"
// #include "fvm.H"
// #include "GenFvMatrix.H"

// #include "incompressibleTwoPhaseMixture.H"
// #include "immiscibleIncompressibleTwoPhaseMixture.H"
// #include "turbulentTransportModel.H"
// // namespace Foam
// // {

// // tmp<fvVectorMatrix>
// // turbulenceModelLinearViscousStressDivDevRhoReff
// // (
// //     volVectorField& U,
// //     incompressible::turbulenceModel& turbulence
// // )
// // {
// //     return
// //     (
// //       - fvc::div((turbulence.alpha()*turbulence.rho()*turbulence.nuEff())*dev2(T(fvc::grad(U))))
// //       - fvm::laplacian(turbulence.alpha()*turbulence.rho()*turbulence.nuEff(), U)
// //     );
// // }

// // }

// template<class BasicTurbulenceModel>
// Foam::linearViscousStress<BasicTurbulenceModel>::linearViscousStress
// (
//     const word& modelName,
//     const alphaField& alpha,
//     const rhoField& rho,
//     const volVectorField& U,
//     const surfaceScalarField& alphaRhoPhi,
//     const surfaceScalarField& phi,
//     const transportModel& transport,
//     const word& propertiesName
// )
// :
//     BasicTurbulenceModel
//     (
//         modelName,
//         alpha,
//         rho,
//         U,
//         alphaRhoPhi,
//         phi,
//         transport,
//         propertiesName
//     )
// {}

// // * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

// template<class BasicTurbulenceModel>
// Foam::tmp<Foam::fvVectorMatrix>
// Foam::linearViscousStress<BasicTurbulenceModel>::divDevRhoReff
// (
//     const volScalarField& rho,
//     incompressible::turbulenceModel& turbulence
// ) const
// {
//     return
//     (
//       - fvc::div((turbulence->alpha_*rho*turbulence->nuEff())*dev2(T(fvc::grad(U))))
//       - fvm::laplacian(turbulence->alpha_*rho*turbulence->nuEff(), U)
//     );
// }
