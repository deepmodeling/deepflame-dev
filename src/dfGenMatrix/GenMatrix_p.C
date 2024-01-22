#include "GenFvMatrix.H"
#include "multivariateGaussConvectionScheme.H"
#include "gaussConvectionScheme.H"
#include "snGradScheme.H"
#include "linear.H"
#include "orthogonalSnGrad.H"
#include "lduMesh.H"

#define TIME
#ifdef TIME 
#define TICK0(prefix)\
    double prefix##_tick_0 = MPI_Wtime();

#define TICK(prefix,start,end)\
    double prefix##_tick_##end = MPI_Wtime();\
    Info << #prefix << "_time_" << #end << " : " << prefix##_tick_##end - prefix##_tick_##start << endl;

#else
#define TICK0(prefix) ;
#define TICK(prefix,start,end) ;
#endif
namespace Foam{

template<class Type>
Foam::tmp<Foam::GeometricField<Type, Foam::fvPatchField, Foam::volMesh>>
UEqn_H
(
    fvMatrix<Type>& UEqn
)
{
    TICK0(UEqn_H);
    const GeometricField<Type, fvPatchField, volMesh>& psi_ = UEqn.psi();
    const Field<Type>& source_ = UEqn.source();
    FieldField<Field, Type>& internalCoeffs_ = UEqn.internalCoeffs();
    TICK(UEqn_H, 0, 1);
    tmp<GeometricField<Type, fvPatchField, volMesh>> tHphi
    (
        new GeometricField<Type, fvPatchField, volMesh>
        (
            IOobject
            (
                "H("+psi_.name()+')',
                psi_.instance(),
                psi_.mesh(),
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            psi_.mesh(),
            UEqn.dimensions()/dimVol,
            extrapolatedCalculatedFvPatchScalarField::typeName
        )
    );
    GeometricField<Type, fvPatchField, volMesh>& Hphi = tHphi.ref();
    TICK(UEqn_H, 1, 2);

    for (direction cmpt=0; cmpt<Type::nComponents; cmpt++)
    {
        scalarField psiCmpt(psi_.primitiveField().component(cmpt));

        scalarField boundaryDiagCmpt(psi_.size(), 0.0);
        // addBoundaryDiag(boundaryDiagCmpt, cmpt); // add internal coeffs
        forAll(internalCoeffs_, patchi)
        {
            labelList addr = UEqn.lduAddr().patchAddr(patchi);
            tmp<Field<scalar>> internalCoeffs_cmpt = internalCoeffs_[patchi].component(cmpt);
            tmp<Field<scalar>> internalCoeffs_ave = cmptAv(internalCoeffs_[patchi]);
            forAll(addr, facei)
            {
                boundaryDiagCmpt[addr[facei]] = -internalCoeffs_cmpt()[facei] + internalCoeffs_ave()[facei];
            }
        }

        // boundaryDiagCmpt.negate();
        // addCmptAvBoundaryDiag(boundaryDiagCmpt); // add ave internal coeffs

        Hphi.primitiveFieldRef().replace(cmpt, boundaryDiagCmpt*psiCmpt);
    }
    TICK(UEqn_H, 2, 3);

    // lduMatrix::H
    const lduAddressing& lduAddr = UEqn.mesh().lduAddr();
    tmp<Field<Type>> tHpsi
    (
        new Field<Type>(lduAddr.size(), Zero)
    );
    Field<Type> & Hpsi = tHpsi.ref();
    Type* __restrict__ HpsiPtr = Hpsi.begin();
    const Type* __restrict__ psiPtr = psi_.begin();
    const label* __restrict__ uPtr = lduAddr.upperAddr().begin();
    const label* __restrict__ lPtr = lduAddr.lowerAddr().begin();
    const scalar* __restrict__ lowerPtr = UEqn.lower().begin();
    const scalar* __restrict__ upperPtr = UEqn.upper().begin();
    TICK(UEqn_H, 3, 4);

    for (label face=0; face<nFaces; face++)
    {
        HpsiPtr[uPtr[face]] -= lowerPtr[face]*psiPtr[lPtr[face]];
        HpsiPtr[lPtr[face]] -= upperPtr[face]*psiPtr[uPtr[face]];
    }

    TICK(UEqn_H, 4, 5);

    Hphi.primitiveFieldRef() += (tHpsi + source_);

    TICK(UEqn_H, 5, 6);

    FieldField<Field, vector> boundaryCoeffs_ = UEqn.boundaryCoeffs();
    forAll(psi_.boundaryField(), patchi)
    {
        const fvPatchField<vector>& ptf = psi_.boundaryField()[patchi];
        const Field<vector>& pbc = boundaryCoeffs_[patchi];
        if (!ptf.coupled())
        {
            // addToInternalField(lduAddr().patchAddr(patchi), pbc, source);
            labelList addr = lduAddr.patchAddr(patchi);
            forAll(addr, facei)
            {
                Hphi.primitiveFieldRef()[addr[facei]] += pbc[facei];
            }
        }
        else
        {
            const tmp<Field<vector>> tpnf = ptf.patchNeighbourField();
            const Field<vector>& pnf = tpnf();

            const labelUList& addr = lduAddr.patchAddr(patchi);

            forAll(addr, facei)
            {
                Hphi.primitiveFieldRef()[addr[facei]] += cmptMultiply(pbc[facei], pnf[facei]);
            }
        }
    }
    TICK(UEqn_H, 6, 7);

    // addBoundarySource(Hphi.primitiveFieldRef());

    Hphi.primitiveFieldRef() /= psi_.mesh().V();
    TICK(UEqn_H, 7, 8);

    Hphi.correctBoundaryConditions();
    TICK(UEqn_H, 8, 9);

    typename Type::labelType validComponents
    (
        psi_.mesh().template validComponents<Type>()
    );

    for (direction cmpt=0; cmpt<Type::nComponents; cmpt++)
    {
        if (validComponents[cmpt] == -1)
        {
            Hphi.replace
            (
                cmpt,
                dimensionedScalar(Hphi.dimensions(), 0)
            );
        }
    }
    TICK(UEqn_H, 9, 10);

    return tHphi;
}

tmp<volScalarField>
rAUConstructor
(
    fvMatrix<vector>& UEqn
)
{
    TICK0(rAUConstructor);

    const scalarField& diag_ = UEqn.diag();
    const GeometricField<vector, fvPatchField, volMesh>& psi_ = UEqn.psi();
    const fvMesh& mesh = psi_.mesh();
    tmp<volScalarField> rAU
    (
        volScalarField::New
        (
            "rAu",
            mesh,
            dimensionSet(-1,3,1,0,0,0,0),
            extrapolatedCalculatedFvPatchScalarField::typeName
        )
    );
    TICK(rAUConstructor, 0, 1);
    // D()
    FieldField<Field, vector>& internalCoeffs_ = UEqn.internalCoeffs();
    tmp<scalarField> tdiag(new scalarField(diag_)); // D
    scalarField& diagD = tdiag.ref();
    TICK(rAUConstructor, 1, 2);

    // addCmptAvBoundaryDiag(tdiag.ref());
    const lduAddressing& lduAddr = UEqn.mesh().lduAddr();
    forAll(internalCoeffs_, patchi)
    {
        // addToInternalField
        // (
        //     lduAddr().patchAddr(patchi),
        //     cmptAv(internalCoeffs_[patchi]), // average
        //     diag
        // );
        labelList addr = lduAddr.patchAddr(patchi);
        tmp<Field<scalar>> internalCoeffs_ave = cmptAv(internalCoeffs_[patchi]);
        forAll(addr, facei)
        {
            diagD[addr[facei]] += internalCoeffs_ave()[facei];
        }
    }
    TICK(rAUConstructor, 2, 3);

    rAU.ref().primitiveFieldRef() = 1.0 / (tdiag / mesh.V());

    // #pragma omp parallel for
    // for(label i = 0; i < diagD.size(); ++i){
    //     rAU.ref().primitiveFieldRef()[i] = 1.0 / tdiag()[i] / mesh.V()[i];
    // }

    TICK(rAUConstructor, 3, 4);
    rAU.ref().correctBoundaryConditions();
    TICK(rAUConstructor, 4, 5);

    return rAU;
}

tmp<surfaceScalarField>
rhorAUfConstructor
(
    const volScalarField& rhorAU,
    const surfaceScalarField& linear_weights
)
{
    const fvMesh& mesh = rhorAU.mesh();

    tmp<surfaceInterpolationScheme<scalar>> tinterpScheme_(new linear<scalar>(mesh));
    tmp<surfaceScalarField> tgamma = tinterpScheme_().interpolate(rhorAU);


    // construct new field
    tmp<surfaceScalarField> trhorAUf
    (
        new GeometricField<scalar, fvsPatchField, surfaceMesh>
        (
            IOobject
            (
                "interpolate("+rhorAU.name()+')',
                rhorAU.instance(),
                rhorAU.db()
            ),
            mesh,
            dimensionSet(0,0,1,0,0,0,0)
        )
    );
    const scalar* const __restrict__ linearWeightsPtr = linear_weights.primitiveField().begin(); // get linear weight
    
    surfaceScalarField& rhorAUf = trhorAUf.ref();
    scalar* rhorAUfPtr = &rhorAUf[0];
    const scalar* const __restrict__ rhorAUPtr = rhorAU.primitiveField().begin();

    const labelUList& P = mesh.owner();
    const labelUList& N = mesh.neighbour();

    #pragma omp parallel for
    for (label fi=0; fi<P.size(); fi++)
    {
        rhorAUfPtr[fi] = (linearWeightsPtr[fi]*(rhorAUPtr[P[fi]] - rhorAUPtr[N[fi]]) + rhorAUPtr[N[fi]]);
    }

    forAll(linear_weights.boundaryField(), Ki)
    {
        const fvsPatchScalarField& pLambda = linear_weights.boundaryField()[Ki];
        fvsPatchScalarField& psf = rhorAUf.boundaryFieldRef()[Ki];

        if (rhorAU.boundaryField()[Ki].coupled())
        {
            psf =
                (
                    pLambda*rhorAU.boundaryField()[Ki].patchInternalField()
                + (1.0 - pLambda)*rhorAU.boundaryField()[Ki].patchNeighbourField()
                );
        }
        else
        {
            psf = rhorAU.boundaryField()[Ki];
        }
    }

    // return trhorAUf;
    return tgamma;
}

tmp<surfaceScalarField>
phiHbyAConstructor
(
    const volScalarField& rho,
    const volVectorField& HbyA,
    const surfaceScalarField& rhorAUf,
    const surfaceScalarField& tddtCorr,
    const surfaceScalarField& linear_weights
)
{
    const fvMesh& mesh = rho.mesh(); 
    // tmp<surfaceScalarField> tphiHbyA
    // (
    //     surfaceScalarField::New
    //     (
    //         "phiHbyA",
    //         mesh,
    //         dimensionSet(1,0,-1,0,0,0,0)
    //     )
    // );

    // fvc::interpolate(rho)
    const scalar* const __restrict__ linearWeightsPtr = linear_weights.primitiveField().begin(); // get linear weight
    tmp<surfaceScalarField> trhof
    (
        new GeometricField<scalar, fvsPatchField, surfaceMesh>
        (
            IOobject
            (
                "interpolate("+rho.name()+')',
                rho.instance(),
                rho.db()
            ),
            mesh,
            rho.dimensions()
        )
    );
    surfaceScalarField& rhof = trhof.ref();
    scalar* rhofPtr = &rhof[0];
    const scalar* const __restrict__ rhoPtr = rho.primitiveField().begin();

    const labelUList& P = mesh.owner();
    const labelUList& N = mesh.neighbour();

    for (label fi=0; fi<P.size(); fi++)
    {
        rhofPtr[fi] = (linearWeightsPtr[fi]*(rhoPtr[P[fi]] - rhoPtr[N[fi]]) + rhoPtr[N[fi]]);
    }

    forAll(linear_weights.boundaryField(), Ki)
    {
        const fvsPatchScalarField& pLambda = linear_weights.boundaryField()[Ki];
        fvsPatchScalarField& psf = rhof.boundaryFieldRef()[Ki];

        if (rho.boundaryField()[Ki].coupled())
        {
            psf =
                (
                    pLambda*rho.boundaryField()[Ki].patchInternalField()
                + (1.0 - pLambda)*rho.boundaryField()[Ki].patchNeighbourField()
                );
        }
        else
        {
            psf = rho.boundaryField()[Ki];
        }
    }

    // tmp<surfaceScalarField> ttest = fvc::interpolate(rho);
    // surfaceScalarField test = ttest.ref();
    // check_field_equal(test, rhof);


    // fvc::flux(HbyA)
    tmp<surfaceScalarField> tHbyAf
    (
        new GeometricField<scalar, fvsPatchField, surfaceMesh>
        (
            IOobject
            (
                "interpolate("+HbyA.name()+')',
                HbyA.instance(),
                HbyA.db()
            ),
            mesh,
            mesh.Sf().dimensions()*HbyA.dimensions()
        )
    );
    surfaceScalarField& HbyAf = tHbyAf.ref();
    scalar* HbyAfPtr = &HbyAf[0];
    const vector* const __restrict__ HbyAPtr = HbyA.primitiveField().begin();
    const surfaceVectorField& Sf = mesh.Sf();

    for (label fi=0; fi<P.size(); fi++)
    {
        HbyAfPtr[fi] = Sf[fi] & (linearWeightsPtr[fi]*(HbyAPtr[P[fi]] - HbyAPtr[N[fi]]) + HbyAPtr[N[fi]]);
    }

    forAll(linear_weights.boundaryField(), Ki)
    {
        const fvsPatchScalarField& pLambda = linear_weights.boundaryField()[Ki];
        const fvsPatchVectorField& pSf = Sf.boundaryField()[Ki];
        fvsPatchScalarField& psf = HbyAf.boundaryFieldRef()[Ki];

        if (HbyA.boundaryField()[Ki].coupled())
        {
            psf =
                pSf
            &   (
                    pLambda*HbyA.boundaryField()[Ki].patchInternalField()
                + (1.0 - pLambda)*HbyA.boundaryField()[Ki].patchNeighbourField()
                );
        }
        else
        {
            psf = pSf & HbyA.boundaryField()[Ki];
        }
    }

    // tmp<surfaceScalarField> tphiHbyA = trhof * tHbyAf + rhorAUf * tddtCorr;

    // return trhof;
    return tHbyAf;
}

tmp<fvScalarMatrix>
GenMatrix_p(
    const volScalarField& rho,
    volScalarField& p,
    const surfaceScalarField& phiHbyA,
    const surfaceScalarField& rhorAUf,
    const volScalarField& psi
)
{
    TICK0(GenMatrix_p);
    const fvMesh& mesh = p.mesh();
    assert(mesh.moving() == false);

    label nCells = mesh.nCells();

    // basic matrix
    tmp<fvScalarMatrix> tfvm
    (
        new fvScalarMatrix
        (
            p,
            psi.dimensions()*p.dimensions()*dimVol/dimTime
        )
    );
    fvScalarMatrix& fvm = tfvm.ref();

    scalar* __restrict__ diagPtr = fvm.diag().begin();
    scalar* __restrict__ sourcePtr = fvm.source().begin();
    scalar* __restrict__ upperPtr = fvm.upper().begin();

    const labelUList& l = fvm.lduAddr().lowerAddr();
    const labelUList& u = fvm.lduAddr().upperAddr();

    scalar rDeltaT = 1.0/mesh.time().deltaTValue();

    // fvmddt
    auto fvmDdtTmp = psi * correction(EulerDdtSchemeFvmDdt(p));

    TICK(GenMatrix_p, 0, 1);

    // fvcddt
    // auto fvcDdtTmp = EulerDdtSchemeFvcDdt(rho);
    const scalar* const __restrict__ rhoPtr = rho.primitiveField().begin();
    const scalar* const __restrict__ rhoOldTimePtr = rho.oldTime().primitiveField().begin();
    const scalar* const __restrict__ meshVPtr = mesh.V().begin();

    // fvcdiv
    // auto fvcDivTmp = gaussConvectionSchemeFvcDiv(phiHbyA);
    const scalar* const __restrict__ phiHbyAPtr = phiHbyA.primitiveField().begin();

    // fvmLaplacian
    // auto fvmLaplacianTmp = gaussLaplacianSchemeFvmLaplacian(rhorAUf, p);
    tmp<fv::snGradScheme<scalar>> tsnGradScheme_(new fv::orthogonalSnGrad<scalar>(mesh));
    const surfaceScalarField& deltaCoeffs = tsnGradScheme_().deltaCoeffs(p)();

    surfaceScalarField gammaMagSf
    (
        rhorAUf * mesh.magSf()
    );

    const scalar* const __restrict__ deltaCoeffsPtr = deltaCoeffs.primitiveField().begin();
    const scalar* const __restrict__ gammaMagSfPtr = gammaMagSf.primitiveField().begin();
    // Info << "fvmLaplacian = " << time_end - time_begin << endl;
    TICK(GenMatrix_p, 1, 2);

    // merge
    double *fvcDivPtr = new double[nCells]{0.};
    
    for(label f = 0; f < nFaces; ++f){
        scalar var1 = deltaCoeffsPtr[f] * gammaMagSfPtr[f];
        // lowerPtr[f] = var1;
        upperPtr[f] = var1;
        diagPtr[l[f]] -= var1;
        diagPtr[u[f]] -= var1;
        fvcDivPtr[l[f]] += phiHbyAPtr[f];
        fvcDivPtr[u[f]] -= phiHbyAPtr[f];
    }

    TICK(GenMatrix_p, 2, 3);
    // - boundary loop
    forAll(p.boundaryField(), patchi)
    {
        const fvPatchField<scalar>& psf = p.boundaryField()[patchi];
        const fvsPatchScalarField& pGamma = gammaMagSf.boundaryField()[patchi];
        const fvsPatchScalarField& pDeltaCoeffs =
            deltaCoeffs.boundaryField()[patchi];

        const labelUList& pFaceCells =
            mesh.boundary()[patchi].faceCells();

        const fvsPatchField<scalar>& pssf = phiHbyA.boundaryField()[patchi];

        if (psf.coupled())
        {
            fvm.internalCoeffs()[patchi] =
                pGamma*psf.gradientInternalCoeffs(pDeltaCoeffs);
            fvm.boundaryCoeffs()[patchi] =
                -pGamma*psf.gradientBoundaryCoeffs(pDeltaCoeffs);
        }
        else
        {
            fvm.internalCoeffs()[patchi] = pGamma*psf.gradientInternalCoeffs();
            fvm.boundaryCoeffs()[patchi] = -pGamma*psf.gradientBoundaryCoeffs();
        }
        forAll(mesh.boundary()[patchi], facei)
        {
            fvcDivPtr[pFaceCells[facei]] += pssf[facei];
        }
    }

    TICK(GenMatrix_p, 3, 4);
    // - cell loop

    #pragma omp parallel for
    for(label c = 0; c < nCells; ++c){
        sourcePtr[c] += rDeltaT * (rhoPtr[c] - rhoOldTimePtr[c]) * meshVPtr[c];
        sourcePtr[c] += fvcDivPtr[c];
    }

    tmp<fvScalarMatrix> fvm_final
    (
        fvmDdtTmp 
        // + fvcDivTmp
        - fvm
    );
    TICK(GenMatrix_p, 4, 5);

    return fvm_final;
}

template
Foam::tmp<Foam::GeometricField<vector, Foam::fvPatchField, Foam::volMesh>>
UEqn_H
(
    fvMatrix<vector>& UEqn
);

}

