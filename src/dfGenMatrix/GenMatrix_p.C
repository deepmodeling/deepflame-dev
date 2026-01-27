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
    const GeometricField<Type, fvPatchField, volMesh>& psi_ = UEqn.psi();
    const Field<Type>& source_ = UEqn.source();
    FieldField<Field, Type>& internalCoeffs_ = UEqn.internalCoeffs();
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

    for (label face=0; face<nFaces; face++)
    {
        HpsiPtr[uPtr[face]] -= lowerPtr[face]*psiPtr[lPtr[face]];
        HpsiPtr[lPtr[face]] -= upperPtr[face]*psiPtr[uPtr[face]];
    }

    Hphi.primitiveFieldRef() += (tHpsi + source_);

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

    // addBoundarySource(Hphi.primitiveFieldRef());

    Hphi.primitiveFieldRef() /= psi_.mesh().V();

    Hphi.correctBoundaryConditions();

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

    return tHphi;
}

tmp<volScalarField>
rAUConstructor
(
    fvMatrix<vector>& UEqn
)
{
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
    // D()
    FieldField<Field, vector>& internalCoeffs_ = UEqn.internalCoeffs();
    tmp<scalarField> tdiag(new scalarField(diag_)); // D
    scalarField& diagD = tdiag.ref();

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

    rAU.ref().primitiveFieldRef() = 1.0 / (tdiag / mesh.V());

// #ifdef _OPENMP
    // #pragma omp parallel for
// #endif
    // for(label i = 0; i < diagD.size(); ++i){
    //     rAU.ref().primitiveFieldRef()[i] = 1.0 / tdiag()[i] / mesh.V()[i];
    // }

    rAU.ref().correctBoundaryConditions();

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

#ifdef _OPENMP
    #pragma omp parallel for
#endif
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
    const surfaceScalarField& phi,
    const volScalarField& rAU,
    const volVectorField& U,
    const volVectorField& HbyA,
    const volScalarField& thermoPsi,
    // debug
    surfaceScalarField& rhorAUf,
    surfaceScalarField& phiHbyA
)
{
    clockTime clock;
    const fvMesh& mesh = p.mesh(); 
    assert(mesh.moving() == false);

    const MeshSchedule& meshSchedule = getSchedule();

    const label nCells = meshSchedule.nCells();
    const label nFaces = meshSchedule.nFaces();
    const label nPatches = meshSchedule.nPatches();
    const List<MeshSchedule::PatchType> patchTypes = meshSchedule.patchTypes();
    const labelList& patchSizes = meshSchedule.patchSizes();

    // basic variables
    const surfaceScalarField& weights = mesh.surfaceInterpolation::weights(); // interpolation weight (linear)
    const scalar *weightsPtr = &weights[0];
    const scalar* const __restrict__ meshSfPtr = &mesh.Sf()[0][0];
    const label *own = &mesh.owner()[0];
    const label *nei = &mesh.neighbour()[0];

    // basic matrix
    tmp<fvScalarMatrix> tfvm
    (
        new fvScalarMatrix
        (
            p,
            thermoPsi.dimensions()*p.dimensions()*dimVol/dimTime
        )
    );
    fvScalarMatrix& fvm = tfvm.ref();
    scalar rDeltaT = 1.0/mesh.time().deltaTValue();

    scalar* __restrict__ diagPtr = fvm.diag().begin();
    scalar* __restrict__ sourcePtr = fvm.source().begin();
    scalar* __restrict__ upperPtr = fvm.upper().begin();

    // allocate memory
    typedef scalar* scalarPtr;
    typedef const scalar* constScalarPtr;
    typedef const label* constLabelPtr;
    Info << "Gen_p init : " << clock.timeIncrement() << endl;

    const scalar *rAUPtr = rAU.begin();
    const scalar *rhoPtr = rho.begin();
    const scalar *meshVPtr = mesh.V().begin();
    const scalar *UOldPtr = (scalar*)U.oldTime().begin();
    const scalar *rhoOldPtr = rho.oldTime().begin();
    const scalar *phiPtr = phi.begin();
    const scalar *phiOldPtr = phi.oldTime().begin();
    const scalar *HbyAPtr = (scalar*)HbyA.begin();
    const scalar *pOldPtr = p.oldTime().begin();
    const scalar *thermoPsiPtr = thermoPsi.begin();
    const scalar *magSfPtr = mesh.magSf().begin();
    const scalar *deltaCoeffsPtr = &mesh.deltaCoeffs()[0];
    scalar *pPtr = p.begin();

    scalar *rhorAUfPtr = new scalar[nFaces];
    scalar *phiHbyAPtr = new scalar[nFaces];

    constLabelPtr *boundary_face_cell = new constLabelPtr[nPatches];
    constScalarPtr *boundary_mag_sf = new constScalarPtr[nPatches];
    constScalarPtr* boundary_weights = new constScalarPtr[nPatches];
    constScalarPtr *boundary_delta_coeffs = new constScalarPtr[nPatches];
    constScalarPtr *boundary_sf = new constScalarPtr[nPatches];

    constScalarPtr *boundary_phi = new constScalarPtr[nPatches];
    constScalarPtr *boundary_phiOld = new constScalarPtr[nPatches];
    scalarPtr *boundary_phiHbyAPtr = new scalarPtr[nPatches];
    scalarPtr *boundary_rhorAUf = new scalarPtr[nProcessBoundarySurfaces];

    constScalarPtr *boundary_rAU = new constScalarPtr[nPatches];
    scalarPtr *boundary_rAU_internal = new scalarPtr[nPatches];

    constScalarPtr *boundary_HbyA = new constScalarPtr[nPatches];
    scalarPtr *boundary_HbyA_internal = new scalarPtr[nPatches];

    constScalarPtr *boundary_rho = new constScalarPtr[nPatches];
    scalarPtr *boundary_rho_internal = new scalarPtr[nPatches];

    constScalarPtr *boundary_rhoOld = new constScalarPtr[nPatches];
    scalarPtr *boundary_rhoOld_internal = new scalarPtr[nPatches];

    constScalarPtr *boundary_Uold = new constScalarPtr[nPatches];
    scalarPtr *boundary_Uold_internal = new scalarPtr[nPatches];

    scalarPtr* internal_coeffs = new scalarPtr[nPatches];
    scalarPtr* boundary_coeffs = new scalarPtr[nPatches];
    scalarPtr* value_internal_coeffs = new scalarPtr[nPatches];
    scalarPtr* value_boundary_coeffs = new scalarPtr[nPatches];
    scalarPtr* gradient_internal_coeffs = new scalarPtr[nPatches];
    scalarPtr* gradient_boundary_coeffs = new scalarPtr[nPatches];

    for (label patchi = 0; patchi < nPatches; ++patchi) {
        label patchSize = patchSizes[patchi];
        boundary_face_cell[patchi] = mesh.boundary()[patchi].faceCells().begin(); // patchSize
        boundary_mag_sf[patchi] = mesh.magSf().boundaryField()[patchi].begin(); // patchSize
        boundary_weights[patchi] = weights.boundaryField()[patchi].begin(); // patchSize
        boundary_delta_coeffs[patchi] = mesh.deltaCoeffs().boundaryField()[patchi].begin(); // patchSize
        boundary_sf[patchi] = (scalar*)mesh.Sf().boundaryField()[patchi].begin(); // 3 * patchSize

        boundary_phi[patchi] = phi.boundaryField()[patchi].begin(); // patchSize
        boundary_phiOld[patchi] = phi.oldTime().boundaryField()[patchi].begin(); // patchSize
        boundary_phiHbyAPtr[patchi] = new scalar[patchSize]; // patchSize
        boundary_rhorAUf[patchi] = new scalar[patchSize]; // patchSize

        const fvPatchScalarField& pRAU = rAU.boundaryField()[patchi];
        boundary_rAU[patchi] = pRAU.begin(); // patchSize

        const fvPatchVectorField& pHbyA = HbyA.boundaryField()[patchi];
        boundary_HbyA[patchi] = (scalar*)pHbyA.begin(); // 3 * patchSize

        const fvPatchScalarField& pRho = rho.boundaryField()[patchi];
        boundary_rho[patchi] = pRho.begin(); // patchSize

        const fvPatchScalarField& pRhoOld = rho.oldTime().boundaryField()[patchi];
        boundary_rhoOld[patchi] = pRhoOld.begin(); // patchSize

        const fvPatchVectorField& pUOld = U.oldTime().boundaryField()[patchi];
        boundary_Uold[patchi] = (scalar*)pUOld.begin(); // 3 * patchSize

        internal_coeffs[patchi] = new scalar[patchSize];
        boundary_coeffs[patchi] = new scalar[patchSize];
        value_internal_coeffs[patchi] = new scalar[patchSize];
        value_boundary_coeffs[patchi] = new scalar[patchSize];
        gradient_internal_coeffs[patchi] = new scalar[patchSize];
        gradient_boundary_coeffs[patchi] = new scalar[patchSize];

        if(patchTypes[patchi] == MeshSchedule::PatchType::processor){
            scalarField patchRAUInternal = 
                    dynamic_cast<const processorFvPatchField<scalar>&>(pRAU).patchInternalField()();
            boundary_rAU_internal[patchi] = new scalar[patchSize]; // patchSize
            memcpy(boundary_rAU_internal[patchi], &patchRAUInternal[0], patchSize * sizeof(scalar));

            vectorField patchHbyAInternal =
                    dynamic_cast<const processorFvPatchField<vector>&>(pHbyA).patchInternalField()();
            boundary_HbyA_internal[patchi] = new scalar[3*patchSize]; // 3 * patchSize
            memcpy(boundary_HbyA_internal[patchi], &patchHbyAInternal[0][0], 3*patchSize * sizeof(scalar));

            scalarField patchRhoInternal = 
                    dynamic_cast<const processorFvPatchField<scalar>&>(pRho).patchInternalField()();
            boundary_rho_internal[patchi] = new scalar[patchSize]; // patchSize
            memcpy(boundary_rho_internal[patchi], &patchRhoInternal[0], patchSize * sizeof(scalar));

            scalarField patchRhoOldInternal = 
                    dynamic_cast<const processorFvPatchField<scalar>&>(pRhoOld).patchInternalField()();
            boundary_rhoOld_internal[patchi] = new scalar[patchSize]; // patchSize
            memcpy(boundary_rhoOld_internal[patchi], &patchRhoOldInternal[0], patchSize * sizeof(scalar));

            vectorField patchUOldInternal =
                    dynamic_cast<const processorFvPatchField<vector>&>(pUOld).patchInternalField()();
            boundary_Uold_internal[patchi] = new scalar[3*patchSize]; // 3 * patchSize
            memcpy(boundary_Uold_internal[patchi], &patchUOldInternal[0][0], 3*patchSize * sizeof(scalar));
        } else if (patchTypes[patchi] == MeshSchedule::PatchType::wall) {
            boundary_rAU_internal[patchi] = nullptr;
            boundary_HbyA_internal[patchi] = nullptr;
            boundary_rho_internal[patchi] = nullptr;
            boundary_rhoOld_internal[patchi] = nullptr;
            boundary_Uold_internal[patchi] = nullptr;
        } else {
            Info << "boundary condition not supported" << endl;
            std::exit(-1);
        }
    }

    Info << "Gen_p allocate : " << clock.timeIncrement() << endl;

    // update boundary coeffs
    for (label patchi = 0; patchi < nPatches; ++patchi) {
        if (patchTypes[patchi] == MeshSchedule::PatchType::wall) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (label f = 0; f < patchSizes[patchi]; ++f){
                value_internal_coeffs[patchi][f] = 1.;
                value_boundary_coeffs[patchi][f] = 0.;
                gradient_internal_coeffs[patchi][f] = 0.;
                gradient_boundary_coeffs[patchi][f] = 0.;
            } 
        } else if (patchTypes[patchi] == MeshSchedule::PatchType::processor) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (label f = 0; f < patchSizes[patchi]; ++f){
                scalar bouDeltaCoeffs = boundary_delta_coeffs[patchi][f];
                scalar bouWeight = boundary_weights[patchi][f];
                value_internal_coeffs[patchi][f] = bouWeight;
                value_boundary_coeffs[patchi][f] = 1 - bouWeight;
                gradient_internal_coeffs[patchi][f] = -1 * bouDeltaCoeffs;
                gradient_boundary_coeffs[patchi][f] = bouDeltaCoeffs;
            }
        }
    }

    Info << "Gen_p update boundary coeffs : " << clock.timeIncrement() << endl;

    // get rhorAUf
    // - internal
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (label i = 0; i < nFaces; ++i) {
        scalar w = weightsPtr[i];
        label owner = own[i];
        label neighbor = nei[i];
        scalar rhorAU_owner = rhoPtr[owner] * rAUPtr[owner];
        scalar rhorAU_neighbor = rhoPtr[neighbor] * rAUPtr[neighbor];
        rhorAUfPtr[i] = w * (rhorAU_owner - rhorAU_neighbor) + rhorAU_neighbor;
        rhorAUf[i] = rhorAUfPtr[i];
    }

    // - boundary
#ifdef _OPENMP
#pragma omp parallel
#endif
    for (label patchi = 0; patchi < nPatches; ++patchi) {
        fvsPatchScalarField& patchrhorAUf = const_cast<fvsPatchScalarField&>(rhorAUf.boundaryField()[patchi]);
        if (patchTypes[patchi] == MeshSchedule::PatchType::wall) {
#ifdef _OPENMP
#pragma omp for
#endif
            for (label f = 0; f < patchSizes[patchi]; ++f) {
                boundary_rhorAUf[patchi][f] = boundary_rho[patchi][f] * boundary_rAU[patchi][f];
                patchrhorAUf[f] = boundary_rhorAUf[patchi][f];
            }
        } else if (patchTypes[patchi] == MeshSchedule::PatchType::processor) {
#ifdef _OPENMP
#pragma omp for
#endif
            for (label f = 0; f < patchSizes[patchi]; ++f) {
                scalar bouWeight = boundary_weights[patchi][f];
                scalar neighbor_boundary_vf3 = boundary_rho[patchi][f] * boundary_rAU[patchi][f];
                scalar internal_boundary_vf3 = boundary_rho_internal[patchi][f] * boundary_rAU_internal[patchi][f];
                boundary_rhorAUf[patchi][f] = bouWeight * internal_boundary_vf3 + (1 - bouWeight) * neighbor_boundary_vf3;
                patchrhorAUf[f] = boundary_rhorAUf[patchi][f];
            }
        }
    }

    Info << "Gen_p rhorAUf : " << clock.timeIncrement() << endl;

    // get phiHbyA
    // - get_phiCorr_internal
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (label i = 0; i < nFaces; ++i) {
        scalar weights = weightsPtr[i];
        label owner = own[i];
        label neighbor = nei[i];
        scalar Sfx = meshSfPtr[i * 3];
        scalar Sfy = meshSfPtr[i * 3 + 1];
        scalar Sfz = meshSfPtr[i * 3 + 2];
        scalar vf_own_x = UOldPtr[owner * 3 + 0] * rhoOldPtr[owner];
        scalar vf_own_y = UOldPtr[owner * 3 + 1] * rhoOldPtr[owner];
        scalar vf_own_z = UOldPtr[owner * 3 + 2] * rhoOldPtr[owner];
        scalar vf_nei_x = UOldPtr[neighbor * 3 + 0] * rhoOldPtr[neighbor];
        scalar vf_nei_y = UOldPtr[neighbor * 3 + 1] * rhoOldPtr[neighbor];
        scalar vf_nei_z = UOldPtr[neighbor * 3 + 2] * rhoOldPtr[neighbor];
        scalar ssfx_vf = (weights * (vf_own_x - vf_nei_x) + vf_nei_x);
        scalar ssfy_vf = (weights * (vf_own_y - vf_nei_y) + vf_nei_y);
        scalar ssfz_vf = (weights * (vf_own_z - vf_nei_z) + vf_nei_z);
        scalar phiCorrVal = phiOldPtr[i] - (Sfx * ssfx_vf + Sfy * ssfy_vf + Sfz * ssfz_vf);
        scalar phiVal = phiOldPtr[i];
        scalar tddtCouplingCoeff = 1. - min(fabs(phiCorrVal)/fabs(phiVal) + Foam::SMALL, 1.);
        scalar phiHbyA_ = tddtCouplingCoeff * rDeltaT * phiCorrVal * rhorAUfPtr[i];
        scalar ssfx_HbyA = (weights * (HbyAPtr[owner * 3 + 0] - HbyAPtr[neighbor * 3 + 0]) + HbyAPtr[neighbor * 3 + 0]);
        scalar ssfy_HbyA = (weights * (HbyAPtr[owner * 3 + 1] - HbyAPtr[neighbor * 3 + 1]) + HbyAPtr[neighbor * 3 + 1]);
        scalar ssfz_HbyA = (weights * (HbyAPtr[owner * 3 + 2] - HbyAPtr[neighbor * 3 + 2]) + HbyAPtr[neighbor * 3 + 2]);
        scalar vf_interp = (weights * (rhoPtr[owner] - rhoPtr[neighbor]) + rhoPtr[neighbor]);
        phiHbyA_ += (Sfx * ssfx_HbyA + Sfy * ssfy_HbyA + Sfz * ssfz_HbyA) * vf_interp;
        phiHbyAPtr[i] = phiHbyA_;
        phiHbyA[i] = phiHbyA_;
    }


    Info << "Gen_p phiHbyAPtr : " << clock.timeIncrement() << endl;

    // - multi_fvc_flux_fvc_intepolate_boundary
#ifdef _OPENMP
#pragma omp parallel
#endif
    for (label patchi = 0; patchi < nPatches; ++patchi) {
        fvsPatchScalarField& patchphiHbyA = const_cast<fvsPatchScalarField&>(phiHbyA.boundaryField()[patchi]);
        if (patchTypes[patchi] == MeshSchedule::PatchType::processor) {
#ifdef _OPENMP
#pragma omp for
#endif
            for (label f = 0; f < patchSizes[patchi]; ++f) {
                scalar bouWeight = boundary_weights[patchi][f];
                scalar bouSfx = boundary_sf[patchi][3 * f];
                scalar bouSfy = boundary_sf[patchi][3 * f + 1];
                scalar bouSfz = boundary_sf[patchi][3 * f + 2];
                scalar boussfx = (1 - bouWeight) * boundary_HbyA[patchi][3 * f] + bouWeight * boundary_HbyA_internal[patchi][3 * f];
                scalar boussfy = (1 - bouWeight) * boundary_HbyA[patchi][3 * f + 1] +bouWeight * boundary_HbyA_internal[patchi][3 * f + 1];
                scalar boussfz = (1 - bouWeight) * boundary_HbyA[patchi][3 * f + 2] +bouWeight * boundary_HbyA_internal[patchi][3 * f + 2];
                scalar bouvf = (1 - bouWeight) * boundary_rho[patchi][f] + bouWeight * boundary_rho_internal[patchi][f];
                boundary_phiHbyAPtr[patchi][f] = (bouSfx * boussfx + bouSfy * boussfy + bouSfz * boussfz) * bouvf;
                patchphiHbyA[f] = boundary_phiHbyAPtr[patchi][f];
            }
        } else if (patchTypes[patchi] == MeshSchedule::PatchType::wall) {
#ifdef _OPENMP
#pragma omp for
#endif
            for (label f = 0; f < patchSizes[patchi]; ++f) {
                scalar bouSfx = boundary_sf[patchi][3 * f];
                scalar bouSfy = boundary_sf[patchi][3 * f + 1];
                scalar bouSfz = boundary_sf[patchi][3 * f + 2];
                scalar boussfx = boundary_HbyA[patchi][3 * f];
                scalar boussfy = boundary_HbyA[patchi][3 * f + 1];
                scalar boussfz = boundary_HbyA[patchi][3 * f + 2];
                scalar bouvf = boundary_rho[patchi][f];
                boundary_phiHbyAPtr[patchi][f] = (bouSfx * boussfx + bouSfy * boussfy + bouSfz * boussfz) * bouvf;
                patchphiHbyA[f] = boundary_phiHbyAPtr[patchi][f];
            }
        }
    }

    Info << "Gen_p boundary phiHbyAPtr : " << clock.timeIncrement() << endl;

    // fvm_ddt
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (label i = 0; i < nCells; ++i) {
        scalar srcVal = sourcePtr[i];
        assert(srcVal == 0);
        scalar APsi = - diagPtr[i] * pPtr[i] + srcVal;
        scalar tPsiVal = thermoPsiPtr[i];
        diagPtr[i] += rDeltaT * meshVPtr[i] * tPsiVal;
        sourcePtr[i] += (rDeltaT * pOldPtr[i] * meshVPtr[i] - APsi) * tPsiVal - (rhoPtr[i] - rhoOldPtr[i]) * rDeltaT * meshVPtr[i];
    }

    // fvc_div
    // - internal
    
#ifdef OPT_FACE2CELL_COLORING_SCHEDULE

    {
        const XYBlock1DColoringStructuredMeshSchedule& coloringSchedule = XYBlock1DColoringStructuredMeshSchedule::getXYBlock1DColoringStructuredMeshSchedule();
        const labelList& face_scheduling = coloringSchedule.face_scheduling();

        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for(label face_scheduling_i = 0; face_scheduling_i < face_scheduling.size()-1; face_scheduling_i += 2){
            label face_start = face_scheduling[face_scheduling_i]; 
            label face_end = face_scheduling[face_scheduling_i+1];
            for (label j = face_start; j < face_end; ++j) {
                label owner = own[j];
                label neighbor = nei[j];
                scalar issf = phiHbyAPtr[j];
                sourcePtr[owner] -= issf;
                sourcePtr[neighbor] += issf;
            }
        }
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for(label face_scheduling_i = 1; face_scheduling_i < face_scheduling.size(); face_scheduling_i += 2){
            label face_start = face_scheduling[face_scheduling_i]; 
            label face_end = face_scheduling[face_scheduling_i+1];
            for (label j = face_start; j < face_end; ++j) {
                label owner = own[j];
                label neighbor = nei[j];

                scalar issf = phiHbyAPtr[j];
                sourcePtr[owner] -= issf;
                sourcePtr[neighbor] += issf;
            }
        }
        
    }

#else

    for (label j = 0; j < nFaces; ++j) {
        label owner = own[j];
        label neighbor = nei[j];
        scalar issf = phiHbyAPtr[j];
        sourcePtr[owner] -= issf;
        sourcePtr[neighbor] += issf;
    }

#endif
 
    // - boundary
    for (label patchi = 0; patchi < nPatches; ++patchi) {
        for (label f = 0; f < patchSizes[patchi]; ++f) {
            label cellIndex = boundary_face_cell[patchi][f];
            sourcePtr[cellIndex] -= boundary_phiHbyAPtr[patchi][f];
        }
    }

    // fvm_laplacian_surface_scalar_vol_scalar
    // - internal

#ifdef OPT_FACE2CELL_COLORING_SCHEDULE

    {
        const XYBlock1DColoringStructuredMeshSchedule& coloringMeshSchedule = XYBlock1DColoringStructuredMeshSchedule::getXYBlock1DColoringStructuredMeshSchedule();
        const labelList& face_scheduling = coloringMeshSchedule.face_scheduling();
        
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for(label face_scheduling_i = 0; face_scheduling_i < face_scheduling.size()-1; face_scheduling_i += 2){
            label face_start = face_scheduling[face_scheduling_i]; 
            label face_end = face_scheduling[face_scheduling_i+1];
            for (label j = face_start; j < face_end; ++j) {
                label owner = own[j];
                label neighbor = nei[j];
                scalar face_gamma = rhorAUfPtr[j];
                scalar value = face_gamma * magSfPtr[j] * deltaCoeffsPtr[j] * -1;
                upperPtr[j] += value;
                diagPtr[owner] -= value;
                diagPtr[neighbor] -= value;
            }
        }
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for(label face_scheduling_i = 1; face_scheduling_i < face_scheduling.size(); face_scheduling_i += 2){
            label face_start = face_scheduling[face_scheduling_i]; 
            label face_end = face_scheduling[face_scheduling_i+1];
            for (label j = face_start; j < face_end; ++j) {
                label owner = own[j];
                label neighbor = nei[j];
                scalar face_gamma = rhorAUfPtr[j];
                scalar value = face_gamma * magSfPtr[j] * deltaCoeffsPtr[j] * -1;
                upperPtr[j] += value;
                diagPtr[owner] -= value;
                diagPtr[neighbor] -= value;
            }
        }
    }

#else

    for (label j = 0; j < nFaces; ++j) {
        label owner = own[j];
        label neighbor = nei[j];
        scalar face_gamma = rhorAUfPtr[j];
        scalar value = face_gamma * magSfPtr[j] * deltaCoeffsPtr[j] * -1;
        upperPtr[j] += value;
        diagPtr[owner] -= value;
        diagPtr[neighbor] -= value;
    }

#endif


    Info << "Gen_p source diag upper : " << clock.timeIncrement() << endl;

    // - boundary
    for (label patchi = 0; patchi < nPatches; ++patchi) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (label f = 0; f < patchSizes[patchi]; ++f) {
            scalar boundary_value = boundary_rhorAUf[patchi][f] * boundary_mag_sf[patchi][f];
            fvm.internalCoeffs()[patchi][f] = -1 * boundary_value * gradient_internal_coeffs[patchi][f];
            fvm.boundaryCoeffs()[patchi][f] = boundary_value * gradient_boundary_coeffs[patchi][f];
        }
    }

    Info << "Gen_p internalCoeffs boundaryCoeffs : " << clock.timeIncrement() << endl;

    // free ptrs

    for (label patchi = 0; patchi < nPatches; ++patchi) {
        delete [] boundary_phiHbyAPtr[patchi];
        delete [] boundary_rhorAUf[patchi];
        delete [] internal_coeffs[patchi];
        delete [] boundary_coeffs[patchi];
        delete [] value_internal_coeffs[patchi];
        delete [] value_boundary_coeffs[patchi];
        delete [] gradient_internal_coeffs[patchi];
        delete [] gradient_boundary_coeffs[patchi];
        if(patchTypes[patchi] == MeshSchedule::PatchType::processor){
            delete [] boundary_rAU_internal[patchi];
            delete [] boundary_HbyA_internal[patchi];
            delete [] boundary_rho_internal[patchi];
            delete [] boundary_rhoOld_internal[patchi];
            delete [] boundary_Uold_internal[patchi];
        }
    }

    delete [] rhorAUfPtr;
    delete [] phiHbyAPtr;
    delete [] boundary_face_cell;
    delete [] boundary_mag_sf;
    delete [] boundary_weights;
    delete [] boundary_delta_coeffs;
    delete [] boundary_sf;
    delete [] boundary_phi;
    delete [] boundary_phiOld;
    delete [] boundary_phi;
    delete [] boundary_phiHbyAPtr;
    delete [] boundary_rhorAUf;
    delete [] boundary_rAU;
    delete [] boundary_rAU_internal;
    delete [] boundary_HbyA;
    delete [] boundary_HbyA_internal;
    delete [] boundary_rho;
    delete [] boundary_rho_internal;
    delete [] boundary_rhoOld;
    delete [] boundary_rhoOld_internal;
    delete [] boundary_Uold;
    delete [] boundary_Uold_internal;
    delete [] internal_coeffs;
    delete [] boundary_coeffs;
    delete [] value_internal_coeffs;
    delete [] value_boundary_coeffs;
    delete [] gradient_internal_coeffs;
    delete [] gradient_boundary_coeffs;

    Info << "Gen_p free : " << clock.timeIncrement() << endl;
    Info << "Gen_p Total : " << clock.elapsedTime() << endl;
    
    return tfvm;
}

void postProcess_P(
    volScalarField& p,
    fvScalarMatrix& pEqn,
    surfaceScalarField& phi,
    volVectorField& U,
    const volScalarField& rAU,
    const volVectorField& HbyA,
    volScalarField& K,
    volScalarField& dpdt,

    surfaceScalarField& phiHbyA
)
{
    clockTime clock;
    const fvMesh& mesh = p.mesh();

    const MeshSchedule& meshSchedule = getSchedule();

    const label nCells = meshSchedule.nCells();
    const label nFaces = meshSchedule.nFaces();
    const label nPatches = meshSchedule.nPatches();
    const List<MeshSchedule::PatchType> patchTypes = meshSchedule.patchTypes();
    const labelList& patchSizes = meshSchedule.patchSizes();
    scalar rDeltaT = 1.0/mesh.time().deltaTValue();

    // basic variable
    const label *own = &mesh.owner()[0];
    const label *nei = &mesh.neighbour()[0];
    const surfaceScalarField& weights = mesh.surfaceInterpolation::weights(); // interpolation weight (linear)
    const scalar *weightsPtr = &weights[0];
    const scalar *meshVPtr = &mesh.V()[0];
    const scalar *deltaCoeffsPtr = &mesh.deltaCoeffs()[0];
    const scalar *magSfPtr = &mesh.magSf()[0];
    const scalar* const __restrict__ meshSfPtr = &mesh.Sf()[0][0];

    const scalar* __restrict__ diagPtr = &pEqn.diag()[0];
    const scalar* __restrict__ sourcePtr = &pEqn.source()[0];
    const scalar* __restrict__ upperPtr = &pEqn.upper()[0];
    
    // allocate
    typedef scalar* scalarPtr;
    typedef const scalar* constScalarPtr;
    typedef const label* constLabelPtr;

    scalar *pPtr = p.begin();
    scalar *pOldPtr = p.oldTime().begin();
    scalar *flux = new scalar[nFaces];
    scalar *phiPtr = phi.begin();
    scalar *phiHbyAPtr = phiHbyA.begin();
    scalar *UPtr = (scalar*) U.begin();
    const scalar *rAUPtr = rAU.begin();
    scalar *HbyAPtr = (scalar*) HbyA.begin();
    scalar *KPtr = K.begin();
    scalar *dpdtPtr = dpdt.begin();

    constLabelPtr* faceCells = new constLabelPtr[nPatches];
    scalarPtr* boundary_p = new scalarPtr[nPatches];
    scalarPtr* boundary_p_internal = new scalarPtr[nPatches];
    scalarPtr* boundary_u = new scalarPtr[nPatches];
    scalarPtr* boundary_u_internal = new scalarPtr[nPatches];
    scalarPtr* boundary_flux = new scalarPtr[nPatches];
    scalarPtr* boundary_phi = new scalarPtr[nPatches];
    scalarPtr* boundary_phiHbyA = new scalarPtr[nPatches];
    constScalarPtr* boundary_sf = new constScalarPtr[nPatches];
    constScalarPtr* boundary_weights = new constScalarPtr[nPatches];
    scalarPtr* boundary_K = new scalarPtr[nPatches];
    constScalarPtr* internal_coeffs = new constScalarPtr[nPatches];
    constScalarPtr* boundary_coeffs = new constScalarPtr[nPatches];

    for(label patchi = 0; patchi < nPatches; ++patchi) {
        label patchSize = patchSizes[patchi];
        faceCells[patchi] = mesh.boundary()[patchi].faceCells().begin(); // patchSize
        
        fvPatchScalarField& patchp = const_cast<fvPatchScalarField&>(p.boundaryField()[patchi]);
        boundary_p[patchi] = patchp.begin(); // patchSize

        fvsPatchScalarField& patchphi = const_cast<fvsPatchScalarField&>(phi.boundaryField()[patchi]);
        boundary_phi[patchi] = patchphi.begin(); // patchSize

        fvsPatchScalarField& patchphiHbyA = const_cast<fvsPatchScalarField&>(phiHbyA.boundaryField()[patchi]);
        boundary_phiHbyA[patchi] = patchphiHbyA.begin(); // patchSize

        fvPatchVectorField& patchU = const_cast<fvPatchVectorField&>(U.boundaryField()[patchi]);
        boundary_u[patchi] = (scalar*) patchU.begin(); // patchSize * 3

        fvPatchScalarField& patchK = const_cast<fvPatchScalarField&>(K.boundaryField()[patchi]);
        boundary_K[patchi] = patchK.begin(); // patchSize

        boundary_sf[patchi] = (scalar*) mesh.Sf().boundaryField()[patchi].begin(); // patchSize * 3
        boundary_weights[patchi] = mesh.surfaceInterpolation::weights().boundaryField()[patchi].begin(); // patchSize
        internal_coeffs[patchi] = pEqn.internalCoeffs()[patchi].begin(); // patchSize
        boundary_coeffs[patchi] = pEqn.boundaryCoeffs()[patchi].begin(); // patchSize

        boundary_flux[patchi] = new scalar[patchSize];
        if(patchTypes[patchi] == MeshSchedule::PatchType::processor){
            boundary_p_internal[patchi] = new scalar[patchSize];
            boundary_u_internal[patchi] = new scalar[patchSize * 3];
        } else if(patchTypes[patchi] == MeshSchedule::PatchType::wall){
            boundary_p_internal[patchi] = nullptr;
            boundary_u_internal[patchi] = nullptr;
        } else {
            Info << "boundary condition not supported" << endl;
            std::exit(-1);
        }
    }

    Info << "postProcess_P init : " << clock.timeIncrement() << endl;

    // correct boundary conditions
#ifdef _OPENMP
#pragma omp parallel
#endif
    for(label patchi = 0; patchi < nPatches; ++patchi) {
        scalar* boundary_p_patch = boundary_p[patchi];
        scalar* boundary_p_internal_patch = boundary_p_internal[patchi];
        const label* faceCells_patch = faceCells[patchi];
        if (patchTypes[patchi] == MeshSchedule::PatchType::processor){
#ifdef _OPENMP
#pragma omp for
#endif
            for (label p = 0; p < patchSizes[patchi]; ++p) {
                label faceCell = faceCells_patch[p];
                boundary_p_internal_patch[p] = pPtr[faceCell];
            }
        } else {
#ifdef _OPENMP
#pragma omp for
#endif
            for (label p = 0; p < patchSizes[patchi]; ++p) {
                label faceCell = faceCells_patch[p];
                boundary_p_patch[p] = pPtr[faceCell];
            }
        }
    }
    for (label patchi = 0; patchi < nPatches; ++patchi) {
        if (patchTypes[patchi] == MeshSchedule::PatchType::processor){
            MPI_Sendrecv(
                boundary_p_internal[patchi], patchSizes[patchi], MPI_DOUBLE, neighbProcNo[patchi], 0,
                boundary_p[patchi], patchSizes[patchi], MPI_DOUBLE, neighbProcNo[patchi], 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );
        }
    }

    Info << "postProcess_P correct boundary conditions : " << clock.timeIncrement() << endl;
    
    // update phi
    // fvMtx_flux
    // - lduMatrix_faceH
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (label f = 0; f < nFaces; ++f) {
        label owner = own[f];
        label neighbor = nei[f];
        flux[f] = upperPtr[f] * pPtr[neighbor] - upperPtr[f] * pPtr[owner];
    }

    // - boundary_flux
#ifdef _OPENMP
#pragma omp parallel
#endif
    for(label patchi = 0; patchi < nPatches; ++patchi) {
        if (patchTypes[patchi] == MeshSchedule::PatchType::processor){
#ifdef _OPENMP
#pragma omp for
#endif
            for(label p = 0; p < patchSizes[patchi]; ++p){
                scalar internal_contrib = boundary_p_internal[patchi][p] * internal_coeffs[patchi][p];
                scalar neighbor_contrib = boundary_p[patchi][p] * boundary_coeffs[patchi][p];
                boundary_flux[patchi][p] = internal_contrib - neighbor_contrib;
            }
        } else {
#ifdef _OPENMP
#pragma omp for
#endif
            for(label p = 0; p < patchSizes[patchi]; ++p){
                label faceCell = faceCells[patchi][p];
                boundary_flux[patchi][p] = pPtr[faceCell] * internal_coeffs[patchi][p];
            }
        }
    }

    // field_add_scalar
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (label f = 0; f < nFaces; ++f) {
        phiPtr[f] = phiHbyAPtr[f] + flux[f];
    }

    for(label patchi = 0; patchi < nPatches; ++patchi) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(label p = 0; p < patchSizes[patchi]; ++p){
            boundary_phi[patchi][p] = boundary_phiHbyA[patchi][p] + boundary_flux[patchi][p];
        }
    }
    Info << "postProcess_P update phi : " << clock.timeIncrement() << endl;

    // update U
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (label c = 0; c < nCells; ++c) {
        UPtr[c * 3] = 0.;
        UPtr[c * 3 + 1] = 0.;
        UPtr[c * 3 + 2] = 0.;
    }

    // - fvc_grad_scalar_internal
#ifdef OPT_FACE2CELL_COLORING_SCHEDULE

    {
        const XYBlock1DColoringStructuredMeshSchedule& coloringSchedule = XYBlock1DColoringStructuredMeshSchedule::getXYBlock1DColoringStructuredMeshSchedule();
        const labelList& face_scheduling = coloringSchedule.face_scheduling();

        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for(label face_scheduling_i = 0; face_scheduling_i < face_scheduling.size()-1; face_scheduling_i += 2){
            label face_start = face_scheduling[face_scheduling_i]; 
            label face_end = face_scheduling[face_scheduling_i+1];
            for (label j = face_start; j < face_end; ++j) {
                label owner = own[j];
                label neighbor = nei[j];
                scalar w = weightsPtr[j];
                scalar Sfx = meshSfPtr[j * 3];
                scalar Sfy = meshSfPtr[j * 3 + 1];
                scalar Sfz = meshSfPtr[j * 3 + 2];
                scalar ssf = w * (p[owner] - p[neighbor]) + p[neighbor];
                scalar grad_x = Sfx * ssf;
                scalar grad_y = Sfy * ssf;
                scalar grad_z = Sfz * ssf;
                UPtr[owner * 3 + 0] += grad_x;
                UPtr[owner * 3 + 1] += grad_y;
                UPtr[owner * 3 + 2] += grad_z;
                UPtr[neighbor * 3 + 0] -= grad_x;
                UPtr[neighbor * 3 + 1] -= grad_y;
                UPtr[neighbor * 3 + 2] -= grad_z;
            }
        }
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for(label face_scheduling_i = 1; face_scheduling_i < face_scheduling.size(); face_scheduling_i += 2){
            label face_start = face_scheduling[face_scheduling_i]; 
            label face_end = face_scheduling[face_scheduling_i+1];
            for (label j = face_start; j < face_end; ++j) {
                label owner = own[j];
                label neighbor = nei[j];
                scalar w = weightsPtr[j];
                scalar Sfx = meshSfPtr[j * 3];
                scalar Sfy = meshSfPtr[j * 3 + 1];
                scalar Sfz = meshSfPtr[j * 3 + 2];
                scalar ssf = w * (p[owner] - p[neighbor]) + p[neighbor];
                scalar grad_x = Sfx * ssf;
                scalar grad_y = Sfy * ssf;
                scalar grad_z = Sfz * ssf;
                UPtr[owner * 3 + 0] += grad_x;
                UPtr[owner * 3 + 1] += grad_y;
                UPtr[owner * 3 + 2] += grad_z;
                UPtr[neighbor * 3 + 0] -= grad_x;
                UPtr[neighbor * 3 + 1] -= grad_y;
                UPtr[neighbor * 3 + 2] -= grad_z;
            }
        }
    }

#else

    for (label j = 0; j < nFaces; ++j) {
        label owner = own[j];
        label neighbor = nei[j];
        scalar w = weightsPtr[j];
        scalar Sfx = meshSfPtr[j * 3];
        scalar Sfy = meshSfPtr[j * 3 + 1];
        scalar Sfz = meshSfPtr[j * 3 + 2];
        scalar ssf = w * (p[owner] - p[neighbor]) + p[neighbor];
        scalar grad_x = Sfx * ssf;
        scalar grad_y = Sfy * ssf;
        scalar grad_z = Sfz * ssf;
        UPtr[owner * 3 + 0] += grad_x;
        UPtr[owner * 3 + 1] += grad_y;
        UPtr[owner * 3 + 2] += grad_z;
        UPtr[neighbor * 3 + 0] -= grad_x;
        UPtr[neighbor * 3 + 1] -= grad_y;
        UPtr[neighbor * 3 + 2] -= grad_z;
    }

#endif

    // - fvc_grad_scalar_boundary
    for(label patchi = 0; patchi < nPatches; ++patchi) {
        const label* faceCells_patch = faceCells[patchi];
        if (patchTypes[patchi] == MeshSchedule::PatchType::processor){
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (label p = 0; p < patchSizes[patchi]; ++p){
                scalar bouWeight = boundary_weights[patchi][p];
                scalar bouSfx = boundary_sf[patchi][p * 3];
                scalar bouSfy = boundary_sf[patchi][p * 3 + 1];
                scalar bouSfz = boundary_sf[patchi][p * 3 + 2];
                scalar boup = (1 - bouWeight) * boundary_p[patchi][p] + bouWeight * boundary_p_internal[patchi][p];
                label faceCell = faceCells_patch[p];
                scalar grad_x = bouSfx * boup;
                scalar grad_y = bouSfy * boup;
                scalar grad_z = bouSfz * boup;
                UPtr[faceCell * 3] += grad_x;
                UPtr[faceCell * 3 + 1] += grad_y;
                UPtr[faceCell * 3 + 2] += grad_z;
            }
        } else {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (label p = 0; p < patchSizes[patchi]; ++p){
                scalar boup = boundary_p[patchi][p];
                scalar bouSfx = boundary_sf[patchi][p * 3];
                scalar bouSfy = boundary_sf[patchi][p * 3 + 1];
                scalar bouSfz = boundary_sf[patchi][p * 3 + 2];
                label faceCell = faceCells_patch[p];
                scalar grad_x = bouSfx * boup;
                scalar grad_y = bouSfy * boup;
                scalar grad_z = bouSfz * boup;
                UPtr[faceCell * 3] += grad_x;
                UPtr[faceCell * 3 + 1] += grad_y;
                UPtr[faceCell * 3 + 2] += grad_z;
            }
        }
    }

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (label c = 0; c < nCells; ++c) {
        scalar meshVRTmp = 1. / meshVPtr[c];
        scalar rAU = rAUPtr[c];
        UPtr[c * 3 + 0] *= meshVRTmp * rAU;
        UPtr[c * 3 + 1] *= meshVRTmp * rAU;
        UPtr[c * 3 + 2] *= meshVRTmp * rAU;
        UPtr[c * 3 + 0] = HbyAPtr[c * 3 + 0] - UPtr[c * 3 + 0];
        UPtr[c * 3 + 1] = HbyAPtr[c * 3 + 1] - UPtr[c * 3 + 1];
        UPtr[c * 3 + 2] = HbyAPtr[c * 3 + 2] - UPtr[c * 3 + 2];
    }

    Info << "postProcess_P update U : " << clock.timeIncrement() << endl;

    // - correct_boundary_conditions_vector
#ifdef _OPENMP
#pragma omp parallel
#endif
    for(label patchi = 0; patchi < nPatches; ++patchi) {
        scalar* boundary_u_patch = boundary_u[patchi];
        scalar* boundary_u_internal_patch = boundary_u_internal[patchi];
        const label* faceCells_patch = faceCells[patchi];
        if (patchTypes[patchi] == MeshSchedule::PatchType::processor){
#ifdef _OPENMP
#pragma omp for
#endif
            for (label p = 0; p < patchSizes[patchi]; ++p) {
                label faceCell = faceCells_patch[p];
                boundary_u_internal_patch[p * 3] = UPtr[faceCell * 3];
                boundary_u_internal_patch[p * 3 + 1] = UPtr[faceCell * 3 + 1];
                boundary_u_internal_patch[p * 3 + 2] = UPtr[faceCell * 3 + 2];
            }
        } else {
#ifdef _OPENMP
#pragma omp for
#endif
            for (label p = 0; p < patchSizes[patchi]; ++p) {
                label faceCell = faceCells_patch[p];
                boundary_u_patch[p * 3] = UPtr[faceCell * 3];
                boundary_u_patch[p * 3 + 1] = UPtr[faceCell * 3 + 1];
                boundary_u_patch[p * 3 + 2] = UPtr[faceCell * 3 + 2];
            }
        }
    }
    for (label patchi = 0; patchi < nPatches; ++patchi) {
        if (patchTypes[patchi] == MeshSchedule::PatchType::processor){
            MPI_Sendrecv(
                boundary_u_internal[patchi], patchSizes[patchi] * 3, MPI_DOUBLE, neighbProcNo[patchi], 0,
                boundary_u[patchi], patchSizes[patchi] * 3, MPI_DOUBLE, neighbProcNo[patchi], 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );
        }
    }
    Info << "postProcess_P update U boundary : " << clock.timeIncrement() << endl;

    // vector_half_mag_square
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (label c = 0; c < nCells; ++c) {
        scalar u = UPtr[c * 3 + 1];
        scalar v = UPtr[c * 3 + 1];
        scalar w = UPtr[c * 3 + 2];
        KPtr[c] = 0.5 * (u * u + v * v + w * w);
    }

    for (label patchi = 0; patchi < nPatches; ++patchi) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (label p = 0; p < patchSizes[patchi]; ++p) {
            scalar bouu = boundary_u[patchi][p * 3];
            scalar bouv = boundary_u[patchi][p * 3 + 1];
            scalar bouw = boundary_u[patchi][p * 3 + 2];
            boundary_K[patchi][p] = 0.5 * (bouu * bouu + bouv * bouv + bouw * bouw);
        }
    }
    Info << "postProcess_P update K : " << clock.timeIncrement() << endl;

    // calculate dpdt
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (label c = 0; c < nCells; ++c) {
        dpdtPtr[c] = (pPtr[c] - pOldPtr[c]) * rDeltaT;
    }

    Info << "postProcess_P update dpdt : " << clock.timeIncrement() << endl;


    for(label patchi = 0; patchi < nPatches; ++patchi) {
        delete [] boundary_flux[patchi];
        if(patchTypes[patchi] == MeshSchedule::PatchType::processor){
            delete [] boundary_p_internal[patchi];
            delete [] boundary_u_internal[patchi];
        }
    }

    delete [] flux;
    delete [] faceCells;
    delete [] boundary_p;
    delete [] boundary_p_internal;
    delete [] boundary_u;
    delete [] boundary_u_internal;
    delete [] boundary_flux;
    delete [] boundary_phi;
    delete [] boundary_phiHbyA;
    delete [] boundary_sf;
    delete [] boundary_weights;
    delete [] boundary_K;
    delete [] internal_coeffs;
    delete [] boundary_coeffs;
    Info << "postProcess_P free : " << clock.timeIncrement() << endl;
    Info << "postProcess_P total : " << clock.elapsedTime() << endl;
}

// tmp<fvScalarMatrix>
// GenMatrix_p(
//     const volScalarField& rho,
//     volScalarField& p,
//     const surfaceScalarField& phiHbyA,
//     const surfaceScalarField& rhorAUf,
//     const volScalarField& psi
// )
// {
//     const fvMesh& mesh = p.mesh();
//     assert(mesh.moving() == false);

//     label nCells = mesh.nCells();

//     // basic matrix
//     tmp<fvScalarMatrix> tfvm
//     (
//         new fvScalarMatrix
//         (
//             p,
//             psi.dimensions()*p.dimensions()*dimVol/dimTime
//         )
//     );
//     fvScalarMatrix& fvm = tfvm.ref();

//     scalar* __restrict__ diagPtr = fvm.diag().begin();
//     scalar* __restrict__ sourcePtr = fvm.source().begin();
//     scalar* __restrict__ upperPtr = fvm.upper().begin();

//     const labelUList& l = fvm.lduAddr().lowerAddr();
//     const labelUList& u = fvm.lduAddr().upperAddr();

//     scalar rDeltaT = 1.0/mesh.time().deltaTValue();

//     // fvmddt
//     auto fvmDdtTmp = psi * correction(EulerDdtSchemeFvmDdt(p));

//     // fvcddt
//     // auto fvcDdtTmp = EulerDdtSchemeFvcDdt(rho);
//     const scalar* const __restrict__ rhoPtr = rho.primitiveField().begin();
//     const scalar* const __restrict__ rhoOldTimePtr = rho.oldTime().primitiveField().begin();
//     const scalar* const __restrict__ meshVPtr = mesh.V().begin();

//     // fvcdiv
//     // auto fvcDivTmp = gaussConvectionSchemeFvcDiv(phiHbyA);
//     const scalar* const __restrict__ phiHbyAPtr = phiHbyA.primitiveField().begin();

//     // fvmLaplacian
//     // auto fvmLaplacianTmp = gaussLaplacianSchemeFvmLaplacian(rhorAUf, p);
//     tmp<fv::snGradScheme<scalar>> tsnGradScheme_(new fv::orthogonalSnGrad<scalar>(mesh));
//     const surfaceScalarField& deltaCoeffs = tsnGradScheme_().deltaCoeffs(p)();

//     surfaceScalarField gammaMagSf
//     (
//         rhorAUf * mesh.magSf()
//     );

//     const scalar* const __restrict__ deltaCoeffsPtr = deltaCoeffs.primitiveField().begin();
//     const scalar* const __restrict__ gammaMagSfPtr = gammaMagSf.primitiveField().begin();
//     // Info << "fvmLaplacian = " << time_end - time_begin << endl;

//     // merge
//     double *fvcDivPtr = new double[nCells]{0.};
    
//     for(label f = 0; f < nFaces; ++f){
//         scalar var1 = deltaCoeffsPtr[f] * gammaMagSfPtr[f];
//         // lowerPtr[f] = var1;
//         upperPtr[f] = var1;
//         diagPtr[l[f]] -= var1;
//         diagPtr[u[f]] -= var1;
//         fvcDivPtr[l[f]] += phiHbyAPtr[f];
//         fvcDivPtr[u[f]] -= phiHbyAPtr[f];
//     }

//     // - boundary loop
//     forAll(p.boundaryField(), patchi)
//     {
//         const fvPatchField<scalar>& psf = p.boundaryField()[patchi];
//         const fvsPatchScalarField& pGamma = gammaMagSf.boundaryField()[patchi];
//         const fvsPatchScalarField& pDeltaCoeffs =
//             deltaCoeffs.boundaryField()[patchi];

//         const labelUList& pFaceCells =
//             mesh.boundary()[patchi].faceCells();

//         const fvsPatchField<scalar>& pssf = phiHbyA.boundaryField()[patchi];

//         if (psf.coupled())
//         {
//             fvm.internalCoeffs()[patchi] =
//                 pGamma*psf.gradientInternalCoeffs(pDeltaCoeffs);
//             fvm.boundaryCoeffs()[patchi] =
//                 -pGamma*psf.gradientBoundaryCoeffs(pDeltaCoeffs);
//         }
//         else
//         {
//             fvm.internalCoeffs()[patchi] = pGamma*psf.gradientInternalCoeffs();
//             fvm.boundaryCoeffs()[patchi] = -pGamma*psf.gradientBoundaryCoeffs();
//         }
//         forAll(mesh.boundary()[patchi], facei)
//         {
//             fvcDivPtr[pFaceCells[facei]] += pssf[facei];
//         }
//     }

//     // - cell loop

//     #pragma omp parallel for
//     for(label c = 0; c < nCells; ++c){
//         sourcePtr[c] += rDeltaT * (rhoPtr[c] - rhoOldTimePtr[c]) * meshVPtr[c];
//         sourcePtr[c] += fvcDivPtr[c];
//     }

//     tmp<fvScalarMatrix> fvm_final
//     (
//         fvmDdtTmp 
//         // + fvcDivTmp
//         - fvm
//     );

//     return fvm_final;
// }

template
Foam::tmp<Foam::GeometricField<vector, Foam::fvPatchField, Foam::volMesh>>
UEqn_H
(
    fvMatrix<vector>& UEqn
);

}

