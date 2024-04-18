#include "ellMatrix.H"
#include <cassert>
#include <sstream>
#include <cstdio>
#include "PstreamGlobals.H"
#include "Residuals.H"

namespace Foam{


const label ellMatrix::solver::defaultMaxIter_ = 1000;

template<class Type>
SolverPerformance<Type> ellMatrix::solve(
    GeometricField<Type, fvPatchField, volMesh>& psi,
    const Field<Type>& source,
    const FieldField<Field, Type>& internalCoeffs,
    const FieldField<Field, Type>& boundaryCoeffs,
    const dictionary& solverControls
){
    SolverPerformance<Type> solverPerfVec
    (
        "ellMatrix::solve",
        psi.name()
    );

    scalarField saveDiag(diag());

    Field<Type> sourceCpy(source);

    addBoundarySource(sourceCpy, psi, boundaryCoeffs);

    typename Type::labelType validComponents
    (
        psi.mesh().template validComponents<Type>()
    );

    for (direction cmpt=0; cmpt<Type::nComponents; cmpt++)
    {
        if (validComponents[cmpt] == -1) continue;

        // copy field and source
        scalarField psiCmpt(psi.primitiveField().component(cmpt));
        addBoundaryDiag(diag(), internalCoeffs, cmpt);

        scalarField sourceCmpt(sourceCpy.component(cmpt));

        FieldField<Field, scalar> bouCoeffsCmpt
        (
            boundaryCoeffs.component(cmpt)
        );

        FieldField<Field, scalar> intCoeffsCmpt
        (
            internalCoeffs.component(cmpt)
        );

        lduInterfaceFieldPtrsList interfaces =
            psi.boundaryField().scalarInterfaces();

        // Use the initMatrixInterfaces and updateMatrixInterfaces to correct
        // bouCoeffsCmpt for the explicit part of the coupled boundary
        // conditions
        initMatrixInterfaces
        (
            bouCoeffsCmpt,
            interfaces,
            psiCmpt,
            sourceCmpt,
            cmpt
        );

        updateMatrixInterfaces
        (
            bouCoeffsCmpt,
            interfaces,
            psiCmpt,
            sourceCmpt,
            cmpt
        );

        solverPerformance solverPerf;

        // Solver call
        solverPerf = ellMatrix::solver::New
        (
            psi.name() + pTraits<Type>::componentNames[cmpt],
            *this,
            bouCoeffsCmpt,
            intCoeffsCmpt,
            interfaces,
            solverControls
        )->solve(psiCmpt, sourceCmpt, cmpt);

        if (SolverPerformance<Type>::debug)
        {
            solverPerf.print(Info.masterStream(this->mesh().comm()));
        }

        solverPerfVec.replace(cmpt, solverPerf);
        solverPerfVec.solverName() = solverPerf.solverName();

        psi.primitiveFieldRef().replace(cmpt, psiCmpt);
        diag() = saveDiag;
    }

    psi.correctBoundaryConditions();

    Residuals<Type>::append(psi.mesh(), solverPerfVec);

    return solverPerfVec;
}

template<>
solverPerformance ellMatrix::solve
(
    GeometricField<scalar, fvPatchField, volMesh>& psi,
    const Field<scalar>& source,
    const FieldField<Field, scalar>& internalCoeffs,
    const FieldField<Field, scalar>& boundaryCoeffs,
    const dictionary& solverControls
)
{
    scalarField saveDiag(diag());
    addBoundaryDiag(diag(), internalCoeffs, 0);
    const_cast<scalarField&>(ldu().diag()) = diag();

    scalarField sourceCpy(source);
    addBoundarySource(sourceCpy, psi, boundaryCoeffs, false);

    Foam::autoPtr<Foam::ellMatrix::solver> solver = ellMatrix::solver::New
    (
        psi.name(),
        *this,
        boundaryCoeffs,
        internalCoeffs,
        psi.boundaryField().scalarInterfaces(),
        solverControls
    );
    
    // Solver call
    solverPerformance solverPerf = solver->solve(psi.primitiveFieldRef(), sourceCpy);

    if (solverPerformance::debug)
    {
        solverPerf.print(Info.masterStream(mesh().comm()));
    }

    diag() = saveDiag;
    const_cast<scalarField&>(ldu().diag()) = saveDiag;

    psi.correctBoundaryConditions();

    Residuals<scalar>::append(psi.mesh(), solverPerf);

#ifdef _PROFILING_
    solver->print_time();
#endif    

    return solverPerf;
}

template<class Type>
SolverPerformance<Type> ellMatrix::solve(
    GeometricField<Type, fvPatchField, volMesh>& psi,
    const Field<Type>& source,
    const FieldField<Field, Type>& internalCoeffs,
    const FieldField<Field, Type>& boundaryCoeffs

){
    const dictionary solverControls = psi.mesh().solverDict
    (
        psi.select
        (
            psi.mesh().data::template lookupOrDefault<bool>
            ("finalIteration", false)
        )
    );
    return solve(psi,source,internalCoeffs,boundaryCoeffs,solverControls);
}

template<class Type>
SolverPerformance<Type> ellMatrix::solve(
    GeometricField<Type, fvPatchField, volMesh>& psi,
    const Field<Type>& source,
    const FieldField<Field, Type>& internalCoeffs,
    const FieldField<Field, Type>& boundaryCoeffs,
    const word& name
){
    const dictionary solverControls = psi.mesh().solverDict
    (
        psi.mesh().data::template lookupOrDefault<bool>
        ("finalIteration", false)
        ? word(name + "Final")
        : name
    );
    return solve(psi,source,internalCoeffs,boundaryCoeffs,solverControls);
}


template<class Type>
void ellMatrix::addBoundarySource
(
    Field<Type>& source,
    const GeometricField<Type, fvPatchField, volMesh>& psi,
    const FieldField<Field, Type>& boundaryCoeffs,
    const bool couples
) const {
    forAll(psi.boundaryField(), patchi)
    {
        const fvPatchField<Type>& ptf = psi.boundaryField()[patchi];
        const Field<Type>& pbc = boundaryCoeffs[patchi];

        if (!ptf.coupled())
        {
            addToInternalField(lduAddr().patchAddr(patchi), pbc, source);
        }
        else if (couples)
        {
            const tmp<Field<Type>> tpnf = ptf.patchNeighbourField();
            const Field<Type>& pnf = tpnf();

            const labelUList& addr = lduAddr().patchAddr(patchi);

            forAll(addr, facei)
            {
                source[addr[facei]] += cmptMultiply(pbc[facei], pnf[facei]);
            }
        }
    }
}

template<class Type>
void ellMatrix::addBoundaryDiag
(
    scalarField& diag,
    const FieldField<Field, Type>& internalCoeffs,
    const direction solveCmpt
) const {
    forAll(internalCoeffs, patchi)
    {
        addToInternalField
        (
            lduAddr().patchAddr(patchi),
            internalCoeffs[patchi].component(solveCmpt),
            diag
        );
    }
}


template<class Type2>
void ellMatrix::addToInternalField
(
    const labelUList& addr,
    const Field<Type2>& pf,
    Field<Type2>& intf
) const {
    if (addr.size() != pf.size())
    {
        FatalErrorInFunction
            << "sizes of addressing and field are different"
            << abort(FatalError);
    }

    forAll(addr, facei)
    {
        intf[addr[facei]] += pf[facei];
    }
}

template<class Type2>
void ellMatrix::addToInternalField
(
    const labelUList& addr,
    const tmp<Field<Type2>>& tpf,
    Field<Type2>& intf
) const
{
    addToInternalField(addr, tpf(), intf);
    tpf.clear();
}

template
void ellMatrix::addToInternalField<scalar>
(
    const labelUList& addr,
    const Field<scalar>& pf,
    Field<scalar>& intf
) const;

template
void ellMatrix::addToInternalField<vector>
(
    const labelUList& addr,
    const Field<vector>& pf,
    Field<vector>& intf
) const;

template
void ellMatrix::addBoundarySource<scalar>
(
    Field<scalar>& source,
    const GeometricField<scalar, fvPatchField, volMesh>& psi,
    const FieldField<Field, scalar>& boundaryCoeffs,
    const bool couples
) const;

template
void ellMatrix::addBoundarySource<vector>
(
    Field<vector>& source,
    const GeometricField<vector, fvPatchField, volMesh>& psi,
    const FieldField<Field, vector>& boundaryCoeffs,
    const bool couples
) const;

template
void ellMatrix::addBoundaryDiag<scalar>
(
    scalarField& diag,
    const FieldField<Field, scalar>& internalCoeffs,
    const direction solveCmpt
) const;

template
void ellMatrix::addBoundaryDiag<vector>
(
    scalarField& diag,
    const FieldField<Field, vector>& internalCoeffs,
    const direction solveCmpt
) const;

template
SolverPerformance<vector> ellMatrix::solve<vector>(
    GeometricField<vector, fvPatchField, volMesh>& psi,
    const Field<vector>& source,
    const FieldField<Field, vector>& internalCoeffs,
    const FieldField<Field, vector>& boundaryCoeffs,
    const dictionary& solverControls
);

template
SolverPerformance<scalar> ellMatrix::solve<scalar>(
    GeometricField<scalar, fvPatchField, volMesh>& psi,
    const Field<scalar>& source,
    const FieldField<Field, scalar>& internalCoeffs,
    const FieldField<Field, scalar>& boundaryCoeffs
);

template
SolverPerformance<vector> ellMatrix::solve<vector>(
    GeometricField<vector, fvPatchField, volMesh>& psi,
    const Field<vector>& source,
    const FieldField<Field, vector>& internalCoeffs,
    const FieldField<Field, vector>& boundaryCoeffs
);

template
SolverPerformance<scalar> ellMatrix::solve<scalar>(
    GeometricField<scalar, fvPatchField, volMesh>& psi,
    const Field<scalar>& source,
    const FieldField<Field, scalar>& internalCoeffs,
    const FieldField<Field, scalar>& boundaryCoeffs,
    const word& name
);

template
SolverPerformance<vector> ellMatrix::solve<vector>(
    GeometricField<vector, fvPatchField, volMesh>& psi,
    const Field<vector>& source,
    const FieldField<Field, vector>& internalCoeffs,
    const FieldField<Field, vector>& boundaryCoeffs,
    const word& name
);




}