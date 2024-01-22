#include "divMatrix.H"
#include <cassert>
#include <sstream>
#include <cstdio>
#include "PstreamGlobals.H"
#include "Residuals.H"

namespace Foam{


const label divMatrix::solver::defaultMaxIter_ = 1000;

template<class Type>
SolverPerformance<Type> divMatrix::solve(
    GeometricField<Type, fvPatchField, volMesh>& psi,
    const Field<Type>& source,
    const FieldField<Field, Type>& internalCoeffs,
    const FieldField<Field, Type>& boundaryCoeffs,
    const dictionary& solverControls
){
    double addBoundarySource_time = 0.;
    double addBoundaryDiag_time = 0.;
    double initMatrixInterfaces_time = 0.;
    double updateMatrixInterfaces_time = 0.;
    double solver_construct_time = 0.;
    double solve_time = 0.;
    double psi_correctBoundaryConditions_time = 0.;

    clockTime clock;

    SolverPerformance<Type> solverPerfVec
    (
        "divMatrix::solve",
        psi.name()
    );

    scalarField saveDiag(diag());

    Field<Type> sourceCpy(source);

    addBoundarySource(sourceCpy, psi, boundaryCoeffs);

    addBoundarySource_time += clock.timeIncrement();

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

        addBoundaryDiag_time += clock.timeIncrement();

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

        initMatrixInterfaces_time += clock.timeIncrement();

        updateMatrixInterfaces
        (
            bouCoeffsCmpt,
            interfaces,
            psiCmpt,
            sourceCmpt,
            cmpt
        );

        updateMatrixInterfaces_time += clock.timeIncrement();

        solverPerformance solverPerf;

        Foam::autoPtr<Foam::divMatrix::solver> solver = divMatrix::solver::New
        (
            psi.name() + pTraits<Type>::componentNames[cmpt],
            *this,
            bouCoeffsCmpt,
            intCoeffsCmpt,
            interfaces,
            solverControls
        );

        solver_construct_time += clock.timeIncrement();

        // Solver call
        solverPerf = solver->solve(psiCmpt, sourceCmpt, cmpt);

        solve_time += clock.timeIncrement();

#ifdef _PROFILING_
        solver->print_time();
#endif    

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
        
    psi_correctBoundaryConditions_time += clock.timeIncrement();

    Residuals<Type>::append(psi.mesh(), solverPerfVec);

    double total_solve_time = clock.elapsedTime();

#ifdef _PROFILING_
    Info << "divMatrix solve<Type> profiling ----------------------------" << solve_time << endl;
    Info << "Total solve time : " << total_solve_time << endl;
    Info << "addBoundaryDiag_time : " << addBoundaryDiag_time << ", " << addBoundaryDiag_time / total_solve_time * 100 << "%" << endl;
    Info << "addBoundarySource_time : " << addBoundarySource_time << ", " << addBoundarySource_time / total_solve_time * 100 << "%" << endl;
    Info << "initMatrixInterfaces_time : " << initMatrixInterfaces_time << ", " << initMatrixInterfaces_time / total_solve_time * 100 << "%" << endl;
    Info << "updateMatrixInterfaces_time : " << updateMatrixInterfaces_time << ", " << updateMatrixInterfaces_time / total_solve_time * 100 << "%" << endl;
    Info << "solver_construct_time : " << solver_construct_time << ", " << solver_construct_time / total_solve_time * 100 << "%" << endl;
    Info << "solve_time : " << solve_time << ", " << solve_time / total_solve_time * 100 << "%" << endl;
    Info << "psi_correctBoundaryConditions_time : " << psi_correctBoundaryConditions_time << ", " << psi_correctBoundaryConditions_time / total_solve_time * 100 << "%" << endl;
    Info << "-------------------------------------------------------------" << solve_time << endl;
#endif    

    return solverPerfVec;
}

template<>
solverPerformance divMatrix::solve
(
    GeometricField<scalar, fvPatchField, volMesh>& psi,
    const Field<scalar>& source,
    const FieldField<Field, scalar>& internalCoeffs,
    const FieldField<Field, scalar>& boundaryCoeffs,
    const dictionary& solverControls
)
{
    double addBoundaryDiag_time = 0.;
    double addBoundarySource_time = 0.;
    double solver_construct_time = 0.;
    double solve_time = 0.;
    double psi_correctBoundaryConditions_time = 0.;
    double total_solve_time = 0.;

    clockTime clock;

    scalarField saveDiag(diag());
    addBoundaryDiag(diag(), internalCoeffs, 0);
    const_cast<scalarField&>(ldu().diag()) = diag();

    addBoundaryDiag_time += clock.timeIncrement();

    scalarField sourceCpy(source);
    addBoundarySource(sourceCpy, psi, boundaryCoeffs, false);

    addBoundarySource_time += clock.timeIncrement();

    Foam::autoPtr<Foam::divMatrix::solver> solver = divMatrix::solver::New
    (
        psi.name(),
        *this,
        boundaryCoeffs,
        internalCoeffs,
        psi.boundaryField().scalarInterfaces(),
        solverControls
    );
    solver_construct_time += clock.timeIncrement();
    
    // Solver call
    solverPerformance solverPerf = solver->solve(psi.primitiveFieldRef(), sourceCpy);

    solve_time += clock.timeIncrement();

    if (solverPerformance::debug)
    {
        solverPerf.print(Info.masterStream(mesh().comm()));
    }

    diag() = saveDiag;
    const_cast<scalarField&>(ldu().diag()) = saveDiag;

    psi.correctBoundaryConditions();

    Residuals<scalar>::append(psi.mesh(), solverPerf);

    psi_correctBoundaryConditions_time += clock.timeIncrement();

    total_solve_time += clock.elapsedTime();
#ifdef _PROFILING_
    Info << "divMatrix solve<scalar> profiling ----------------------------" << solve_time << endl;
    Info << "Total solve time : " << total_solve_time << endl;
    Info << "addBoundaryDiag_time : " << addBoundaryDiag_time << ", " << addBoundaryDiag_time / total_solve_time * 100 << "%" << endl;
    Info << "addBoundarySource_time : " << addBoundarySource_time << ", " << addBoundarySource_time / total_solve_time * 100 << "%" << endl;
    Info << "solver_construct_time : " << solver_construct_time << ", " << solver_construct_time / total_solve_time * 100 << "%" << endl;
    Info << "solve_time : " << solve_time << ", " << solve_time / total_solve_time * 100 << "%" << endl;
    Info << "psi_correctBoundaryConditions_time : " << psi_correctBoundaryConditions_time << ", " << psi_correctBoundaryConditions_time / total_solve_time * 100 << "%" << endl;
    Info << "-------------------------------------------------------------" << solve_time << endl;
    solver->print_time();
#endif    

    return solverPerf;
}

template<class Type>
SolverPerformance<Type> divMatrix::solve(
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
SolverPerformance<Type> divMatrix::solve(
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
void divMatrix::addBoundarySource
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
void divMatrix::addBoundaryDiag
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
void divMatrix::addToInternalField
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
void divMatrix::addToInternalField
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
void divMatrix::addToInternalField<scalar>
(
    const labelUList& addr,
    const Field<scalar>& pf,
    Field<scalar>& intf
) const;

template
void divMatrix::addToInternalField<vector>
(
    const labelUList& addr,
    const Field<vector>& pf,
    Field<vector>& intf
) const;

template
void divMatrix::addBoundarySource<scalar>
(
    Field<scalar>& source,
    const GeometricField<scalar, fvPatchField, volMesh>& psi,
    const FieldField<Field, scalar>& boundaryCoeffs,
    const bool couples
) const;

template
void divMatrix::addBoundarySource<vector>
(
    Field<vector>& source,
    const GeometricField<vector, fvPatchField, volMesh>& psi,
    const FieldField<Field, vector>& boundaryCoeffs,
    const bool couples
) const;

template
void divMatrix::addBoundaryDiag<scalar>
(
    scalarField& diag,
    const FieldField<Field, scalar>& internalCoeffs,
    const direction solveCmpt
) const;

template
void divMatrix::addBoundaryDiag<vector>
(
    scalarField& diag,
    const FieldField<Field, vector>& internalCoeffs,
    const direction solveCmpt
) const;

template
SolverPerformance<vector> divMatrix::solve<vector>(
    GeometricField<vector, fvPatchField, volMesh>& psi,
    const Field<vector>& source,
    const FieldField<Field, vector>& internalCoeffs,
    const FieldField<Field, vector>& boundaryCoeffs,
    const dictionary& solverControls
);

template
SolverPerformance<scalar> divMatrix::solve<scalar>(
    GeometricField<scalar, fvPatchField, volMesh>& psi,
    const Field<scalar>& source,
    const FieldField<Field, scalar>& internalCoeffs,
    const FieldField<Field, scalar>& boundaryCoeffs
);

template
SolverPerformance<vector> divMatrix::solve<vector>(
    GeometricField<vector, fvPatchField, volMesh>& psi,
    const Field<vector>& source,
    const FieldField<Field, vector>& internalCoeffs,
    const FieldField<Field, vector>& boundaryCoeffs
);

template
SolverPerformance<scalar> divMatrix::solve<scalar>(
    GeometricField<scalar, fvPatchField, volMesh>& psi,
    const Field<scalar>& source,
    const FieldField<Field, scalar>& internalCoeffs,
    const FieldField<Field, scalar>& boundaryCoeffs,
    const word& name
);

template
SolverPerformance<vector> divMatrix::solve<vector>(
    GeometricField<vector, fvPatchField, volMesh>& psi,
    const Field<vector>& source,
    const FieldField<Field, vector>& internalCoeffs,
    const FieldField<Field, vector>& boundaryCoeffs,
    const word& name
);




}