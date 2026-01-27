#include "dfMatrix.H"
#include <cassert>
#include <sstream>
#include <cstdio>
#include "PstreamGlobals.H"
#include "Residuals.H"
#include "clockTime.H"

namespace Foam{

template<class Type>
SolverPerformance<Type> dfMatrix::solve(
    GeometricField<Type, fvPatchField, volMesh>& psi,
    const Field<Type>& source,
    const FieldField<Field, Type>& internalCoeffs,
    const FieldField<Field, Type>& boundaryCoeffs,
    const dictionary& solverControls
){
    double misc_time = 0.;
    double diag_copy_time = 0.;
    double addBoundaryDiag_time = 0.;
    double addBoundarySource_time = 0.;
    double initMatrixInterfaces_time = 0.;
    double updateMatrixInterfaces_time = 0.;
    double build_solver_time = 0.;
    double solver_solve_time = 0.;
    double correctBoundaryConditions_time = 0.;

    clockTime solveClock;

    SolverPerformance<Type> solverPerfVec
    (
        "dfMatrix::solve",
        psi.name()
    );

    misc_time += solveClock.timeIncrement();

    scalarField saveDiag(diag());

    Field<Type> sourceCpy(source);

    diag_copy_time += solveClock.timeIncrement();

    addBoundarySource(sourceCpy, psi, boundaryCoeffs);

    addBoundarySource_time += solveClock.timeIncrement();

    typename Type::labelType validComponents
    (
        psi.mesh().template validComponents<Type>()
    );

    for (direction cmpt=0; cmpt<Type::nComponents; cmpt++)
    {
        if (validComponents[cmpt] == -1) continue;

        // copy field and source
        scalarField psiCmpt(psi.primitiveField().component(cmpt));

        misc_time += solveClock.timeIncrement();
        
        addBoundaryDiag(diag(), internalCoeffs, cmpt);

        addBoundaryDiag_time += solveClock.timeIncrement();

        const_cast<scalarField&>(ldu().diag()) = diag();

        scalarField sourceCmpt(sourceCpy.component(cmpt));

        diag_copy_time += solveClock.timeIncrement();

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

        misc_time += solveClock.timeIncrement();

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

        initMatrixInterfaces_time += solveClock.timeIncrement();

        updateMatrixInterfaces
        (
            bouCoeffsCmpt,
            interfaces,
            psiCmpt,
            sourceCmpt,
            cmpt
        );

        updateMatrixInterfaces_time += solveClock.timeIncrement();

        // Solver call
        // solverPerf = dfMatrix::solver::New
        // (
        //     psi.name() + pTraits<Type>::componentNames[cmpt],
        //     *this,
        //     bouCoeffsCmpt,
        //     intCoeffsCmpt,
        //     interfaces,
        //     solverControls
        // )->solve(psiCmpt, sourceCmpt, cmpt);

        Foam::autoPtr<Foam::dfMatrix::solver> solver = dfMatrix::solver::New
        (
            psi.name() + pTraits<Type>::componentNames[cmpt],
            *this,
            bouCoeffsCmpt,
            intCoeffsCmpt,
            interfaces,
            solverControls
        );

        build_solver_time += solveClock.timeIncrement();

        solverPerformance solverPerf = solver->solve(psiCmpt, sourceCmpt, cmpt);

        solver_solve_time += solveClock.timeIncrement(); 

        if (SolverPerformance<Type>::debug)
        {
            solverPerf.print(Info.masterStream(this->mesh().comm()));
        }

        solverPerfVec.replace(cmpt, solverPerf);
        solverPerfVec.solverName() = solverPerf.solverName();

        psi.primitiveFieldRef().replace(cmpt, psiCmpt);

        misc_time += solveClock.timeIncrement();

        diag() = saveDiag;
        const_cast<scalarField&>(ldu().diag()) = saveDiag;

        diag_copy_time += solveClock.timeIncrement();
    }

    psi.correctBoundaryConditions();

    correctBoundaryConditions_time += solveClock.timeIncrement();

    Residuals<Type>::append(psi.mesh(), solverPerfVec);

    misc_time += solveClock.timeIncrement();
    double solve_time = solveClock.elapsedTime();

    // print time
    Info << "dfMatrix::solve profiling -----------------------------------------------------------------" << endl;
    Info << "solve time : " << solve_time << endl;
    Info << "diag_copy time : " << diag_copy_time << ", " << diag_copy_time / solve_time * 100 << "%" << endl;
    Info << "addBoundaryDiag time : " << addBoundaryDiag_time << ", " << addBoundaryDiag_time / solve_time * 100 << "%" << endl;
    Info << "addBoundarySource time : " << addBoundarySource_time << ", " << addBoundarySource_time / solve_time * 100 << "%" << endl;
    Info << "initMatrixInterfaces time : " << initMatrixInterfaces_time << ", " << initMatrixInterfaces_time / solve_time * 100 << "%" << endl;
    Info << "updateMatrixInterfaces time : " << updateMatrixInterfaces_time << ", " << updateMatrixInterfaces_time / solve_time * 100 << "%" << endl;
    Info << "build_solver time : " << build_solver_time << ", " << build_solver_time / solve_time * 100 << "%" << endl;
    Info << "solver_solve time : " << solver_solve_time << ", " << solver_solve_time / solve_time * 100 << "%" << endl;
    Info << "correctBoundaryConditions time : " << correctBoundaryConditions_time << ", " << correctBoundaryConditions_time / solve_time * 100 << "%" << endl;
    Info << "misc time : " << misc_time << ", " << misc_time / solve_time * 100 << "%" << endl;
    Info << "----------------------------------------------------------------------------------------" << endl;
    return solverPerfVec;
}

template<>
solverPerformance dfMatrix::solve
(
    GeometricField<scalar, fvPatchField, volMesh>& psi,
    const Field<scalar>& source,
    const FieldField<Field, scalar>& internalCoeffs,
    const FieldField<Field, scalar>& boundaryCoeffs,
    const dictionary& solverControls
)
{
    double misc_time = 0.;
    double diag_copy_time = 0.;
    double addBoundaryDiag_time = 0.;
    double addBoundarySource_time = 0.;
    double build_solver_time = 0.;
    double solver_solve_time = 0.;
    double correctBoundaryConditions_time = 0.;

    clockTime solveClock;

    scalarField saveDiag(diag());

    diag_copy_time += solveClock.timeIncrement();

    // addBoundaryDiag(diag(), internalCoeffs, 0);
    addBoundaryDiag(diag(), internalCoeffs, 0);

    addBoundaryDiag_time += solveClock.timeIncrement();

    const_cast<scalarField&>(ldu().diag()) = diag();

    diag_copy_time += solveClock.timeIncrement();

    scalarField sourceCpy(source);

    diag_copy_time += solveClock.timeIncrement();

    addBoundarySource(sourceCpy, psi, boundaryCoeffs, false);

    addBoundarySource_time += solveClock.timeIncrement();

    Foam::autoPtr<Foam::dfMatrix::solver> solver = dfMatrix::solver::New
    (
        psi.name(),
        *this,
        boundaryCoeffs,
        internalCoeffs,
        psi.boundaryField().scalarInterfaces(),
        solverControls
    );

    build_solver_time += solveClock.timeIncrement();
    
    // Solver call
    solverPerformance solverPerf = solver->solve(psi.primitiveFieldRef(), sourceCpy);

    solver_solve_time += solveClock.timeIncrement();

    if (solverPerformance::debug)
    {
        solverPerf.print(Info.masterStream(mesh().comm()));
    }

    diag() = saveDiag;
    const_cast<scalarField&>(ldu().diag()) = saveDiag;

    diag_copy_time += solveClock.timeIncrement();

    psi.correctBoundaryConditions();

    correctBoundaryConditions_time += solveClock.timeIncrement();

    Residuals<scalar>::append(psi.mesh(), solverPerf);

    misc_time += solveClock.timeIncrement();
    double solve_time = solveClock.elapsedTime();

    // print time
    Info << "dfMatrix::solve profiling -----------------------------------------------------------------" << endl;
    Info << "solve time : " << solve_time << endl;
    Info << "diag_copy time : " << diag_copy_time << ", " << diag_copy_time / solve_time * 100 << "%" << endl;
    Info << "addBoundaryDiag time : " << addBoundaryDiag_time << ", " << addBoundaryDiag_time / solve_time * 100 << "%" << endl;
    Info << "addBoundarySource time : " << addBoundarySource_time << ", " << addBoundarySource_time / solve_time * 100 << "%" << endl;
    Info << "build_solver time : " << build_solver_time << ", " << build_solver_time / solve_time * 100 << "%" << endl;
    Info << "solver_solve time : " << solver_solve_time << ", " << solver_solve_time / solve_time * 100 << "%" << endl;
    Info << "correctBoundaryConditions time : " << correctBoundaryConditions_time << ", " << correctBoundaryConditions_time / solve_time * 100 << "%" << endl;
    Info << "misc time : " << misc_time << ", " << misc_time / solve_time * 100 << "%" << endl;
    Info << "----------------------------------------------------------------------------------------" << endl;

    return solverPerf;
}

template<class Type>
SolverPerformance<Type> dfMatrix::solve(
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
SolverPerformance<Type> dfMatrix::solve(
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
void dfMatrix::addBoundarySource
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
void dfMatrix::addBoundaryDiag
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
void dfMatrix::addToInternalField
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
void dfMatrix::addToInternalField
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
void dfMatrix::addToInternalField<scalar>
(
    const labelUList& addr,
    const Field<scalar>& pf,
    Field<scalar>& intf
) const;

template
void dfMatrix::addToInternalField<vector>
(
    const labelUList& addr,
    const Field<vector>& pf,
    Field<vector>& intf
) const;

template
void dfMatrix::addBoundarySource<scalar>
(
    Field<scalar>& source,
    const GeometricField<scalar, fvPatchField, volMesh>& psi,
    const FieldField<Field, scalar>& boundaryCoeffs,
    const bool couples
) const;

template
void dfMatrix::addBoundarySource<vector>
(
    Field<vector>& source,
    const GeometricField<vector, fvPatchField, volMesh>& psi,
    const FieldField<Field, vector>& boundaryCoeffs,
    const bool couples
) const;

template
void dfMatrix::addBoundaryDiag<scalar>
(
    scalarField& diag,
    const FieldField<Field, scalar>& internalCoeffs,
    const direction solveCmpt
) const;

template
void dfMatrix::addBoundaryDiag<vector>
(
    scalarField& diag,
    const FieldField<Field, vector>& internalCoeffs,
    const direction solveCmpt
) const;

template
SolverPerformance<vector> dfMatrix::solve<vector>(
    GeometricField<vector, fvPatchField, volMesh>& psi,
    const Field<vector>& source,
    const FieldField<Field, vector>& internalCoeffs,
    const FieldField<Field, vector>& boundaryCoeffs,
    const dictionary& solverControls
);

template
SolverPerformance<scalar> dfMatrix::solve<scalar>(
    GeometricField<scalar, fvPatchField, volMesh>& psi,
    const Field<scalar>& source,
    const FieldField<Field, scalar>& internalCoeffs,
    const FieldField<Field, scalar>& boundaryCoeffs
);

template
SolverPerformance<vector> dfMatrix::solve<vector>(
    GeometricField<vector, fvPatchField, volMesh>& psi,
    const Field<vector>& source,
    const FieldField<Field, vector>& internalCoeffs,
    const FieldField<Field, vector>& boundaryCoeffs
);

template
SolverPerformance<scalar> dfMatrix::solve<scalar>(
    GeometricField<scalar, fvPatchField, volMesh>& psi,
    const Field<scalar>& source,
    const FieldField<Field, scalar>& internalCoeffs,
    const FieldField<Field, scalar>& boundaryCoeffs,
    const word& name
);

template
SolverPerformance<vector> dfMatrix::solve<vector>(
    GeometricField<vector, fvPatchField, volMesh>& psi,
    const Field<vector>& source,
    const FieldField<Field, vector>& internalCoeffs,
    const FieldField<Field, vector>& boundaryCoeffs,
    const word& name
);




}