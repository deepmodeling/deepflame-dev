/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2019 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::dfGAMGSolver

Description
    Geometric agglomerated algebraic multigrid solver.

  Characteristics:
      - Requires positive definite, diagonally dominant matrix.
      - Agglomeration algorithm: selectable and optionally cached.
      - Restriction operator: summation.
      - Prolongation operator: injection.
      - Smoother: Gauss-Seidel.
      - Coarse matrix creation: central coefficient: summation of fine grid
        central coefficients with the removal of intra-cluster face;
        off-diagonal coefficient: summation of off-diagonal faces.
      - Coarse matrix scaling: performed by correction scaling, using steepest
        descent optimisation.
      - Type of cycle: V-cycle with optional pre-smoothing.
      - Coarsest-level matrix solved using PCG or PBiCGStab.

SourceFiles
    dfGAMGSolver.C
    dfGAMGSolverAgglomerateMatrix.C
    dfGAMGSolverInterpolate.C
    dfGAMGSolverScale.C
    dfGAMGSolverSolve.C

\*---------------------------------------------------------------------------*/

#ifndef dfGAMGSolver_H
#define dfGAMGSolver_H

#include "dfGAMGAgglomeration.H"
#include "lduMatrix.H"
#include "dfMatrix.H"
#include "labelField.H"
#include "primitiveFields.H"
#include "LUscalarMatrix.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                           Class dfGAMGSolver Declaration
\*---------------------------------------------------------------------------*/

class dfGAMGSolver
:
    public dfMatrix::solver
{
    // Private Data

        bool cacheAgglomeration_;

        //- Number of pre-smoothing sweeps
        label nPreSweeps_;

        //- Lever multiplier for the number of pre-smoothing sweeps
        label preSweepsLevelMultiplier_;

        //- Maximum number of pre-smoothing sweeps
        label maxPreSweeps_;

        //- Number of post-smoothing sweeps
        label nPostSweeps_;

        //- Lever multiplier for the number of post-smoothing sweeps
        label postSweepsLevelMultiplier_;

        //- Maximum number of post-smoothing sweeps
        label maxPostSweeps_;

        //- Number of smoothing sweeps on finest mesh
        label nFinestSweeps_;

        //- Number of smoothing sweeps on coarsest mesh
        label nCoarsestSweeps_;
        
        //- Choose if the corrections should be interpolated after injection.
        //  By default corrections are not interpolated.
        bool interpolateCorrection_;

        //- Choose if the corrections should be scaled.
        //  By default corrections for symmetric matrices are scaled
        //  but not for asymmetric matrices.
        bool scaleCorrection_;

        //- solve the coarsest level
        bool solveCoarsest_;

        //- Direct or iteratively solve the coarsest level
        bool directSolveCoarsest_;

        //- The agglomeration
        const dfGAMGAgglomeration& agglomeration_;

        //- Hierarchy of matrix levels
        PtrList<lduMatrix> matrixLevels_;


        //- Hierarchy of interfaces.
        PtrList<PtrList<lduInterfaceField>> primitiveInterfaceLevels_;

        //- Hierarchy of interfaces in lduInterfaceFieldPtrs form
        PtrList<lduInterfaceFieldPtrsList> interfaceLevels_;

        //- Hierarchy of interface boundary coefficients
        PtrList<FieldField<Field, scalar>> interfaceLevelsBouCoeffs_;

        //- Hierarchy of interface internal coefficients
        PtrList<FieldField<Field, scalar>> interfaceLevelsIntCoeffs_;

        //- LU decompsed coarsest matrix
        autoPtr<LUscalarMatrix> coarsestLUMatrixPtr_;

        // Private Member Functions

        //- Simplified access to interface level
        const lduInterfaceFieldPtrsList& interfaceLevel
        (
            const label i
        ) const;

        //- Simplified access to matrix level
        const lduMatrix& matrixLevel(const label i) const;

        const dfMatrix& dfMatrixLevel(const label i) const;


        //- Simplified access to interface boundary coeffs level
        const FieldField<Field, scalar>& interfaceBouCoeffsLevel
        (
            const label i
        ) const;

        //- Simplified access to interface internal coeffs level
        const FieldField<Field, scalar>& interfaceIntCoeffsLevel
        (
            const label i
        ) const;

        //- Agglomerate coarse matrix. Supply mesh to use - so we can
        //  construct temporary matrix on the fine mesh (instead of the coarse
        //  mesh)
        void agglomerateMatrix
        (
            const label fineLevelIndex,
            const lduMesh& coarseMesh,
            const lduInterfacePtrsList& coarseMeshInterfaces
        );

        //- Agglomerate coarse interface coefficients
        void agglomerateInterfaceCoefficients
        (
            const label fineLevelIndex,
            const lduInterfacePtrsList& coarseMeshInterfaces,
            PtrList<lduInterfaceField>& coarsePrimInterfaces,
            lduInterfaceFieldPtrsList& coarseInterfaces,
            FieldField<Field, scalar>& coarseInterfaceBouCoeffs,
            FieldField<Field, scalar>& coarseInterfaceIntCoeffs
        ) const;

        //- Collect matrices from other processors
        void gatherMatrices
        (
            const labelList& procIDs,
            const lduMesh& dummyMesh,
            const label meshComm,

            const lduMatrix& mat,
            const FieldField<Field, scalar>& interfaceBouCoeffs,
            const FieldField<Field, scalar>& interfaceIntCoeffs,
            const lduInterfaceFieldPtrsList& interfaces,

            PtrList<lduMatrix>& otherMats,
            PtrList<FieldField<Field, scalar>>& otherBouCoeffs,
            PtrList<FieldField<Field, scalar>>& otherIntCoeffs,
            List<boolList>& otherTransforms,
            List<List<label>>& otherRanks
        ) const;

        //- Agglomerate processor matrices
        void procAgglomerateMatrix
        (
            // Agglomeration information
            const labelList& procAgglomMap,
            const List<label>& agglomProcIDs,

            const label levelI,

            // Resulting matrix
            autoPtr<lduMatrix>& allMatrixPtr,
            FieldField<Field, scalar>& allInterfaceBouCoeffs,
            FieldField<Field, scalar>& allInterfaceIntCoeffs,
            PtrList<lduInterfaceField>& allPrimitiveInterfaces,
            lduInterfaceFieldPtrsList& allInterfaces
        ) const;

        //- Agglomerate processor matrices
        void procAgglomerateMatrix
        (
            const labelList& procAgglomMap,
            const List<label>& agglomProcIDs,
            const label levelI
        );

        //- Interpolate the correction after injected prolongation
        void interpolate
        (
            scalarField& psi,
            scalarField& Apsi,
            const dfMatrix& m,
            const FieldField<Field, scalar>& interfaceBouCoeffs,
            const lduInterfaceFieldPtrsList& interfaces,
            const direction cmpt
        ) const;

        //- Interpolate the correction after injected prolongation and
        //  re-normalise
        void interpolate
        (
            scalarField& psi,
            scalarField& Apsi,
            const dfMatrix& m,
            const FieldField<Field, scalar>& interfaceBouCoeffs,
            const lduInterfaceFieldPtrsList& interfaces,
            const labelList& restrictAddressing,
            const scalarField& psiC,
            const direction cmpt
        ) const;

        //- Calculate and apply the scaling factor from Acf, coarseSource
        //  and coarseField.
        //  At the same time do a Jacobi iteration on the coarseField using
        //  the Acf provided after the coarseField values are used for the
        //  scaling factor.

        void scale
        (
            scalarField& field,
            scalarField& Acf,
            const dfMatrix& A,
            const FieldField<Field, scalar>& interfaceLevelBouCoeffs,
            const lduInterfaceFieldPtrsList& interfaceLevel,
            const scalarField& source,
            const direction cmpt
        ) const;


        //- Create and return the dictionary to specify the PCG solver
        //  to solve the coarsest level
        dictionary PCGsolverDict
        (
            const scalar tol,
            const scalar relTol
        ) const;

        //- Create and return the dictionary to specify the PBiCGStab solver
        //  to solve the coarsest level
        dictionary PBiCGStabSolverDict
        (
            const scalar tol,
            const scalar relTol
        ) const;

        //- Solve the coarsest level with either an iterative or direct solver
        void solveCoarsestLevel
        (
            scalarField& coarsestCorrField,
            const scalarField& coarsestSource
        ) const;

protected:

        PtrList<dfMatrix> dfMatrixLevels_;

        //- Read control parameters from the control dictionary
        virtual void readControls();

        //- Initialise the data structures for the V-cycle
        void initVcycle
        (
            PtrList<scalarField>& coarseCorrFields,
            PtrList<scalarField>& coarseSources,
            PtrList<dfMatrix::smoother>& smoothers,
            scalarField& scratch1,
            scalarField& scratch2
        ) const;

        //- Perform a single GAMG V-cycle with pre, post and finest smoothing.
        void Vcycle
        (
            const PtrList<dfMatrix::smoother>& smoothers,
            scalarField& psi,
            const scalarField& source,
            scalarField& Apsi,
            scalarField& finestCorrection,
            scalarField& finestResidual,

            scalarField& scratch1,
            scalarField& scratch2,

            PtrList<scalarField>& coarseCorrFields,
            PtrList<scalarField>& coarseSources,
            const direction cmpt=0
        ) const;

public:

    friend class GAMGPreconditioner;

    //- Runtime type information
    TypeName("GAMG");


    // Constructors

        //- Construct from lduMatrix and solver controls
        dfGAMGSolver
        (
            const word& fieldName,
            const dfMatrix& matrix,
            const FieldField<Field, scalar>& interfaceBouCoeffs,
            const FieldField<Field, scalar>& interfaceIntCoeffs,
            const lduInterfaceFieldPtrsList& interfaces,
            const dictionary& solverControls
        );


    //- Destructor
    virtual ~dfGAMGSolver();


    // Member Functions

        //- Solve
        virtual solverPerformance solve
        (
            scalarField& psi,
            const scalarField& source,
            const direction cmpt=0
        ) const;



        mutable double solve_time = 0;
        mutable double spmv_time = 0.;
        mutable double normFactor_time = 0.;
        mutable double initVcycle_time = 0.;
        mutable double Vcycle_time = 0.;
        mutable double vector_update_time = 0.;
        mutable double gSumMag_time = 0.;
        mutable double allreduce_time = 0.;
        mutable double misc_time = 0.;


        mutable double Vcycle_total_time = 0.;
        mutable double smooth_time = 0.;
        mutable double scale_time = 0.;
        mutable double interpolate_time = 0.;
        mutable double restrictField_time = 0.;
        mutable double prolongField_time = 0.;
        mutable double solveCoarsest_time = 0.;
        mutable double Vcycle_vector_update_time = 0.;
        mutable double Vcycle_spmv_time = 0.;
        mutable double Vcycle_misc_time = 0.;


        void print_time() const {
            // double Vcycle_other_time = Vcycle_total_time - smooth_time - scale_time - interpolate_time - restrictField_time - prolongField_time - solveCoarsest_time - Vcycle_vector_update_time - Vcycle_spmv_time - Vcycle_misc_time;
            Info << "dfGAMG Vcycle solve profiling -----------------------------------------------------------------" << endl;
            Info << "Vcycle_total time : " << Vcycle_total_time << endl;
            Info << "smooth time : " << smooth_time << ", " << smooth_time / Vcycle_total_time * 100 << "%" << endl;
            Info << "scale time : " << scale_time << ", " << scale_time / Vcycle_total_time * 100 << "%" << endl;
            Info << "interpolate time : " << interpolate_time << ", " << interpolate_time / Vcycle_total_time * 100 << "%" << endl;
            Info << "restrictField time : " << restrictField_time << ", " << restrictField_time / Vcycle_total_time * 100 << "%" << endl;
            Info << "prolongField time : " << prolongField_time << ", " << prolongField_time / Vcycle_total_time * 100 << "%" << endl;
            Info << "solveCoarsest time : " << solveCoarsest_time << ", " << solveCoarsest_time / Vcycle_total_time * 100 << "%" << endl;
            Info << "Vcycle_vector_update time : " << Vcycle_vector_update_time << ", " << Vcycle_vector_update_time / Vcycle_total_time * 100 << "%" << endl;
            Info << "Vcycle_spmv time : " << Vcycle_spmv_time << ", " << Vcycle_spmv_time / Vcycle_total_time * 100 << "%" << endl;
            Info << "Vcycle_misc time : " << Vcycle_misc_time << ", " << Vcycle_misc_time / Vcycle_total_time * 100 << "%" << endl;
            // Info << "Vcycle_other time : " << Vcycle_other_time << ", " << Vcycle_other_time / Vcycle_total_time * 100 << "%" << endl;
            Info << "----------------------------------------------------------------------------------------" << endl; 
            // double other_time = solve_time - spmv_time - normFactor_time - initVcycle_time - Vcycle_time - vector_update_time - gSumMag_time - misc_time - allreduce_time;
            Info << "dfGAMG solve profiling -----------------------------------------------------------------" << endl;
            Info << "solve time : " << solve_time << endl;
            Info << "spmv time : " << spmv_time << ", " << spmv_time / solve_time * 100 << "%" << endl;
            Info << "normFactor time : " << normFactor_time << ", " << normFactor_time / solve_time * 100 << "%" << endl;
            Info << "initVcycle time : " << initVcycle_time << ", " << initVcycle_time / solve_time * 100 << "%" << endl;
            Info << "Vcycle time : " << Vcycle_time << ", " << Vcycle_time / solve_time * 100 << "%" << endl;
            Info << "vector_update time : " << vector_update_time << ", " << vector_update_time / solve_time * 100 << "%" << endl;
            Info << "gSumMag time : " << gSumMag_time << ", " << gSumMag_time / solve_time * 100 << "%" << endl;
            Info << "misc time : " << misc_time << ", " << misc_time / solve_time * 100 << "%" << endl;
            // Info << "other time : " << other_time << ", " << other_time / solve_time * 100 << "%" << endl;
            Info << "----------------------------------------------------------------------------------------" << endl;

        }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
