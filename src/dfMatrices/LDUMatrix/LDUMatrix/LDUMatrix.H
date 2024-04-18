#pragma once
#include <fvMesh.H>
#include <vector>
#include <string>
#include "autoPtr.H"
#include "primitiveFieldsFwd.H"
#include "FieldField.H"
#include "typeInfo.H"
#include "lduInterfaceFieldPtrsList.H"
#include "runTimeSelectionTables.H"
#include "solverPerformance.H"
#include "InfoProxy.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "tmp.H"
#include "className.H"

#include "lduMatrix.H"


namespace Foam{

class LDUMatrix{
private:
    // const fvMesh& mesh_;
    const lduMatrix& lduMatrix_;

    //- Coefficients (not including interfaces)
    scalarField *lowerPtr_, *diagPtr_, *upperPtr_;

public:
    // Static data
    // Declare name of the class and its debug switch
    ClassName("LDUMatrix");


    LDUMatrix(const lduMatrix& lduMatrix);

    // Copy constructor
    LDUMatrix(const LDUMatrix& other) = delete;

    // Copy constructor
    // void init_value_from_lduMatrix(const lduMatrix& lduMatrix);

    //- Destructor
    ~LDUMatrix();

    const lduMatrix& ldu() const {return lduMatrix_;}

    //- Return the LDU mesh from which the addressing is obtained
    const lduMesh& mesh() const {return lduMatrix_.mesh();}
    //- Return the LDU addressing
    const lduAddressing& lduAddr() const {return lduMatrix_.lduAddr();}

    //- Return the patch evaluation schedule
    const lduSchedule& patchSchedule() const {return lduAddr().patchSchedule();}

    // Access to coefficients

    scalarField& lower();
    scalarField& diag();
    scalarField& upper();

    scalarField& lower(const label size);
    scalarField& diag(const label nCoeffs);
    scalarField& upper(const label nCoeffs);

    const scalarField& lower() const;
    const scalarField& diag() const;
    const scalarField& upper() const;

    bool hasDiag() const
    {
        return (diagPtr_);
    }

    bool hasUpper() const
    {
        return (upperPtr_);
    }

    bool hasLower() const
    {
        return (lowerPtr_);
    }

    bool diagonal() const
    {
        return (diagPtr_ && !lowerPtr_ && !upperPtr_);
    }

    bool symmetric() const
    {
        return (diagPtr_ && (!lowerPtr_ && upperPtr_));
    }

    bool asymmetric() const
    {
        return (diagPtr_ && lowerPtr_ && upperPtr_);
    }

    mutable double Amul_time_ = 0.;
    mutable double Amul_Communication_init_time_ = 0.;
    mutable double Amul_Communication_update_time_ = 0.;
    mutable double Amul_Compute_time_ = 0.;
    mutable double Jacobi_time_ = 0.;

    void print_time() const {
        Info << "Amul time : " << Amul_time_ << endl; 
        Info << "Amul Communication init time : " << Amul_Communication_init_time_ << ", " << Amul_Communication_init_time_ / Amul_time_ * 100. << "%" << endl; 
        Info << "Amul Communication update time : " << Amul_Communication_update_time_ << ", " << Amul_Communication_update_time_ / Amul_time_ * 100. << "%" << endl; 
        Info << "Amul Compute time : " << Amul_Compute_time_ << ", " << Amul_Compute_time_ / Amul_time_ * 100. << "%" << endl; 
        Info << "Jocabi time : " << Jacobi_time_ << endl; 
    }

    //- Matrix multiplication with updated interfaces.
    void Amul
    (
        scalarField&,
        const tmp<scalarField>&,
        const FieldField<Field, scalar>&,
        const lduInterfaceFieldPtrsList&,
        const direction cmpt
    ) const;

    void Amul
    (
        scalarField&,
        const scalarField&,
        const FieldField<Field, scalar>&,
        const lduInterfaceFieldPtrsList&,
        const direction cmpt
    ) const;
    void SpMV
    (
        scalarField&,
        const scalarField&
    ) const;
    void SpMV_naive
    (
        scalarField&,
        const scalarField&
    ) const;
    void Jacobi
    (
        scalarField&,
        scalarField&
    ) const;
    void Jacobi_naive
    (
        scalarField&,
        scalarField&
    ) const;
    void SymGaussSeidel
    (
        scalarField&,
        scalarField&
    ) const;

    void SymGaussSeidel_naive
    (
        scalarField&,
        scalarField&
    ) const;
    
    //- Sum the coefficients on each row of the matrix
    void sumA
    (
        scalarField&,
        const FieldField<Field, scalar>&,
        const lduInterfaceFieldPtrsList&
    ) const;


    void residual
    (
        scalarField& rA,
        const scalarField& psi,
        const scalarField& source,
        const FieldField<Field, scalar>& interfaceBouCoeffs,
        const lduInterfaceFieldPtrsList& interfaces,
        const direction cmpt
    ) const;

    tmp<scalarField> residual
    (
        const scalarField& psi,
        const scalarField& source,
        const FieldField<Field, scalar>& interfaceBouCoeffs,
        const lduInterfaceFieldPtrsList& interfaces,
        const direction cmpt
    ) const;


    //- Initialise the update of interfaced interfaces
    //  for matrix operations
    void initMatrixInterfaces
    (
        const FieldField<Field, scalar>& interfaceCoeffs,
        const lduInterfaceFieldPtrsList& interfaces,
        const scalarField& psiif,
        scalarField& result,
        const direction cmpt
    ) const;

    //- Update interfaced interfaces for matrix operations
    void updateMatrixInterfaces
    (
        const FieldField<Field, scalar>& interfaceCoeffs,
        const lduInterfaceFieldPtrsList& interfaces,
        const scalarField& psiif,
        scalarField& result,
        const direction cmpt
    ) const;


    template<class Type>
    SolverPerformance<Type> solve(
        GeometricField<Type, fvPatchField, volMesh>& psi,
        const Field<Type>& source,
        const FieldField<Field, Type>& internalCoeffs,
        const FieldField<Field, Type>& boundaryCoeffs
    );

    template<class Type>
    SolverPerformance<Type> solve(
        GeometricField<Type, fvPatchField, volMesh>& psi,
        const Field<Type>& source,
        const FieldField<Field, Type>& internalCoeffs,
        const FieldField<Field, Type>& boundaryCoeffs,
        const word& name
    );

    template<class Type>
    SolverPerformance<Type> solve(
        GeometricField<Type, fvPatchField, volMesh>& psi,
        const Field<Type>& source,
        const FieldField<Field, Type>& internalCoeffs,
        const FieldField<Field, Type>& boundaryCoeffs,
        const dictionary& solverControls
    );


    template<class Type2>
    void addToInternalField
    (
        const labelUList& addr,
        const Field<Type2>& pf,
        Field<Type2>& intf
    ) const;

    template<class Type2>
    void addToInternalField
    (
        const labelUList& addr,
        const tmp<Field<Type2>>& tpf,
        Field<Type2>& intf
    ) const;

    template<class Type>
    void addBoundarySource
    (
        Field<Type>& source,
        const GeometricField<Type, fvPatchField, volMesh>& psi,
        const FieldField<Field, Type>& boundaryCoeffs,
        const bool couples=true
    ) const;

    template<class Type>
    void addBoundaryDiag
    (
        scalarField& diag,
        const FieldField<Field, Type>& internalCoeffs,
        const direction solveCmpt
    ) const;

    class solver{
    protected:
        word fieldName_;
        const LDUMatrix& matrix_;
        const FieldField<Field, scalar>& interfaceBouCoeffs_;
        const FieldField<Field, scalar>& interfaceIntCoeffs_;
        lduInterfaceFieldPtrsList interfaces_;

        //- Dictionary of controls
        dictionary controlDict_;
        //- Default maximum number of iterations in the solver
        static const label defaultMaxIter_;
        //- Maximum number of iterations in the solver
        label maxIter_;
        //- Minimum number of iterations in the solver
        label minIter_;
        //- Final convergence tolerance
        scalar tolerance_;
        //- Convergence tolerance relative to the initial
        scalar relTol_;
        //- Read the control parameters from the controlDict_
        virtual void readControls();
    public:
        //- Runtime type information
        virtual const word& type() const = 0;
        declareRunTimeSelectionTable
        (
            autoPtr,
            solver,
            symMatrix,
            (
                const word& fieldName,
                const LDUMatrix& matrix,
                const FieldField<Field, scalar>& interfaceBouCoeffs,
                const FieldField<Field, scalar>& interfaceIntCoeffs,
                const lduInterfaceFieldPtrsList& interfaces,
                const dictionary& solverControls
            ),
            (
                fieldName,
                matrix,
                interfaceBouCoeffs,
                interfaceIntCoeffs,
                interfaces,
                solverControls
            )
        );
        declareRunTimeSelectionTable
        (
            autoPtr,
            solver,
            asymMatrix,
            (
                const word& fieldName,
                const LDUMatrix& matrix,
                const FieldField<Field, scalar>& interfaceBouCoeffs,
                const FieldField<Field, scalar>& interfaceIntCoeffs,
                const lduInterfaceFieldPtrsList& interfaces,
                const dictionary& solverControls
            ),
            (
                fieldName,
                matrix,
                interfaceBouCoeffs,
                interfaceIntCoeffs,
                interfaces,
                solverControls
            )
        );

        solver(
            const word& fieldName,
            const LDUMatrix& matrix,
            const FieldField<Field, scalar>& interfaceBouCoeffs,
            const FieldField<Field, scalar>& interfaceIntCoeffs,
            const lduInterfaceFieldPtrsList& interfaces,
            const dictionary& solverControls
        );

        //- Return a new solver
        static autoPtr<solver> New
        (
            const word& fieldName,
            const LDUMatrix& matrix,
            const FieldField<Field, scalar>& interfaceBouCoeffs,
            const FieldField<Field, scalar>& interfaceIntCoeffs,
            const lduInterfaceFieldPtrsList& interfaces,
            const dictionary& solverControls
        );

        //- Destructor
        virtual ~solver() {}

        // Member Functions
        // Access
        const word& fieldName() const {return fieldName_;}
        const LDUMatrix& matrix() const {return matrix_;}
        const FieldField<Field, scalar>& interfaceBouCoeffs() const {return interfaceBouCoeffs_;}
        const FieldField<Field, scalar>& interfaceIntCoeffs() const {return interfaceIntCoeffs_;}
        const lduInterfaceFieldPtrsList& interfaces() const{return interfaces_;}

        //- Read and reset the solver parameters from the given stream
        virtual void read(const dictionary&);
        virtual solverPerformance solve
        (
            scalarField& psi,
            const scalarField& source,
            const direction cmpt=0
        ) const = 0;

        //- Return the matrix norm used to normalise the residual for the
        //  stopping criterion
        scalar normFactor
        (
            const scalarField& psi,
            const scalarField& source,
            const scalarField& Apsi,
            scalarField& tmpField
        ) const;

        virtual void print_time() const {}

    };

    //- Abstract base-class for LDUMatrix smoothers
    class smoother
    {
    protected:
        // Protected data
        word fieldName_;
        const LDUMatrix& matrix_;
        const FieldField<Field, scalar>& interfaceBouCoeffs_;
        const FieldField<Field, scalar>& interfaceIntCoeffs_;
        const lduInterfaceFieldPtrsList& interfaces_;

    public:
        //- Find the smoother name (directly or from a sub-dictionary)
        static word getName(const dictionary&);
        //- Runtime type information
        virtual const word& type() const = 0;
        // Declare run-time constructor selection tables
        declareRunTimeSelectionTable
        (
            autoPtr,
            smoother,
            symMatrix,
            (
                const word& fieldName,
                const LDUMatrix& matrix,
                const FieldField<Field, scalar>& interfaceBouCoeffs,
                const FieldField<Field, scalar>& interfaceIntCoeffs,
                const lduInterfaceFieldPtrsList& interfaces
            ),
            (
                fieldName,
                matrix,
                interfaceBouCoeffs,
                interfaceIntCoeffs,
                interfaces
            )
        );

        declareRunTimeSelectionTable
        (
            autoPtr,
            smoother,
            asymMatrix,
            (
                const word& fieldName,
                const LDUMatrix& matrix,
                const FieldField<Field, scalar>& interfaceBouCoeffs,
                const FieldField<Field, scalar>& interfaceIntCoeffs,
                const lduInterfaceFieldPtrsList& interfaces
            ),
            (
                fieldName,
                matrix,
                interfaceBouCoeffs,
                interfaceIntCoeffs,
                interfaces
            )
        );


        // Constructors
        smoother
        (
            const word& fieldName,
            const LDUMatrix& matrix,
            const FieldField<Field, scalar>& interfaceBouCoeffs,
            const FieldField<Field, scalar>& interfaceIntCoeffs,
            const lduInterfaceFieldPtrsList& interfaces
        );


        // Selectors

        //- Return a new smoother
        static autoPtr<smoother> New
        (
            const word& fieldName,
            const LDUMatrix& matrix,
            const FieldField<Field, scalar>& interfaceBouCoeffs,
            const FieldField<Field, scalar>& interfaceIntCoeffs,
            const lduInterfaceFieldPtrsList& interfaces,
            const dictionary& solverControls
        );


        //- Destructor
        virtual ~smoother()
        {}


        // Member Functions

        // Access

        const word& fieldName() const
        {
            return fieldName_;
        }

        const LDUMatrix& matrix() const
        {
            return matrix_;
        }

        const FieldField<Field, scalar>& interfaceBouCoeffs() const
        {
            return interfaceBouCoeffs_;
        }

        const FieldField<Field, scalar>& interfaceIntCoeffs() const
        {
            return interfaceIntCoeffs_;
        }

        const lduInterfaceFieldPtrsList& interfaces() const
        {
            return interfaces_;
        }


        //- Smooth the solution for a given number of sweeps
        virtual void smooth
        (
            scalarField& psi,
            const scalarField& source,
            const direction cmpt,
            const label nSweeps
        ) const = 0;

        virtual void print_time() const {}

    };


    //- Abstract base-class for LDUMatrix preconditioners
    class preconditioner
    {
    protected:

        // Protected data

        //- Reference to the base-solver this preconditioner is used with
        const solver& solver_;


    public:

        //- Find the preconditioner name (directly or from a sub-dictionary)
        static word getName(const dictionary&);

        //- Runtime type information
        virtual const word& type() const = 0;


        // Declare run-time constructor selection tables

        declareRunTimeSelectionTable
        (
            autoPtr,
            preconditioner,
            symMatrix,
            (
                const solver& sol,
                const dictionary& solverControls
            ),
            (sol, solverControls)
        );

        declareRunTimeSelectionTable
        (
            autoPtr,
            preconditioner,
            asymMatrix,
            (
                const solver& sol,
                const dictionary& solverControls
            ),
            (sol, solverControls)
        );


    // Constructors

        preconditioner
        (
            const solver& sol
        )
        :
            solver_(sol)
        {}


        // Selectors

        //- Return a new preconditioner
        static autoPtr<preconditioner> New
        (
            const solver& sol,
            const dictionary& solverControls
        );


        //- Destructor
        virtual ~preconditioner()
        {}


        // Member Functions

        //- Read and reset the preconditioner parameters
        //  from the given stream
        virtual void read(const dictionary&)
        {}

        //- Return wA the preconditioned form of residual rA
        virtual void precondition
        (
            scalarField& wA,
            const scalarField& rA,
            const direction cmpt=0
        ) const = 0;

        //- Return wT the transpose-matrix preconditioned form of
        //  residual rT.
        //  This is only required for preconditioning asymmetric matrices.
        virtual void preconditionT
        (
            scalarField& wT,
            const scalarField& rT,
            const direction cmpt=0
        ) const
        {
            NotImplemented;
        }

        virtual void print_time() const {}
    };

};

}

