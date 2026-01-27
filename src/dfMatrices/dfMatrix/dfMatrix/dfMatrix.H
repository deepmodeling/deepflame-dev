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
#include <cassert>

#include "lduMatrix.H"
#include "dfInnerMatrix.H"
#include <memory>

namespace Foam{


class dfMatrix{
private:
    // const fvMesh& mesh_;
    const lduMatrix* lduMatrixPtr_;

    std::shared_ptr<dfInnerMatrix> innerMatrixPtr_;

    static InnerMatrixFormat getInnerMatrixTypeFromEnv();

public:
    // Static data
    // Declare name of the class and its debug switch
    ClassName("dfMatrix");


    dfMatrix(const lduMatrix& lduMatrix);

    dfMatrix(const lduMatrix& lduMatrix, const labelList& regionPtr);

    // dfMatrix(const lduMatrix& courseLduMatrix, const labelList& fineRowBlockPtr, const labelList& fineToCoarse);

    dfMatrix(const lduMesh& mesh);

    dfMatrix(const lduMesh& lduMesh, const labelList& regionPtr);

    // dfMatrix(const lduMesh& courseLduMesh, const labelList& fineRowBlockPtr, const labelList& fineToCoarse);

    void valueCopy(lduMatrix& ldu);

    // Copy constructor
    dfMatrix(const dfMatrix& other):lduMatrixPtr_(other.lduMatrixPtr_),innerMatrixPtr_(other.innerMatrixPtr_){}

    //- Destructor
    ~dfMatrix(){};

    const lduMatrix& ldu() const {assert(lduMatrixPtr_ != nullptr); return *lduMatrixPtr_;}

    //- Return the df mesh from which the addressing is obtained
    const lduMesh& mesh() const {assert(lduMatrixPtr_ != nullptr); return lduMatrixPtr_->mesh();}
    //- Return the df addressing
    const lduAddressing& lduAddr() const {assert(lduMatrixPtr_ != nullptr); return lduMatrixPtr_->lduAddr();}

    //- Return the patch evaluation schedule
    const lduSchedule& patchSchedule() const {return lduAddr().patchSchedule();}

    // Access to coefficients
    const dfInnerMatrix& innerMatrix() const {return *innerMatrixPtr_;};

    scalarField& diag() {return innerMatrixPtr_->diag();};

    const scalarField& diag() const {return innerMatrixPtr_->diag();};

    bool diagonal() const
    {
        return innerMatrixPtr_->diagonal();
    }

    bool symmetric() const
    {
        return innerMatrixPtr_->symmetric();
    }

    bool asymmetric() const
    {
        return innerMatrixPtr_->asymmetric();
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

    InnerMatrixFormat getFormat() const {return innerMatrixPtr_->getFormat();};
    const std::string getFormatName() const {return innerMatrixPtr_->getFormatName();};

    void SpMV
    (
        scalarField& Apsi,
        const scalarField& psi
    ) const { innerMatrixPtr_->SpMV(Apsi.begin(), psi.begin()); };

    void SumA(scalarField& sumA) const {
        innerMatrixPtr_->SumA(sumA.begin());
    };

    void GaussSeidel
    (
        scalarField& psi,
        scalarField& bPrime
    ) const { innerMatrixPtr_->GaussSeidel(psi.begin(), bPrime.begin()); };

    void Jacobi
    (
        scalarField& psi,
        scalarField& bPrime
    ) const { innerMatrixPtr_->Jacobi(psi.begin(), bPrime.begin()); };

    void calcDILUReciprocalD(
        scalarField& rD
    ) const { innerMatrixPtr_->calcDILUReciprocalD(rD.begin()); };

    void DILUPrecondition(
        scalarField& wA,
        const scalarField& rA,
        const scalarField& rD
    ) const { innerMatrixPtr_->DILUPrecondition(wA.begin(), rA.begin(), rD.begin()); };

    void DILUPreconditionT(
        scalarField& wT,
        const scalarField& rT,
        const scalarField& rD
    ) const { innerMatrixPtr_->DILUPreconditionT(wT.begin(), rT.begin(), rD.begin()); };

    void calcDICReciprocalD(
        scalarField& rD
    ) const { innerMatrixPtr_->calcDICReciprocalD(rD.begin()); };

    void DICPrecondition(
        scalarField& wA,
        const scalarField& rA,
        const scalarField& rD
    ) const { innerMatrixPtr_->DICPrecondition(wA.begin(), rA.begin(), rD.begin()); };


    void SymGaussSeidel
    (
        scalarField& psi,
        scalarField& bPrime
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
        const dfMatrix& matrix_;
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
                const dfMatrix& matrix,
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
                const dfMatrix& matrix,
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
            const dfMatrix& matrix,
            const FieldField<Field, scalar>& interfaceBouCoeffs,
            const FieldField<Field, scalar>& interfaceIntCoeffs,
            const lduInterfaceFieldPtrsList& interfaces,
            const dictionary& solverControls
        );

        //- Return a new solver
        static autoPtr<solver> New
        (
            const word& fieldName,
            const dfMatrix& matrix,
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
        const dfMatrix& matrix() const {return matrix_;}
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

    //- Abstract base-class for dfMatrix smoothers
    class smoother
    {
    protected:
        // Protected data
        word fieldName_;
        const dfMatrix& matrix_;
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
                const dfMatrix& matrix,
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
                const dfMatrix& matrix,
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
            const dfMatrix& matrix,
            const FieldField<Field, scalar>& interfaceBouCoeffs,
            const FieldField<Field, scalar>& interfaceIntCoeffs,
            const lduInterfaceFieldPtrsList& interfaces
        );


        // Selectors

        //- Return a new smoother
        static autoPtr<smoother> New
        (
            const word& fieldName,
            const dfMatrix& matrix,
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

        const dfMatrix& matrix() const
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


    //- Abstract base-class for dfMatrix preconditioners
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

