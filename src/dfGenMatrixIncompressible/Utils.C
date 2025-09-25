#include "GenFvMatrix.H"


namespace Foam{

#define RELATIVE_ERROR_TOLERANCE 1e-9
#define ABSULTE_ERROR_TOLERANCE 1e-9
#define ZERO_TOLERANCE 1e-9

// void check_field_error(const Field<scalar>& answer, const Field<scalar>& check, const word& name){
//     if(answer.size() != check.size()){
//         SeriousError << "In check_field_error, two field have different size !!!" << endl;
//         MPI_Abort(PstreamGlobals::MPI_COMM_FOAM, -1);
//     }
//     // double max_absulte_error = 0.;
//     bool check_faild = false;
//     double max_relative_error = 0.;
//     double max_absulte_error = 0.; 
//     for(label i = 0; i < answer.size(); ++i){
//         double absulte_error = std::abs(answer[i] - check[i]);
//         double relative_error = std::abs(answer[i]) < ZERO_TOLERANCE ? absulte_error : (absulte_error / std::abs(answer[i]));
//         if(absulte_error > ABSULTE_ERROR_TOLERANCE && relative_error > RELATIVE_ERROR_TOLERANCE){
//             Info << name << " error : " << endl;
//             Info << "answer[i] : " << answer[i] << endl;
//             Info << "check[i] : " << check[i] << endl;
//             Info << "absulte_error : " << absulte_error << endl;
//             Info << "relative_error : " << relative_error << endl;
//             check_faild = true;
//         }
//         if(relative_error > max_relative_error){
//             max_absulte_error = absulte_error;
//             max_relative_error = relative_error;
//         }
//     }
//     if(check_faild){
//         MPI_Abort(PstreamGlobals::MPI_COMM_FOAM, -1);
//     }else{
//         Info << "error check pass : " << name << "-------------------------------------" << endl;
//         Info << "max_absulte_error : " << max_absulte_error << endl;
//         Info << "max_relative_error : " << max_relative_error << endl;
//         Info << "---------------------------------------------------------" << endl;
//     }
// }

void check_field_error(const Field<scalar>& answer, const Field<scalar>& check, const word& name){
    bool check_faild = false;
    double max_relative_error = 0.;
    double max_absulte_error = 0.;
    
    // 只有主核心执行检查逻辑
    if (Pstream::master())
    {
        if(answer.size() != check.size()){
            SeriousError << "In check_field_error, two field have different size !!!" << endl;
            check_faild = true;
        }
        else
        {
            for(label i = 0; i < answer.size(); ++i){
                double absulte_error = std::abs(answer[i] - check[i]);
                double relative_error = std::abs(answer[i]) < ZERO_TOLERANCE ? absulte_error : (absulte_error / std::abs(answer[i]));
                
                if(absulte_error > ABSULTE_ERROR_TOLERANCE && relative_error > RELATIVE_ERROR_TOLERANCE){
                    Info << name << " error at index " << i << " : " << endl;
                    Info << "answer[i] : " << answer[i] << endl;
                    Info << "check[i] : " << check[i] << endl;
                    Info << "absulte_error : " << absulte_error << endl;
                    Info << "relative_error : " << relative_error << endl;
                    check_faild = true;
                }
                
                if(relative_error > max_relative_error){
                    max_absulte_error = absulte_error;
                    max_relative_error = relative_error;
                }
            }
        }
        
        if(!check_faild){
            Info << "error check pass : " << name << "-------------------------------------" << endl;
            Info << "max_absulte_error : " << max_absulte_error << endl;
            Info << "max_relative_error : " << max_relative_error << endl;
            Info << "---------------------------------------------------------" << endl;
        }
    }
    
    // 将检查结果广播给所有进程
    Pstream::scatter(check_faild);
    
    // // 如果检查失败，所有进程一起终止
    // if(check_faild){
    //     MPI_Abort(PstreamGlobals::MPI_COMM_FOAM, -1);
    // }
    
    // 同步所有进程
    Pstream::scatter(max_absulte_error);
    Pstream::scatter(max_relative_error);
}

// void check_field_equal(const Field<scalar>& answer, const Field<scalar>& check, const word& name){
//     check_field_error(answer, check, name + " field");
//     return;
// }

// void check_field_boundary_equal(const volScalarField& answer, const volScalarField& check, const word& name){
//     check_field_error(answer, check, name + " field");
//     forAll(answer.boundaryField(), patchi)
//     {
//         check_field_error(answer.boundaryField()[patchi], check.boundaryField()[patchi], name + " boundaryField_" + std::to_string(patchi));
//     }
// }

void check_field_boundary_equal(const volVectorField& answer, const volVectorField& check, const word& name){
    for (direction cmpt=0; cmpt<vector::nComponents; ++cmpt)
    {
        check_field_error(answer.component(cmpt).ref(),check.component(cmpt).ref(), name + " internal_" + std::to_string(cmpt));
    }
    forAll(answer.boundaryField(), patchi)
    {
        for (direction cmpt=0; cmpt<vector::nComponents; ++cmpt)
        {
            check_field_error(answer.boundaryField()[patchi].component(cmpt).ref(),check.boundaryField()[patchi].component(cmpt).ref(), name + " boundary_" + std::to_string(cmpt));
        }
    }
}

void check_field_boundary_equal(const surfaceScalarField& answer, const surfaceScalarField& check, const word& name){
    check_field_error(answer, check, "field");
    forAll(answer.boundaryField(), patchi)
    {
        check_field_error(answer.boundaryField()[patchi], check.boundaryField()[patchi], name + " boundaryField_" + std::to_string(patchi));
    }
}

void check_fvmatrix_equal(const fvScalarMatrix& answer,const fvScalarMatrix& check, const word& name){
    // if(answer.source().begin() == check.source().begin()){
    //     SeriousError << "answer and check are the same matrix." << endl;
    //     MPI_Abort(PstreamGlobals::MPI_COMM_FOAM, -1);
    // } 

    check_field_error(answer.source(),check.source(), name + " source");
    check_field_error(answer.diag(),check.diag(), name + " diag");
    check_field_error(answer.lower(),check.lower(), name + " lower");
    check_field_error(answer.upper(),check.upper(), name + " upper");

    for(label patchi = 0; patchi < const_cast<fvScalarMatrix&>(answer).internalCoeffs().size(); ++patchi)
    {
        check_field_error(const_cast<fvScalarMatrix&>(answer).internalCoeffs()[patchi],const_cast<fvScalarMatrix&>(check).internalCoeffs()[patchi], name + " internalCoeffs_" + std::to_string(patchi));
        check_field_error(const_cast<fvScalarMatrix&>(answer).boundaryCoeffs()[patchi],const_cast<fvScalarMatrix&>(check).boundaryCoeffs()[patchi], name + " boundaryCoeffs_" + std::to_string(patchi));
    }
}

// void check_fvmatrix_equal(const fvVectorMatrix& answer,const fvVectorMatrix& check, const word& name){
//     if(answer.source().begin() == check.source().begin()){
//         SeriousError << "answer and check are the same matrix." << endl;
//         MPI_Abort(PstreamGlobals::MPI_COMM_FOAM, -1);
//     } 

//     for (direction cmpt=0; cmpt<vector::nComponents; ++cmpt)
//     {
//         check_field_error(answer.source().component(cmpt).ref(),check.source().component(cmpt).ref(), name + " source_" + std::to_string(cmpt));
//     }

//     check_field_error(answer.diag(),check.diag(), name + " diag");
//     check_field_error(answer.lower(),check.lower(), name + " lower");
//     check_field_error(answer.upper(),check.upper(), name + " upper");

//     for (direction cmpt=0; cmpt<vector::nComponents; ++cmpt)
//     {
//         forAll(const_cast<fvVectorMatrix&>(answer).internalCoeffs().component(cmpt)(), patchi)
//         {
//             check_field_error(const_cast<fvVectorMatrix&>(answer).internalCoeffs().component(cmpt).ref()[patchi],const_cast<fvVectorMatrix&>(check).internalCoeffs().component(cmpt).ref()[patchi], name + " internalCoeffs_" + std::to_string(cmpt) + "_" + std::to_string(patchi));
//             check_field_error(const_cast<fvVectorMatrix&>(answer).boundaryCoeffs().component(cmpt).ref()[patchi],const_cast<fvVectorMatrix&>(check).boundaryCoeffs().component(cmpt).ref()[patchi], name + " boundaryCoeffs_" + std::to_string(cmpt) + "_" + std::to_string(patchi));
//         }
//     }

// }


void check_fvmatrix_equal(const fvVectorMatrix& answer, const fvVectorMatrix& check, const word& name)
{
    bool check_failed = false;
    word error_msg = "";
    
    // 只有主核心执行检查逻辑
    if (Pstream::master())
    {
        if(answer.source().begin() == check.source().begin())
        {
            error_msg = "answer and check are the same matrix.";
            check_failed = true;
        }
        else
        {
            try
            {
                for (direction cmpt=0; cmpt<vector::nComponents && !check_failed; ++cmpt)
                {
                    check_field_error(answer.source().component(cmpt).ref(),
                                    check.source().component(cmpt).ref(), 
                                    name + " source_" + std::to_string(cmpt));
                }
                
                if (!check_failed) check_field_error(answer.diag(), check.diag(), name + " diag");
                if (!check_failed) check_field_error(answer.lower(), check.lower(), name + " lower");
                if (!check_failed) check_field_error(answer.upper(), check.upper(), name + " upper");
                
                for (direction cmpt=0; cmpt<vector::nComponents && !check_failed; ++cmpt)
                {
                    forAll(const_cast<fvVectorMatrix&>(answer).internalCoeffs().component(cmpt)(), patchi)
                    {
                        check_field_error(const_cast<fvVectorMatrix&>(answer).internalCoeffs().component(cmpt).ref()[patchi],
                                        const_cast<fvVectorMatrix&>(check).internalCoeffs().component(cmpt).ref()[patchi], 
                                        name + " internalCoeffs_" + std::to_string(cmpt) + "_" + std::to_string(patchi));
                        check_field_error(const_cast<fvVectorMatrix&>(answer).boundaryCoeffs().component(cmpt).ref()[patchi],
                                        const_cast<fvVectorMatrix&>(check).boundaryCoeffs().component(cmpt).ref()[patchi], 
                                        name + " boundaryCoeffs_" + std::to_string(cmpt) + "_" + std::to_string(patchi));
                    }
                }
            }
            catch (const Foam::error& e)
            {
                check_failed = true;
                error_msg = "Exception during matrix comparison: " + string(e.what());
            }
        }
        
        if (!check_failed)
        {
            Info << "Matrix check passed: " << name << endl;
        }
    }
    
    // 将检查结果广播给所有进程
    Pstream::scatter(check_failed);
    
    // 如果检查失败，所有进程一起终止
    // if(check_failed)
    // {
    //     if (Pstream::master())
    //     {
    //         SeriousError << "Matrix check failed: " << name << " - " << error_msg << endl;
    //     }
    //     MPI_Abort(PstreamGlobals::MPI_COMM_FOAM, -1);
    // }
    
    // 同步所有进程
    // Pstream::scatter(bool(true));
}














}

