#include "divMatrix.H"
#include "processorDIVGAMGInterfaceField.H"

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

void Foam::divMatrix::initMatrixInterfaces
(
    const FieldField<Field, scalar>& coupleCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const scalarField& psiif,
    scalarField& result,
    const direction cmpt
) const
{
    // if
    // (
    //     Pstream::defaultCommsType == Pstream::commsTypes::blocking
    //  || Pstream::defaultCommsType == Pstream::commsTypes::nonBlocking
    // )
    // {
    //     forAll(interfaces, interfacei)
    //     {
    //         if (interfaces.set(interfacei))
    //         {
    //             interfaces[interfacei].initInterfaceMatrixUpdate
    //             (
    //                 result,
    //                 psiif,
    //                 coupleCoeffs[interfacei],
    //                 cmpt,
    //                 Pstream::defaultCommsType
    //             );
    //         }
    //     }
    // }
    // // else if (Pstream::defaultCommsType == Pstream::commsTypes::scheduled)
    // // {
    // //     const lduSchedule& patchSchedule = this->patchSchedule();

    // //     // Loop over the "global" patches are on the list of interfaces but
    // //     // beyond the end of the schedule which only handles "normal" patches
    // //     for
    // //     (
    // //         label interfacei=patchSchedule.size()/2;
    // //         interfacei<interfaces.size();
    // //         interfacei++
    // //     )
    // //     {
    // //         if (interfaces.set(interfacei))
    // //         {
    // //             interfaces[interfacei].initInterfaceMatrixUpdate
    // //             (
    // //                 result,
    // //                 psiif,
    // //                 coupleCoeffs[interfacei],
    // //                 cmpt,
    // //                 Pstream::commsTypes::blocking
    // //             );
    // //         }
    // //     }
    // // }
    // else
    // {
    //     FatalErrorInFunction
    //         << "Unsupported communications type "
    //         << Pstream::commsTypeNames[Pstream::defaultCommsType]
    //         << exit(FatalError);
    // }
    forAll(interfaces, interfacei) {
        if (interfaces.set(interfacei)) {
            MPI_Request send_req, recv_req;
            if (isA<processorDIVGAMGInterfaceField>(interfaces[interfacei])) {
                const processorDIVGAMGInterfaceField& processorPatch = 
                    dynamic_cast<const processorDIVGAMGInterfaceField&>(interfaces[interfacei]);
                label interfaceiSize = processorPatch.size();
                scalarSendBufListGAMG_[interfacei].resize(interfaceiSize);
                scalarRecvBufListGAMG_[interfacei].resize(interfaceiSize);
                const labelUList& faceCells = processorPatch.procInterface().faceCells();
#ifdef _OPENMP
#pragma omp parallel for
#endif
                for (label facei=0; facei<interfaceiSize; facei++){
                    scalarSendBufListGAMG_[interfacei][facei] = psiif[faceCells[facei]];
                }
                MPI_Isend(scalarSendBufListGAMG_[interfacei].data(), interfaceiSize, MPI_DOUBLE, neighbProcNo[interfacei], 0,
                    MPI_COMM_WORLD, &send_req);
                MPI_Irecv(scalarRecvBufListGAMG_[interfacei].data(), interfaceiSize, MPI_DOUBLE, neighbProcNo[interfacei], 0,
                    MPI_COMM_WORLD, &recv_req);
            } else {
                label interfaceiSize = surfacePerPatch[interfacei];
#ifdef _OPENMP
#pragma omp parallel for
#endif
                for (label facei=0; facei<interfaceiSize; facei++){
                    scalarSendBufList_[interfacei][facei] = psiif[faceCellsGroup[interfacei][facei]];
                }
                MPI_Isend(scalarSendBufList_[interfacei], interfaceiSize, MPI_DOUBLE, neighbProcNo[interfacei], 0,
                    MPI_COMM_WORLD, &send_req);
                MPI_Irecv(scalarRecvBufList_[interfacei], interfaceiSize, MPI_DOUBLE, neighbProcNo[interfacei], 0,
                    MPI_COMM_WORLD, &recv_req);
            }
            send_recv_requests_.push_back(send_req);
            send_recv_requests_.push_back(recv_req);
        }
    }
}


void Foam::divMatrix::updateMatrixInterfaces
(
    const FieldField<Field, scalar>& coupleCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const scalarField& psiif,
    scalarField& result,
    const direction cmpt
) const
{
    // if (Pstream::defaultCommsType == Pstream::commsTypes::blocking)
    // {
    //     forAll(interfaces, interfacei)
    //     {
    //         if (interfaces.set(interfacei))
    //         {
    //             interfaces[interfacei].updateInterfaceMatrix
    //             (
    //                 result,
    //                 psiif,
    //                 coupleCoeffs[interfacei],
    //                 cmpt,
    //                 Pstream::defaultCommsType
    //             );
    //         }
    //     }
    // }
    // else if (Pstream::defaultCommsType == Pstream::commsTypes::nonBlocking)
    // {
    //     // Try and consume interfaces as they become available
    //     bool allUpdated = false;

    //     for (label i=0; i<UPstream::nPollProcInterfaces; i++)
    //     {
    //         allUpdated = true;

    //         forAll(interfaces, interfacei)
    //         {
    //             if (interfaces.set(interfacei))
    //             {
    //                 if (!interfaces[interfacei].updatedMatrix())
    //                 {
    //                     if (interfaces[interfacei].ready())
    //                     {
    //                         interfaces[interfacei].updateInterfaceMatrix
    //                         (
    //                             result,
    //                             psiif,
    //                             coupleCoeffs[interfacei],
    //                             cmpt,
    //                             Pstream::defaultCommsType
    //                         );
    //                     }
    //                     else
    //                     {
    //                         allUpdated = false;
    //                     }
    //                 }
    //             }
    //         }

    //         if (allUpdated)
    //         {
    //             break;
    //         }
    //     }

    //     // Block for everything
    //     if (Pstream::parRun())
    //     {
    //         if (allUpdated)
    //         {
    //             // All received. Just remove all storage of requests
    //             // Note that we don't know what starting number of requests
    //             // was before start of sends and receives (since set from
    //             // initMatrixInterfaces) so set to 0 and loose any in-flight
    //             // requests.
    //             UPstream::resetRequests(0);
    //         }
    //         else
    //         {
    //             // Block for all requests and remove storage
    //             UPstream::waitRequests();
    //         }
    //     }

    //     // Consume
    //     forAll(interfaces, interfacei)
    //     {
    //         if
    //         (
    //             interfaces.set(interfacei)
    //         && !interfaces[interfacei].updatedMatrix()
    //         )
    //         {
    //             interfaces[interfacei].updateInterfaceMatrix
    //             (
    //                 result,
    //                 psiif,
    //                 coupleCoeffs[interfacei],
    //                 cmpt,
    //                 Pstream::defaultCommsType
    //             );
    //         }
    //     }
    // }
    // // else if (Pstream::defaultCommsType == Pstream::commsTypes::scheduled)
    // // {
    // //     const lduSchedule& patchSchedule = this->patchSchedule();

    // //     // Loop over all the "normal" interfaces relating to standard patches
    // //     forAll(patchSchedule, i)
    // //     {
    // //         label interfacei = patchSchedule[i].patch;

    // //         if (interfaces.set(interfacei))
    // //         {
    // //             if (patchSchedule[i].init)
    // //             {
    // //                 interfaces[interfacei].initInterfaceMatrixUpdate
    // //                 (
    // //                     result,
    // //                     psiif,
    // //                     coupleCoeffs[interfacei],
    // //                     cmpt,
    // //                     Pstream::commsTypes::scheduled
    // //                 );
    // //             }
    // //             else
    // //             {
    // //                 interfaces[interfacei].updateInterfaceMatrix
    // //                 (
    // //                     result,
    // //                     psiif,
    // //                     coupleCoeffs[interfacei],
    // //                     cmpt,
    // //                     Pstream::commsTypes::scheduled
    // //                 );
    // //             }
    // //         }
    // //     }

    // //     // Loop over the "global" patches are on the list of interfaces but
    // //     // beyond the end of the schedule which only handles "normal" patches
    // //     for
    // //     (
    // //         label interfacei=patchSchedule.size()/2;
    // //         interfacei<interfaces.size();
    // //         interfacei++
    // //     )
    // //     {
    // //         if (interfaces.set(interfacei))
    // //         {
    // //             interfaces[interfacei].updateInterfaceMatrix
    // //             (
    // //                 result,
    // //                 psiif,
    // //                 coupleCoeffs[interfacei],
    // //                 cmpt,
    // //                 Pstream::commsTypes::blocking
    // //             );
    // //         }
    // //     }
    // // }
    // else
    // {
    //     FatalErrorInFunction
    //         << "Unsupported communications type "
    //         << Pstream::commsTypeNames[Pstream::defaultCommsType]
    //         << exit(FatalError);
    // }
    MPI_Waitall(send_recv_requests_.size(), send_recv_requests_.data(), MPI_STATUSES_IGNORE);
    forAll(interfaces, interfacei){
        if (interfaces.set(interfacei)){
            if (isA<processorDIVGAMGInterfaceField>(interfaces[interfacei])) {
                const processorDIVGAMGInterfaceField& processorPatch = 
                    dynamic_cast<const processorDIVGAMGInterfaceField&>(interfaces[interfacei]);
                label interfaceiSize = processorPatch.size();
                const labelUList& faceCells = processorPatch.procInterface().faceCells();
#ifdef _OPENMP
#pragma omp parallel for
#endif
                for (label facei=0; facei<interfaceiSize; facei++){
                    result[faceCells[facei]] -= coupleCoeffs[interfacei][facei] * scalarRecvBufListGAMG_[interfacei][facei];
                }
            } else {
                label interfaceiSize = surfacePerPatch[interfacei];
#ifdef _OPENMP
#pragma omp parallel for
#endif
                for (label facei=0; facei<interfaceiSize; facei++){
                    result[faceCellsGroup[interfacei][facei]] -= coupleCoeffs[interfacei][facei] * scalarRecvBufList_[interfacei][facei];
                }
            }
        }
    }
    send_recv_requests_.clear();
}


// ************************************************************************* //
