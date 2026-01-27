/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2018 OpenFOAM Foundation
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

Global
    rhoEqn

Description
    Solve the continuity for density.

\*---------------------------------------------------------------------------*/
Info << "Enter rhoEqn " << endl;

#ifdef GPUSolverNew_
    double *h_rho = dfDataBase.getFieldPointer("rho", location::cpu, position::internal);
    double *h_boundary_rho = dfDataBase.getFieldPointer("rho", location::cpu, position::boundary);

    TICK_START;
    rhoEqn_GPU.process();
    rhoEqn_GPU.sync();
    TICK_STOP(GPU process time);

    // rhoEqn_GPU.postProcess(h_rho);
    // rho.oldTime();
    // memcpy(&rho[0], h_rho, dfDataBase.cell_value_bytes);
    // rho.correctBoundaryConditions();


#ifdef DEBUG_
    // checkValue
    TICK_START;
    fvScalarMatrix rhoEqn
    (
        fvm::ddt(rho)
      + fvc::div(phi)
    );
    rhoEqn.solve();
    TICK_STOP(CPU process time);

    int rank = -1;
    if (mpi_init_flag) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    // if (!mpi_init_flag || rank == 0) {
    //     rhoEqn_GPU.compareResult(&rhoEqn.diag()[0], &rhoEqn.source()[0], false);
    // }

    offset = 0;
    forAll(rho.boundaryField(), patchi)
    {
        const fvPatchScalarField& patchRho = rho.boundaryField()[patchi];
        int patchsize = patchRho.size();
        if (patchRho.type() == "processor"
            || patchRho.type() == "processorCyclic") {
            memcpy(h_boundary_rho + offset, &patchRho[0], patchsize * sizeof(double));
            scalarField patchRhoInternal = 
                    dynamic_cast<const processorFvPatchField<scalar>&>(patchRho).patchInternalField()();
            memcpy(h_boundary_rho + offset + patchsize, &patchRhoInternal[0], patchsize * sizeof(double));
            offset += patchsize * 2;
        } else {
            memcpy(h_boundary_rho + offset, &patchRho[0], patchsize * sizeof(double));
            offset += patchsize;
        }
    }
    // if (!mpi_init_flag || rank == 0) {
    //     rhoEqn_GPU.compareRho(&rho[0], h_boundary_rho, false);
    // }
#endif
#else
{
    start1 = MPI_Wtime();
    fvScalarMatrix rhoEqn
    (
        fvm::ddt(rho)
      + fvc::div(phi)
    );
    end1 = MPI_Wtime();
    time_monitor_rhoEqn += double(end1 - start1);
    time_monitor_rhoEqn_mtxAssembly += double(end1 - start1);

    start1 = MPI_Wtime();
#ifdef USE_DF_MATRIX
    dfMatrix df_rhoEqn(rhoEqn, regionPtr);
    auto& psi = const_cast<volScalarField&>(rhoEqn.psi());
    auto& source = rhoEqn.source();
    auto& internalCoeffs = rhoEqn.internalCoeffs();
    auto& boundaryCoeffs = rhoEqn.boundaryCoeffs();
    df_rhoEqn.solve(psi, source, internalCoeffs, boundaryCoeffs);
#else
    rhoEqn.solve();
#endif
    end1 = MPI_Wtime();
    time_monitor_rhoEqn += double(end1 - start1);
    time_monitor_rhoEqn_solve += double(end1 - start1);
}
#endif
// ************************************************************************* //

Info << "Exit rhoEqn " << endl;