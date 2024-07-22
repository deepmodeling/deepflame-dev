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
#ifdef GPUSolverNew_
#ifdef DEBUG_
    // checkValue
    TICK_START;
    fvScalarMatrix rhoEqn
    (
        fvm::ddt(rho)
      + fvc::div(phi)
    );

    // add for test
    double *diag = &rhoEqn.diag()[0];
    double *source = &rhoEqn.source()[0];
    int num_cells = rho.mesh().nCells();
    int num_surfaces = rho.mesh().neighbour().size();

    writeDoubleArrayToFile(diag, num_cells, "rho_diag.host");
    writeDoubleArrayToFile(source, num_cells, "rho_source.host");

    rhoEqn.solve();
    TICK_STOP(CPU process time);

    int rank = -1;
    if (mpi_init_flag) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    writeDoubleArrayToFile(&rho[0], num_cells, "rho_rho_solved.host");

    double *h_boundary_rho = (double*)calloc(mesh_paras.num_boundary_surfaces, sizeof(double));
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

    writeDoubleArrayToFile(&h_boundary_rho[0], mesh_paras.num_boundary_surfaces, "rho_boundary_rho.host");

#endif
#else
{
    start1 = std::clock();
    fvScalarMatrix rhoEqn
    (
        fvm::ddt(rho)
      + fvc::div(phi)
    );
    end1 = std::clock();
    time_monitor_rhoEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_rhoEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);

    start1 = std::clock();
    rhoEqn.solve();
    end1 = std::clock();
    time_monitor_rhoEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_rhoEqn_solve += double(end1 - start1) / double(CLOCKS_PER_SEC);
}
#endif
//#undef CPUSolver_
//#define GPUSolverNew_
// ************************************************************************* //
