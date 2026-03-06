/*---------------------------------------------------------------------------*\
    createMeshes.H — Read regionProperties and create meshes for all regions
\*---------------------------------------------------------------------------*/

regionProperties rp(runTime);

#include "lowMach/createLowMachMeshes.H"
#include "highSpeed/createHighSpeedMeshes.H"
