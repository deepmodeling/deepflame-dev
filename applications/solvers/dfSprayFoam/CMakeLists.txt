project(dfSprayFoam LANGUAGES CXX)

include_directories(
    #${OpenFOAM_APP}/solvers/lagrangian/reactingParcelFoam 
    #${OpenFOAM_APP}/solvers/compressible/rhoPimpleFoam/correctPhi.H
    ${OpenFOAM_SRC}/finiteVolume/lnInclude 
    ${OpenFOAM_SRC}/meshTools/lnInclude 
    ${OpenFOAM_SRC}/sampling/lnInclude 
    ${OpenFOAM_SRC}/TurbulenceModels/turbulenceModels/lnInclude 
    ${OpenFOAM_SRC}/TurbulenceModels/compressible/lnInclude 
    ${OpenFOAM_SRC}/regionModels/regionModel/lnInclude 
    ${OpenFOAM_SRC}/regionModels/surfaceFilmModels/lnInclude 
    ${OpenFOAM_SRC}/dynamicFvMesh/lnInclude 
    ${OpenFOAM_SRC}/meshTools/lnInclude 
    ${OpenFOAM_SRC}/lagrangian/basic/lnInclude 
    ${dflagrangianinter_inc}
    #${DF_SRC}/lagrangian/intermediate/lnInclude 
    ${OpenFOAM_SRC}/lagrangian/intermediate/lnInclude 
    ${dflagrangianspray_inc} 
    #${DF_SRC}/lagrangian/spray/lnInclude
    ${OpenFOAM_SRC}/lagrangian/spray/lnInclude 
    ${OpenFOAM_SRC}/lagrangian/distributionModels/lnInclude 
    ${OpenFOAM_SRC}/transportModels/compressible/lnInclude 
    ${OpenFOAM_SRC}/thermophysicalModels/basic/lnInclude 
    ${dfthermophysicalprop_inc} 
    ${OpenFOAM_SRC}/thermophysicalModels/thermophysicalProperties/lnInclude 
    ${dfthermophysicalslg_inc} 
    ${dfcantera_inc}
    ${dfchemistry_inc}
    ${dfcombustion_inc} 
    ${OpenFOAM_SRC}/Pstream/mpi
    ${CANTERA_ROOT}/include 
    ${MPI_INCLUDE_PATH}
    ${PROJECT_SOURCE_DIR}

)


add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/dfSprayFoam.C)

if(DEFINED ENV{PYTHON_INC_DIR})
  add_definitions(-DUSE_PYTORCH)
  include_directories(
  /usr/include/python3.8
  /home/runze/.local/lib/python3.8/site-packages/pybind11/include
  )
  target_link_libraries(${PROJECT_NAME} /usr/lib/python3.8/config-3.8-x86_64-linux-gnu/libpython3.8.so)
endif()

target_link_libraries(${PROJECT_NAME} $ENV{FOAM_LIBBIN}/libfiniteVolume.so 
libcompressibleTransportModels.so libturbulenceModels.so libsampling.so 
libOpenFOAM.so liblagrangian.so libregionModels.so libdynamicFvMesh.so 
libtopoChangerFvMesh.so libmeshTools.so 
/home/runze/OpenFOAM/ThirdParty-7/platforms/linux64GccDPInt32/lib/openmpi-system/libptscotch.so
/home/runze/OpenFOAM/ThirdParty-7/platforms/linux64GccDPInt32/lib/openmpi-system/libptscotcherrexit.so
/home/runze/OpenFOAM/ThirdParty-7/platforms/linux64GccDPInt32/lib/openmpi-system/libscotch.so

)

target_link_libraries(${PROJECT_NAME} ${CANTERA_ROOT}/lib/libcantera_shared.so.2)

target_link_libraries(${PROJECT_NAME} dfCompressibleTurbulenceModels
dfChemistryModel 
dfThermophysicalProperties 
dfSLGThermo 
dfLagrangianIntermediate
dfLagrangianTurbulence 
dfLagrangianSpray 
dfCanteraMixture 
dfFluidThermophysicalModels
dfCombustionModels 
dfSurfaceFilmModels 
#${MPI_LIBRARIES}  
#$ENV{FOAM_LIBBIN}/openmpi-system/libPstream.so
#libptscotchDecomp.so
)


install(TARGETS ${PROJECT_NAME} DESTINATION bin)