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

\*---------------------------------------------------------------------------*/

#include "basicThermo.H"
#include "psiThermo.H"
#include "CanteraMixture.H"
#include "hePsiThermo.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */


#define defineThermoPhysicsThermo(CThermo,Mixture,ThermoPhys)                  \
                                                                               \
    typedef                                                                    \
        CThermo                                                                \
        <                                                                      \
            ThermoPhys,                                                        \
            Mixture                                                            \
        >                                                                      \
        CThermo##Mixture##ThermoPhys;                                          \
                                                                               \
    defineTemplateTypeNameAndDebugWithName                                     \
    (                                                                          \
        CThermo##Mixture##ThermoPhys,                                          \
        (#CThermo"<" + Mixture::typeName() + ">").c_str(),                     \
        0                                                                      \
    )

#define addThermoPhysicsThermo(BaseThermo,CThermoMixtureThermoPhys)            \
                                                                               \
    addToRunTimeSelectionTable                                                 \
    (                                                                          \
        BaseThermo,                                                            \
        CThermoMixtureThermoPhys,                                              \
        fvMesh                                                                 \
    );                                                                         \

#define makeThermoPhysicsThermos(CThermo,Mixture,ThermoPhys)                   \
                                                                               \
    defineThermoPhysicsThermo(CThermo, Mixture, ThermoPhys);                   \
                                                                               \
    addThermoPhysicsThermo(basicThermo, CThermo##Mixture##ThermoPhys);         \
    addThermoPhysicsThermo(fluidThermo, CThermo##Mixture##ThermoPhys)


makeThermoPhysicsThermos(hePsiThermo, CanteraMixture, psiThermo);

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //