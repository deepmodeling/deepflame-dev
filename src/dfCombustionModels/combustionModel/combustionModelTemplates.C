/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2019-2021 OpenCFD Ltd.
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

// * * * * * * * * * * * * * * * * Selectors * * * * * * * * * * * * * * * * //

template<class CombustionModel>
Foam::autoPtr<CombustionModel> Foam::combustionModel::New
(
    typename CombustionModel::reactionThermo& thermo,
    const compressibleTurbulenceModel& turb,
    const word& combustionProperties
)
{
    IOobject combIO
    (
        thermo.phasePropertyName(combustionProperties),
        thermo.db().time().constant(),
        thermo.db(),
        IOobject::MUST_READ,
        IOobject::NO_WRITE,
        false
    );

    word combModelName("none");
    if (combIO.typeHeaderOk<IOdictionary>(false))
    {
        IOdictionary(combIO).readEntry("combustionModel", combModelName);
    }
    else
    {
        Info<< "Combustion model not active: "
            << thermo.phasePropertyName(combustionProperties)
            << " not found" << endl;
    }

    Info<< "Selecting combustion model " << combModelName << endl;

    const wordList cmpts2(basicThermo::splitThermoName(combModelName, 2));
    const wordList cmpts3(basicThermo::splitThermoName(combModelName, 3));
    if (cmpts2.size() == 2 || cmpts3.size() == 3)
    {
        combModelName = cmpts2.size() ? cmpts2[0] : cmpts3[0];

        WarningInFunction
            << "Template parameters are no longer required when selecting a "
            << combustionModel::typeName << ". This information is now "
            << "obtained directly from the thermodynamics. Actually selecting "
            << "combustion model " << combModelName << "." << endl;
    }


    const word compCombModelName
    (
        combModelName + '<' + CombustionModel::reactionThermo::typeName + '>'
    );

    const word thermoCombModelName
    (
        combModelName + '<' + CombustionModel::reactionThermo::typeName + ','
      + thermo.thermoName() + '>'
    );


    const auto& cnstrTable = *(CombustionModel::dictionaryConstructorTablePtr_);

    auto ctorIter = cnstrTable.cfind(thermoCombModelName);

    if (!ctorIter.found())
    {
        ctorIter = cnstrTable.cfind(compCombModelName);
    }

    if (!ctorIter.found())
    {
        const wordList names(cnstrTable.sortedToc());

        /// DynamicList<word> thisCmpts(6);
        /// thisCmpts.append(CombustionModel::reactionThermo::typeName);
        /// thisCmpts.append(basicThermo::splitThermoName
        /// (
        ///     basicThermo::splitThermoName(thermo.thermoName(), 5)
        /// );
        ///
        /// DynamicList<word> validNames;

        DynamicList<wordList> validCmpts2;
        validCmpts2.append
        (
            // Header
            wordList
            ({
                word::null,
                combustionModel::typeName,
                "reactionThermo"
            })
        );

        DynamicList<wordList> validCmpts7;
        validCmpts7.append
        (
            // Header
            wordList
            ({
                word::null,
                combustionModel::typeName,
                "reactionThermo",
                "transport",
                "thermo",
                "equationOfState",
                "specie",
                "energy"
            })
        );

        for (const word& validName : names)
        {
            wordList cmpts(basicThermo::splitThermoName(validName, 0));

            if (cmpts.size() == 2)
            {
                validCmpts2.append(std::move(cmpts));
            }
            else if (cmpts.size() == 7)
            {
                /// if (thisCmpts == SubList<word>(cmpts, 6, 1))
                /// {
                ///     validNames.append(cmpts[0]);
                /// }
                validCmpts7.append(std::move(cmpts));
            }
        }

        FatalErrorInLookup
        (
            combustionModel::typeName,
            combModelName,
            cnstrTable
        );

        if (validCmpts2.size() > 1)
        {
            FatalError
                << "All " << validCmpts2[0][0] << '/' << validCmpts2[0][1]
                << " combinations are:" << nl << nl;

            printTable(validCmpts2, FatalError) << nl;
        }

        if (validCmpts7.size() > 1)
        {
            FatalError
                << "All " << validCmpts7[0][0] << '/' << validCmpts7[0][1]
                << "/thermoPhysics combinations are:" << nl << nl;

            printTable(validCmpts7, FatalError) << nl;
        }

        FatalError
            << exit(FatalError);
    }

    return autoPtr<CombustionModel>
    (
        ctorIter.val()(combModelName, thermo, turb, combustionProperties)
    );
}


// ************************************************************************* //
