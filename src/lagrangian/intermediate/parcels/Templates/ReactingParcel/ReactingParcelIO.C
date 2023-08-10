/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2016-2022 OpenCFD Ltd.
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

#include "ReactingParcel.H"
#include "IOstreams.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

template<class ParcelType>
Foam::string Foam::ReactingParcel<ParcelType>::propertyList_ =
    Foam::ReactingParcel<ParcelType>::propertyList();


template<class ParcelType>
const std::size_t Foam::ReactingParcel<ParcelType>::sizeofFields
(
    sizeof(scalar)
);


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class ParcelType>
Foam::ReactingParcel<ParcelType>::ReactingParcel
(
    const polyMesh& mesh,
    Istream& is,
    bool readFields,
    bool newFormat
)
:
    ParcelType(mesh, is, readFields, newFormat),
    mass0_(0.0),
    Y_(0)
{
    if (readFields)
    {
        DynamicList<scalar> Ymix;

        if (is.format() == IOstreamOption::ASCII)
        {
            is >> mass0_;
        }
        else if (!is.checkLabelSize<>() || !is.checkScalarSize<>())
        {
            // Non-native label or scalar size

            is.beginRawRead();

            readRawScalar(is, &mass0_);

            is.endRawRead();
        }
        else
        {
            is.read(reinterpret_cast<char*>(&mass0_), sizeofFields);
        }

        is >> Ymix;

        Y_.transfer(Ymix);
    }

    is.check(FUNCTION_NAME);
}


template<class ParcelType>
template<class CloudType>
void Foam::ReactingParcel<ParcelType>::readFields(CloudType& c)
{
    ParcelType::readFields(c);
}


template<class ParcelType>
template<class CloudType, class CompositionType>
void Foam::ReactingParcel<ParcelType>::readFields
(
    CloudType& c,
    const CompositionType& compModel
)
{
    bool valid = c.size();

    ParcelType::readFields(c);

    IOField<scalar> mass0
    (
        c.fieldIOobject("mass0", IOobject::MUST_READ),
        valid
    );
    c.checkFieldIOobject(c, mass0);

    label i = 0;
    for (ReactingParcel<ParcelType>& p : c)
    {
        p.mass0_ = mass0[i];

        ++i;
    }

    // Get names and sizes for each Y...
    const wordList& phaseTypes = compModel.phaseTypes();
    const label nPhases = phaseTypes.size();
    wordList stateLabels(nPhases, "");
    if (compModel.nPhase() == 1)
    {
        stateLabels = compModel.stateLabels()[0];
    }


    // Set storage for each Y... for each parcel
    for (ReactingParcel<ParcelType>& p : c)
    {
        p.Y_.setSize(nPhases, 0.0);
    }

    // Populate Y for each parcel
    forAll(phaseTypes, j)
    {
        IOField<scalar> Y
        (
            c.fieldIOobject
            (
                "Y" + phaseTypes[j] + stateLabels[j],
                 IOobject::MUST_READ
            ),
            valid
        );

        label i = 0;
        for (ReactingParcel<ParcelType>& p : c)
        {
            p.Y_[j] = Y[i];

            ++i;
        }
    }
}


template<class ParcelType>
template<class CloudType>
void Foam::ReactingParcel<ParcelType>::writeFields(const CloudType& c)
{
    ParcelType::writeFields(c);
}


template<class ParcelType>
template<class CloudType, class CompositionType>
void Foam::ReactingParcel<ParcelType>::writeFields
(
    const CloudType& c,
    const CompositionType& compModel
)
{
    ParcelType::writeFields(c);

    const label np = c.size();

    {
        IOField<scalar> mass0(c.fieldIOobject("mass0", IOobject::NO_READ), np);

        label i = 0;
        for (const ReactingParcel<ParcelType>& p : c)
        {
            mass0[i] = p.mass0_;

            ++i;
        }
        mass0.write(np > 0);

        // Write the composition fractions
        const wordList& phaseTypes = compModel.phaseTypes();
        wordList stateLabels(phaseTypes.size(), "");
        if (compModel.nPhase() == 1)
        {
            stateLabels = compModel.stateLabels()[0];
        }

        forAll(phaseTypes, j)
        {
            IOField<scalar> Y
            (
                c.fieldIOobject
                (
                    "Y" + phaseTypes[j] + stateLabels[j],
                    IOobject::NO_READ
                ),
                np
            );

            label i = 0;
            for (const ReactingParcel<ParcelType>& p : c)
            {
                Y[i] = p.Y()[j];

                ++i;
            }

            Y.write(np > 0);
        }
    }
}


template<class ParcelType>
void Foam::ReactingParcel<ParcelType>::writeProperties
(
    Ostream& os,
    const wordRes& filters,
    const word& delim,
    const bool namesOnly
) const
{
    ParcelType::writeProperties(os, filters, delim, namesOnly);

    #undef  writeProp
    #define writeProp(Name, Value)                                            \
        ParcelType::writeProperty(os, Name, Value, namesOnly, delim, filters)

    writeProp("mass0", mass0_);
    writeProp("Y", Y_);

    #undef writeProp
}


template<class ParcelType>
template<class CloudType>
void Foam::ReactingParcel<ParcelType>::readObjects
(
    CloudType& c,
    const objectRegistry& obr
)
{
    ParcelType::readObjects(c, obr);
}


template<class ParcelType>
template<class CloudType>
void Foam::ReactingParcel<ParcelType>::writeObjects
(
    const CloudType& c,
    objectRegistry& obr
)
{
    ParcelType::writeObjects(c, obr);
}


template<class ParcelType>
template<class CloudType, class CompositionType>
void Foam::ReactingParcel<ParcelType>::readObjects
(
    CloudType& c,
    const CompositionType& compModel,
    const objectRegistry& obr
)
{
    ParcelType::readObjects(c, obr);

    if (!c.size()) return;


    auto& mass0 = cloud::lookupIOField<scalar>("mass0", obr);

    label i = 0;
    for (ReactingParcel<ParcelType>& p : c)
    {
        p.mass0_ = mass0[i];

        ++i;
    }

    // The composition fractions
    const wordList& phaseTypes = compModel.phaseTypes();
    wordList stateLabels(phaseTypes.size(), "");
    if (compModel.nPhase() == 1)
    {
        stateLabels = compModel.stateLabels()[0];
    }

    forAll(phaseTypes, j)
    {
        const word fieldName = "Y" + phaseTypes[j] + stateLabels[j];
        auto& Y = cloud::lookupIOField<scalar>(fieldName, obr);

        label i = 0;
        for (ReactingParcel<ParcelType>& p : c)
        {
            p.Y()[j] = Y[i];

            ++i;
        }
    }
}


template<class ParcelType>
template<class CloudType, class CompositionType>
void Foam::ReactingParcel<ParcelType>::writeObjects
(
    const CloudType& c,
    const CompositionType& compModel,
    objectRegistry& obr
)
{
    ParcelType::writeObjects(c, obr);

    const label np = c.size();

    if (np > 0)
    {
        auto& mass0 = cloud::createIOField<scalar>("mass0", np, obr);

        label i = 0;
        for (const ReactingParcel<ParcelType>& p : c)
        {
            mass0[i] = p.mass0_;

            ++i;
        }

        // Write the composition fractions
        const wordList& phaseTypes = compModel.phaseTypes();
        wordList stateLabels(phaseTypes.size(), "");
        if (compModel.nPhase() == 1)
        {
            stateLabels = compModel.stateLabels()[0];
        }

        forAll(phaseTypes, j)
        {
            const word fieldName = "Y" + phaseTypes[j] + stateLabels[j];
            auto& Y = cloud::createIOField<scalar>(fieldName, np, obr);

            label i = 0;
            for (const ReactingParcel<ParcelType>& p : c)
            {
                Y[i] = p.Y()[j];

                ++i;
            }
        }
    }
}


// * * * * * * * * * * * * * * * IOstream Operators  * * * * * * * * * * * * //

template<class ParcelType>
Foam::Ostream& Foam::operator<<
(
    Ostream& os,
    const ReactingParcel<ParcelType>& p
)
{
    if (os.format() == IOstreamOption::ASCII)
    {
        os  << static_cast<const ParcelType&>(p)
            << token::SPACE << p.mass0()
            << token::SPACE << p.Y();
    }
    else
    {
        os  << static_cast<const ParcelType&>(p);
        os.write
        (
            reinterpret_cast<const char*>(&p.mass0_),
            ReactingParcel<ParcelType>::sizeofFields
        );
        os  << p.Y();
    }

    os.check(FUNCTION_NAME);
    return os;
}


// ************************************************************************* //
