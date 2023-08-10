/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2019-2022 OpenCFD Ltd.
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

#include "CollidingParcel.H"
#include "IOstreams.H"
#include "IOField.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

template<class ParcelType>
Foam::string Foam::CollidingParcel<ParcelType>::propertyList_ =
    Foam::CollidingParcel<ParcelType>::propertyList();

template<class ParcelType>
const std::size_t Foam::CollidingParcel<ParcelType>::sizeofFields
(
    offsetof(CollidingParcel<ParcelType>, collisionRecords_)
  - offsetof(CollidingParcel<ParcelType>, f_)
);


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class ParcelType>
Foam::CollidingParcel<ParcelType>::CollidingParcel
(
    const polyMesh& mesh,
    Istream& is,
    bool readFields,
    bool newFormat
)
:
    ParcelType(mesh, is, readFields, newFormat),
    f_(Zero),
    angularMomentum_(Zero),
    torque_(Zero),
    collisionRecords_()
{
    if (readFields)
    {
        if (is.format() == IOstreamOption::ASCII)
        {
            is >> f_;
            is >> angularMomentum_;
            is >> torque_;
        }
        else if (!is.checkLabelSize<>() || !is.checkScalarSize<>())
        {
            // Non-native label or scalar size

            is.beginRawRead();

            readRawScalar(is, f_.data(), vector::nComponents);
            readRawScalar(is, angularMomentum_.data(), vector::nComponents);
            readRawScalar(is, torque_.data(), vector::nComponents);

            is.endRawRead();
        }
        else
        {
            is.read(reinterpret_cast<char*>(&f_), sizeofFields);
        }

        is >> collisionRecords_;
    }

    is.check(FUNCTION_NAME);
}


template<class ParcelType>
template<class CloudType>
void Foam::CollidingParcel<ParcelType>::readFields(CloudType& c)
{
    const bool valid = c.size();

    ParcelType::readFields(c);

    IOField<vector> f(c.fieldIOobject("f", IOobject::MUST_READ), valid);
    c.checkFieldIOobject(c, f);

    IOField<vector> angularMomentum
    (
        c.fieldIOobject("angularMomentum", IOobject::MUST_READ),
        valid
    );
    c.checkFieldIOobject(c, angularMomentum);

    IOField<vector> torque
    (
        c.fieldIOobject("torque", IOobject::MUST_READ),
        valid
    );
    c.checkFieldIOobject(c, torque);

    labelFieldCompactIOField collisionRecordsPairAccessed
    (
        c.fieldIOobject("collisionRecordsPairAccessed", IOobject::MUST_READ),
        valid
    );
    c.checkFieldFieldIOobject(c, collisionRecordsPairAccessed);

    labelFieldCompactIOField collisionRecordsPairOrigProcOfOther
    (
        c.fieldIOobject
        (
            "collisionRecordsPairOrigProcOfOther",
            IOobject::MUST_READ
        ),
        valid
    );
    c.checkFieldFieldIOobject(c, collisionRecordsPairOrigProcOfOther);

    labelFieldCompactIOField collisionRecordsPairOrigIdOfOther
    (
        c.fieldIOobject
        (
            "collisionRecordsPairOrigIdOfOther",
            IOobject::MUST_READ
        ),
        valid
    );
    c.checkFieldFieldIOobject(c, collisionRecordsPairOrigProcOfOther);

    pairDataFieldCompactIOField collisionRecordsPairData
    (
        c.fieldIOobject("collisionRecordsPairData", IOobject::MUST_READ),
        valid
    );
    c.checkFieldFieldIOobject(c, collisionRecordsPairData);

    labelFieldCompactIOField collisionRecordsWallAccessed
    (
        c.fieldIOobject("collisionRecordsWallAccessed", IOobject::MUST_READ),
        valid
    );
    c.checkFieldFieldIOobject(c, collisionRecordsWallAccessed);

    vectorFieldCompactIOField collisionRecordsWallPRel
    (
        c.fieldIOobject("collisionRecordsWallPRel", IOobject::MUST_READ),
        valid
    );
    c.checkFieldFieldIOobject(c, collisionRecordsWallPRel);

    wallDataFieldCompactIOField collisionRecordsWallData
    (
        c.fieldIOobject("collisionRecordsWallData", IOobject::MUST_READ),
        valid
    );
    c.checkFieldFieldIOobject(c, collisionRecordsWallData);

    label i = 0;

    for (CollidingParcel<ParcelType>& p : c)
    {
        p.f_ = f[i];
        p.angularMomentum_ = angularMomentum[i];
        p.torque_ = torque[i];

        p.collisionRecords_ = collisionRecordList
        (
            collisionRecordsPairAccessed[i],
            collisionRecordsPairOrigProcOfOther[i],
            collisionRecordsPairOrigIdOfOther[i],
            collisionRecordsPairData[i],
            collisionRecordsWallAccessed[i],
            collisionRecordsWallPRel[i],
            collisionRecordsWallData[i]
        );

        ++i;
    }
}


template<class ParcelType>
template<class CloudType>
void Foam::CollidingParcel<ParcelType>::writeFields(const CloudType& c)
{
    ParcelType::writeFields(c);

    const label np = c.size();
    const bool valid = np;


    IOField<vector> f(c.fieldIOobject("f", IOobject::NO_READ), np);
    IOField<vector> angMom
    (
        c.fieldIOobject("angularMomentum", IOobject::NO_READ),
        np
    );
    IOField<vector> torque(c.fieldIOobject("torque", IOobject::NO_READ), np);

    labelFieldCompactIOField collisionRecordsPairAccessed
    (
        c.fieldIOobject("collisionRecordsPairAccessed", IOobject::NO_READ),
        np
    );
    labelFieldCompactIOField collisionRecordsPairOrigProcOfOther
    (
        c.fieldIOobject
        (
            "collisionRecordsPairOrigProcOfOther",
            IOobject::NO_READ
        ),
        np
    );
    labelFieldCompactIOField collisionRecordsPairOrigIdOfOther
    (
        c.fieldIOobject("collisionRecordsPairOrigIdOfOther", IOobject::NO_READ),
        np
    );
    pairDataFieldCompactIOField collisionRecordsPairData
    (
        c.fieldIOobject("collisionRecordsPairData", IOobject::NO_READ),
        np
    );
    labelFieldCompactIOField collisionRecordsWallAccessed
    (
        c.fieldIOobject("collisionRecordsWallAccessed", IOobject::NO_READ),
        np
    );
    vectorFieldCompactIOField collisionRecordsWallPRel
    (
        c.fieldIOobject("collisionRecordsWallPRel", IOobject::NO_READ),
        np
    );
    wallDataFieldCompactIOField collisionRecordsWallData
    (
        c.fieldIOobject("collisionRecordsWallData", IOobject::NO_READ),
        np
    );

    label i = 0;
    for (const CollidingParcel<ParcelType>& p : c)
    {
        f[i] = p.f();
        angMom[i] = p.angularMomentum();
        torque[i] = p.torque();

        collisionRecordsPairAccessed[i] = p.collisionRecords().pairAccessed();
        collisionRecordsPairOrigProcOfOther[i] =
            p.collisionRecords().pairOrigProcOfOther();
        collisionRecordsPairOrigIdOfOther[i] =
            p.collisionRecords().pairOrigIdOfOther();
        collisionRecordsPairData[i] = p.collisionRecords().pairData();
        collisionRecordsWallAccessed[i] = p.collisionRecords().wallAccessed();
        collisionRecordsWallPRel[i] = p.collisionRecords().wallPRel();
        collisionRecordsWallData[i] = p.collisionRecords().wallData();

        ++i;
    }

    f.write(valid);
    angMom.write(valid);
    torque.write(valid);

    collisionRecordsPairAccessed.write(valid);
    collisionRecordsPairOrigProcOfOther.write(valid);
    collisionRecordsPairOrigIdOfOther.write(valid);
    collisionRecordsPairData.write(valid);
    collisionRecordsWallAccessed.write(valid);
    collisionRecordsWallPRel.write(valid);
    collisionRecordsWallData.write(valid);
}


template<class ParcelType>
void Foam::CollidingParcel<ParcelType>::writeProperties
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

    writeProp("f", f_);
    writeProp("angularMomentum", angularMomentum_);
    writeProp("torque", torque_);
    //writeProp("collisionRecords", collisionRecords_);

    #undef writeProp
}


template<class ParcelType>
template<class CloudType>
void Foam::CollidingParcel<ParcelType>::readObjects
(
    CloudType& c,
    const objectRegistry& obr
)
{
    ParcelType::readObjects(c, obr);

    if (!c.size()) return;

    const auto& f = cloud::lookupIOField<vector>("f", obr);
    const auto& angMom = cloud::lookupIOField<vector>("angularMomentum", obr);
    const auto& torque = cloud::lookupIOField<vector>("torque", obr);

    label i = 0;
    for (CollidingParcel<ParcelType>& p : c)
    {
        p.f_ = f[i];
        p.angularMomentum_ = angMom[i];
        p.torque_ = torque[i];

        ++i;
    }
}


template<class ParcelType>
template<class CloudType>
void Foam::CollidingParcel<ParcelType>::writeObjects
(
    const CloudType& c,
    objectRegistry& obr
)
{
    ParcelType::writeObjects(c, obr);

    const label np = c.size();

    auto& f = cloud::createIOField<vector>("f", np, obr);
    auto& angMom = cloud::createIOField<vector>("angularMomentum", np, obr);
    auto& torque = cloud::createIOField<vector>("torque", np, obr);

    label i = 0;
    for (const CollidingParcel<ParcelType>& p : c)
    {
        f[i] = p.f();
        angMom[i] = p.angularMomentum();
        torque[i] = p.torque();

        ++i;
    }
}


// * * * * * * * * * * * * * * * IOstream Operators  * * * * * * * * * * * * //

template<class ParcelType>
Foam::Ostream& Foam::operator<<
(
    Ostream& os,
    const CollidingParcel<ParcelType>& p
)
{
    if (os.format() == IOstreamOption::ASCII)
    {
        os  << static_cast<const ParcelType&>(p)
            << token::SPACE << p.f_
            << token::SPACE << p.angularMomentum_
            << token::SPACE << p.torque_
            << token::SPACE << p.collisionRecords_;
    }
    else
    {
        os  << static_cast<const ParcelType&>(p);
        os.write
        (
            reinterpret_cast<const char*>(&p.f_),
            CollidingParcel<ParcelType>::sizeofFields
        );
        os  << p.collisionRecords();
    }

    os.check(FUNCTION_NAME);
    return os;
}


// ************************************************************************* //
