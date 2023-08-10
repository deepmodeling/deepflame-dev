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

#include "KinematicParcel.H"
#include "IOstreams.H"
#include "IOField.H"
#include "Cloud.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

template<class ParcelType>
Foam::string Foam::KinematicParcel<ParcelType>::propertyList_ =
    Foam::KinematicParcel<ParcelType>::propertyList();


template<class ParcelType>
const std::size_t Foam::KinematicParcel<ParcelType>::sizeofFields
(
    sizeof(KinematicParcel<ParcelType>)
  - offsetof(KinematicParcel<ParcelType>, active_)
);


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class ParcelType>
Foam::KinematicParcel<ParcelType>::KinematicParcel
(
    const polyMesh& mesh,
    Istream& is,
    bool readFields,
    bool newFormat
)
:
    ParcelType(mesh, is, readFields, newFormat),
    active_(false),
    typeId_(0),
    nParticle_(0.0),
    d_(0.0),
    dTarget_(0.0),
    U_(Zero),
    rho_(0.0),
    age_(0.0),
    tTurb_(0.0),
    UTurb_(Zero),
    UCorrect_(Zero)
{
    if (readFields)
    {
        if (is.format() == IOstreamOption::ASCII)
        {
            is  >> active_
                >> typeId_
                >> nParticle_
                >> d_
                >> dTarget_
                >> U_
                >> rho_
                >> age_
                >> tTurb_
                >> UTurb_
                >> UCorrect_;
        }
        else if (!is.checkLabelSize<>() || !is.checkScalarSize<>())
        {
            // Non-native label or scalar size

            is.beginRawRead();

            readRawLabel(is, &active_);
            readRawLabel(is, &typeId_);
            readRawScalar(is, &nParticle_);
            readRawScalar(is, &d_);
            readRawScalar(is, &dTarget_);
            readRawScalar(is, U_.data(), vector::nComponents);
            readRawScalar(is, &rho_);
            readRawScalar(is, &age_);
            readRawScalar(is, &tTurb_);
            readRawScalar(is, UTurb_.data(), vector::nComponents);
            readRawScalar(is, UCorrect_.data(), vector::nComponents);

            is.endRawRead();
        }
        else
        {
            is.read(reinterpret_cast<char*>(&active_), sizeofFields);
        }
    }

    is.check(FUNCTION_NAME);
}


template<class ParcelType>
template<class CloudType>
void Foam::KinematicParcel<ParcelType>::readFields(CloudType& c)
{
    const bool valid = c.size();

    ParcelType::readFields(c);

    IOField<label> active
    (
        c.fieldIOobject("active", IOobject::MUST_READ),
        valid
    );
    c.checkFieldIOobject(c, active);

    IOField<label> typeId
    (
        c.fieldIOobject("typeId", IOobject::MUST_READ),
        valid
    );
    c.checkFieldIOobject(c, typeId);

    IOField<scalar> nParticle
    (
        c.fieldIOobject("nParticle", IOobject::MUST_READ),
        valid
    );
    c.checkFieldIOobject(c, nParticle);

    IOField<scalar> d
    (
        c.fieldIOobject("d", IOobject::MUST_READ),
        valid
    );
    c.checkFieldIOobject(c, d);

    IOField<scalar> dTarget
    (
        c.fieldIOobject("dTarget", IOobject::MUST_READ),
        valid
    );
    c.checkFieldIOobject(c, dTarget);

    IOField<vector> U
    (
        c.fieldIOobject("U", IOobject::MUST_READ),
        valid
    );
    c.checkFieldIOobject(c, U);

    IOField<scalar> rho
    (
        c.fieldIOobject("rho", IOobject::MUST_READ),
        valid
    );
    c.checkFieldIOobject(c, rho);

    IOField<scalar> age
    (
        c.fieldIOobject("age", IOobject::MUST_READ),
        valid
    );
    c.checkFieldIOobject(c, age);

    IOField<scalar> tTurb
    (
        c.fieldIOobject("tTurb", IOobject::MUST_READ),
        valid
    );
    c.checkFieldIOobject(c, tTurb);

    IOField<vector> UTurb
    (
        c.fieldIOobject("UTurb", IOobject::MUST_READ),
        valid
    );
    c.checkFieldIOobject(c, UTurb);

    IOField<vector> UCorrect
    (
        c.fieldIOobject("UCorrect", IOobject::MUST_READ),
        valid
    );
    c.checkFieldIOobject(c, UCorrect);

    label i = 0;

    for (KinematicParcel<ParcelType>& p : c)
    {
        p.active_ = active[i];
        p.typeId_ = typeId[i];
        p.nParticle_ = nParticle[i];
        p.d_ = d[i];
        p.dTarget_ = dTarget[i];
        p.U_ = U[i];
        p.rho_ = rho[i];
        p.age_ = age[i];
        p.tTurb_ = tTurb[i];
        p.UTurb_ = UTurb[i];
        p.UCorrect_ = UCorrect[i];

        ++i;
    }
}


template<class ParcelType>
template<class CloudType>
void Foam::KinematicParcel<ParcelType>::writeFields(const CloudType& c)
{
    ParcelType::writeFields(c);

    const label np = c.size();
    const bool valid = np;

    IOField<label> active(c.fieldIOobject("active", IOobject::NO_READ), np);
    IOField<label> typeId(c.fieldIOobject("typeId", IOobject::NO_READ), np);
    IOField<scalar> nParticle
    (
        c.fieldIOobject("nParticle", IOobject::NO_READ),
        np
    );
    IOField<scalar> d(c.fieldIOobject("d", IOobject::NO_READ), np);
    IOField<scalar> dTarget(c.fieldIOobject("dTarget", IOobject::NO_READ), np);
    IOField<vector> U(c.fieldIOobject("U", IOobject::NO_READ), np);
    IOField<scalar> rho(c.fieldIOobject("rho", IOobject::NO_READ), np);
    IOField<scalar> age(c.fieldIOobject("age", IOobject::NO_READ), np);
    IOField<scalar> tTurb(c.fieldIOobject("tTurb", IOobject::NO_READ), np);
    IOField<vector> UTurb(c.fieldIOobject("UTurb", IOobject::NO_READ), np);
    IOField<vector> UCorrect(c.fieldIOobject("UCorrect", IOobject::NO_READ), np);

    label i = 0;

    for (const KinematicParcel<ParcelType>& p : c)
    {
        active[i] = p.active();
        typeId[i] = p.typeId();
        nParticle[i] = p.nParticle();
        d[i] = p.d();
        dTarget[i] = p.dTarget();
        U[i] = p.U();
        rho[i] = p.rho();
        age[i] = p.age();
        tTurb[i] = p.tTurb();
        UTurb[i] = p.UTurb();
        UCorrect[i] = p.UCorrect();

        ++i;
    }

    active.write(valid);
    typeId.write(valid);
    nParticle.write(valid);
    d.write(valid);
    dTarget.write(valid);
    U.write(valid);
    rho.write(valid);
    age.write(valid);
    tTurb.write(valid);
    UTurb.write(valid);
    UCorrect.write(valid);
}


template<class ParcelType>
void Foam::KinematicParcel<ParcelType>::writeProperties
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

    writeProp("active", active_);
    writeProp("typeId", typeId_);
    writeProp("nParticle", nParticle_);
    writeProp("d", d_);
    writeProp("dTarget", dTarget_);
    writeProp("U", U_);
    writeProp("rho", rho_);
    writeProp("age", age_);
    writeProp("tTurb", tTurb_);
    writeProp("UTurb", UTurb_);
    writeProp("UCorrect", UCorrect_);

    #undef writeProp
}


template<class ParcelType>
template<class CloudType>
void Foam::KinematicParcel<ParcelType>::readObjects
(
    CloudType& c,
    const objectRegistry& obr
)
{
    ParcelType::readObjects(c, obr);

    if (!c.size()) return;

    const auto& active = cloud::lookupIOField<label>("active", obr);
    const auto& typeId = cloud::lookupIOField<label>("typeId", obr);
    const auto& nParticle = cloud::lookupIOField<scalar>("nParticle", obr);
    const auto& d = cloud::lookupIOField<scalar>("d", obr);
    const auto& dTarget = cloud::lookupIOField<scalar>("dTarget", obr);
    const auto& U = cloud::lookupIOField<vector>("U", obr);
    const auto& rho = cloud::lookupIOField<scalar>("rho", obr);
    const auto& age = cloud::lookupIOField<scalar>("age", obr);
    const auto& tTurb = cloud::lookupIOField<scalar>("tTurb", obr);
    const auto& UTurb = cloud::lookupIOField<vector>("UTurb", obr);
    const auto& UCorrect = cloud::lookupIOField<vector>("UCorrect", obr);

    label i = 0;

    for (KinematicParcel<ParcelType>& p : c)
    {
        p.active_ = active[i];
        p.typeId_ = typeId[i];
        p.nParticle_ = nParticle[i];
        p.d_ = d[i];
        p.dTarget_ = dTarget[i];
        p.U_ = U[i];
        p.rho_ = rho[i];
        p.age_ = age[i];
        p.tTurb_ = tTurb[i];
        p.UTurb_ = UTurb[i];
        p.UCorrect_ = UCorrect[i];

        ++i;
    }
}


template<class ParcelType>
template<class CloudType>
void Foam::KinematicParcel<ParcelType>::writeObjects
(
    const CloudType& c,
    objectRegistry& obr
)
{
    ParcelType::writeObjects(c, obr);

    const label np = c.size();

    auto& active = cloud::createIOField<label>("active", np, obr);
    auto& typeId = cloud::createIOField<label>("typeId", np, obr);
    auto& nParticle = cloud::createIOField<scalar>("nParticle", np, obr);
    auto& d = cloud::createIOField<scalar>("d", np, obr);
    auto& dTarget = cloud::createIOField<scalar>("dTarget", np, obr);
    auto& U = cloud::createIOField<vector>("U", np, obr);
    auto& rho = cloud::createIOField<scalar>("rho", np, obr);
    auto& age = cloud::createIOField<scalar>("age", np, obr);
    auto& tTurb = cloud::createIOField<scalar>("tTurb", np, obr);
    auto&& UTurb = cloud::createIOField<vector>("UTurb", np, obr);
    auto&& UCorrect = cloud::createIOField<vector>("UCorrect", np, obr);

    label i = 0;

    for (const KinematicParcel<ParcelType>& p : c)
    {
        active[i] = p.active();
        typeId[i] = p.typeId();
        nParticle[i] = p.nParticle();
        d[i] = p.d();
        dTarget[i] = p.dTarget();
        U[i] = p.U();
        rho[i] = p.rho();
        age[i] = p.age();
        tTurb[i] = p.tTurb();
        UTurb[i] = p.UTurb();
        UCorrect[i] = p.UCorrect();

        ++i;
    }
}


// * * * * * * * * * * * * * * * IOstream Operators  * * * * * * * * * * * * //

template<class ParcelType>
Foam::Ostream& Foam::operator<<
(
    Ostream& os,
    const KinematicParcel<ParcelType>& p
)
{
    if (os.format() == IOstreamOption::ASCII)
    {
        os  << static_cast<const ParcelType&>(p)
            << token::SPACE << bool(p.active())
            << token::SPACE << p.typeId()
            << token::SPACE << p.nParticle()
            << token::SPACE << p.d()
            << token::SPACE << p.dTarget()
            << token::SPACE << p.U()
            << token::SPACE << p.rho()
            << token::SPACE << p.age()
            << token::SPACE << p.tTurb()
            << token::SPACE << p.UTurb()
            << token::SPACE << p.UCorrect();
    }
    else
    {
        os  << static_cast<const ParcelType&>(p);
        os.write
        (
            reinterpret_cast<const char*>(&p.active_),
            KinematicParcel<ParcelType>::sizeofFields
        );
    }

    os.check(FUNCTION_NAME);
    return os;
}


// ************************************************************************* //
