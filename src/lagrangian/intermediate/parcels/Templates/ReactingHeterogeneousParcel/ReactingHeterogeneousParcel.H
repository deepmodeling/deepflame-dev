/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2018-2019 OpenCFD Ltd.
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

Class
    Foam::ReactingHeterogeneousParcel

Group
    grpLagrangianIntermediateParcels

Description
    Reacting heterogeneous Parcel

SourceFiles
    ReactingHeterogeneousParcelI.H
    ReactingHeterogeneousParcel.C
    ReactingHeterogeneousParcelIO.C

\*---------------------------------------------------------------------------*/

#ifndef ReactingHeterogeneousParcel_H
#define ReactingHeterogeneousParcel_H

#include "demandDrivenEntry.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

template<class ParcelType>
class ReactingHeterogeneousParcel;

template<class ParcelType>
Ostream& operator<<
(
    Ostream&,
    const ReactingHeterogeneousParcel<ParcelType>&
);

/*---------------------------------------------------------------------------*\
                 Class ReactingHeterogeneousParcel Declaration
\*---------------------------------------------------------------------------*/

template<class ParcelType>
class ReactingHeterogeneousParcel
:
    public ParcelType
{
public:

    //- Size in bytes of the fields
    static const std::size_t sizeofFields;

    //- Class to hold reacting particle constant properties
    class constantProperties
    :
        public ParcelType::constantProperties
    {
        // Private data

            //- Fraction of enthalpy retained by parcel due to surface
            //  reactions
            demandDrivenEntry<scalar> hRetentionCoeff_;

    public:

        // Constructors

            //- Null constructor
            constantProperties();

            //- Copy constructor
            constantProperties(const constantProperties& cp);

            //- Construct from dictionary
            constantProperties(const dictionary& parentDict);


        // Access

            //- Return const access to the fraction of enthalpy retained by
            //  parcel due to surface reactions
            inline scalar hRetentionCoeff() const;
    };


    //- Use base tracking data
    typedef typename ParcelType::trackingData trackingData;


private:

    // Private Member Functions

        //- Return the mixture effective specific heat capacity
        template<class TrackCloudType>
        scalar CpEff
        (
            TrackCloudType& cloud,
            trackingData& td,
            const scalar p,
            const scalar T,
            const label idS
        ) const;

        //- Return the mixture effective sensible enthalpy
        template<class TrackCloudType>
        scalar HsEff
        (
            TrackCloudType& cloud,
            trackingData& td,
            const scalar p,
            const scalar T,
            const label idS
        ) const;

        //- Return the mixture effective latent heat
        template<class TrackCloudType>
        scalar LEff
        (
            TrackCloudType& cloud,
            trackingData& td,
            const scalar p,
            const scalar T,
            const label idS
        ) const;


protected:

    // Protected data

        // Parcel properties

            //- Progress variables for reactions
            scalarField F_;

            //- Flag to identify if the particle can devolatilise and combust
            //  Combustion possible only after volatile content falls below
            //  threshold value.  States include:
            //  0 = can combust but can change
            //  1 = can devolatilise, can combust
            // -1 = cannot devolatilise or combust, and cannot change
            label canCombust_;


    // Protected Member Functions

        //- Return change of volume due to mass exchange
        template<class TrackCloudType>
        scalar updatedDeltaVolume
        (
            TrackCloudType& cloud,
            const scalarField& dMass,
            const scalar p,
            const scalar T
        );


        //- Calculate surface reactions
        template<class TrackCloudType>
        void calcHeterogeneousReactions
        (
            TrackCloudType& cloud,
            trackingData& td,
            const scalar dt,           // timestep
            const scalar Res,          // Re
            const scalar nu,           // nu
            const scalar d,            // diameter
            const scalar T,            // temperature
            const scalar mass,         // mass
            const label canCombust,    // 'can combust' flag
            const scalar N,            // flux of species emitted from particle
            scalar& NCpW,
            const scalarField& YSolid, // solid-phase mass fractions
            scalarField& F,            // progress of each reaction
            scalarField& dMassSRSolid, // solid-phase mass transfer - local
            scalarField& dMassSRCarrier, // carrier phase mass transfer
            scalar& Sh,                // explicit particle enthalpy source
            scalar& dhsTrans           // sensible enthalpy transfer to carrier
        ) const;


public:

    // Static data members

        //- Runtime type information
        TypeName("ReactingHeterogeneousParcel");

        //- String representation of properties
        AddToPropertyList
        (
            ParcelType,
          + " nReactions(F1..FN)"
        );


    // Constructors

        //- Construct from mesh, position and topology
        //  Other properties initialised as null
        inline ReactingHeterogeneousParcel
        (
            const polyMesh& mesh,
            const barycentric& coordinates,
            const label celli,
            const label tetFacei,
            const label tetPti
        );

        //- Construct from a position and a cell, searching for the rest of the
        //  required topology. Other properties are initialised as null.
        inline ReactingHeterogeneousParcel
        (
            const polyMesh& mesh,
            const vector& position,
            const label celli
        );

        //- Construct from components
        inline ReactingHeterogeneousParcel
        (
            const polyMesh& mesh,
            const barycentric& coordinates,
            const label celli,
            const label tetFacei,
            const label tetPti,
            const label typeId,
            const scalar nParticle0,
            const scalar d0,
            const scalar dTarget0,
            const vector& U0,
            const vector& f0,
            const vector& angularMomentum0,
            const vector& torque0,
            const scalarField& Y,
            const scalarField& F,
            const constantProperties& constProps
        );

        //- Construct from Istream
        ReactingHeterogeneousParcel
        (
            const polyMesh& mesh,
            Istream& is,
            bool readFields = true,
            bool newFormat = true
        );

        //- Construct as a copy
        ReactingHeterogeneousParcel(const ReactingHeterogeneousParcel& p);

        //- Construct as a copy
        ReactingHeterogeneousParcel
        (
            const ReactingHeterogeneousParcel& p,
            const polyMesh& mesh
        );

        //- Construct and return a (basic particle) clone
        virtual autoPtr<particle> clone() const
        {
            return autoPtr<particle>(new ReactingHeterogeneousParcel(*this));
        }

        //- Construct and return a (basic particle) clone
        virtual autoPtr<particle> clone(const polyMesh& mesh) const
        {
            return autoPtr<particle>
            (
                new ReactingHeterogeneousParcel(*this, mesh)
            );
        }

        //- Factory class to read-construct particles used for
        //  parallel transfer
        class iNew
        {
            const polyMesh& mesh_;

        public:

            iNew(const polyMesh& mesh)
            :
                mesh_(mesh)
            {}

            autoPtr<ReactingHeterogeneousParcel<ParcelType>> operator()
            (
                Istream& is
            ) const
            {
                return autoPtr<ReactingHeterogeneousParcel<ParcelType>>
                (
                    new ReactingHeterogeneousParcel<ParcelType>
                        (mesh_, is, true)
                );
            }
        };


    // Member Functions

        // Access

            //- Return const access to F
            inline const scalarField& F() const;

            //- Return const access to the canCombust flag
            inline label canCombust() const;


        // Edit

             //- Return access to F
            inline scalarField& F();

            //- Return access to the canCombust flag
            inline label& canCombust();


        // Main calculation loop


            //- Update parcel properties over the time interval
            template<class TrackCloudType>
            void calc
            (
                TrackCloudType& cloud,
                trackingData& td,
                const scalar dt
            );


        // I-O

            //- Read - composition supplied
            template<class CloudType, class CompositionType>
            static void readFields
            (
                CloudType& c,
                const CompositionType& compModel
            );

            //- Read - no composition
            template<class CloudType>
            static void readFields(CloudType& c);

            //- Write - composition supplied
            template<class CloudType, class CompositionType>
            static void writeFields
            (
                const CloudType& c,
                const CompositionType& compModel
            );

            //- Read - no composition
            template<class CloudType>
            static void writeFields(const CloudType& c);

            //- Write individual parcel properties to stream
            void writeProperties
            (
                Ostream& os,
                const wordRes& filters,
                const word& delim,
                const bool namesOnly
            ) const;

            //- Read particle fields as objects from the obr registry
            //  - no composition
            template<class CloudType>
            static void readObjects
            (
                CloudType& c,
                const objectRegistry& obr
            );

            //- Read particle fields as objects from the obr registry
            template<class CloudType, class CompositionType>
            static void readObjects
            (
                CloudType& c,
                const CompositionType& compModel,
                const objectRegistry& obr
            );

            //- Write particle fields as objects into the obr registry
            //  - no composition
            template<class CloudType>
            static void writeObjects
            (
                const CloudType& c,
                objectRegistry& obr
            );

            //- Write particle fields as objects into the obr registry
            template<class CloudType, class CompositionType>
            static void writeObjects
            (
                const CloudType& c,
                const CompositionType& compModel,
                objectRegistry& obr
            );


    // Ostream Operator

        friend Ostream& operator<< <ParcelType>
        (
            Ostream&,
            const ReactingHeterogeneousParcel<ParcelType>&
        );
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#include "ReactingHeterogeneousParcelI.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
    #include "ReactingHeterogeneousParcel.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
