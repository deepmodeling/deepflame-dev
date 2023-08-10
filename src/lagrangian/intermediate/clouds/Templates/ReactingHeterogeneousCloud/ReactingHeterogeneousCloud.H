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
    Foam::ReactingHeterogeneousCloud

Group
    grpLagrangianIntermediateClouds

Description
    Templated base class for reacting heterogeneous cloud

    - Adds to reacting cloud:
      - Heterogeneous reaction model

SourceFiles
    ReactingHeterogeneousCloudI.H
    ReactingHeterogeneousCloud.C

\*---------------------------------------------------------------------------*/

#ifndef ReactingHeterogeneousCloud_H
#define ReactingHeterogeneousCloud_H

#include "reactingHeterogeneousCloud.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Forward declaration of classes
template<class CloudType>
class HeterogeneousReactingModel;


/*---------------------------------------------------------------------------*\
                      Class ReactingHeterogeneousCloud Declaration
\*---------------------------------------------------------------------------*/

template<class CloudType>
class ReactingHeterogeneousCloud
:
    public CloudType,
    public reactingHeterogeneousCloud
{
public:

    // Public typedefs

        //- Type of cloud this cloud was instantiated for
        typedef CloudType cloudType;

        //- Type of parcel the cloud was instantiated for
        typedef typename CloudType::particleType parcelType;

        //- Convenience typedef for this cloud type
        typedef ReactingHeterogeneousCloud<CloudType>
            reactingHeterogeneousCloudType;


private:

    // Private Data

        //- Cloud copy pointer
        autoPtr<ReactingHeterogeneousCloud<CloudType>> cloudCopyPtr_;


    // Private Member Functions

        //- No copy construct
        ReactingHeterogeneousCloud(const ReactingHeterogeneousCloud&) = delete;

        //- No copy assignment
        void operator=(const ReactingHeterogeneousCloud&) = delete;


protected:

    // Protected Data

        // References to the cloud sub-models

            //- Heterogeneous reaction model
            autoPtr
            <
                HeterogeneousReactingModel
                <
                    ReactingHeterogeneousCloud<CloudType>
                >
            > heterogeneousReactionModel_;


    // Protected Member Functions

        // Initialisation

            //- Set cloud sub-models
            void setModels();


        // Cloud evolution functions

            //- Reset state of cloud
            void cloudReset(ReactingHeterogeneousCloud<CloudType>& c);


public:

    // Constructors

        //- Construct given carrier gas fields
        ReactingHeterogeneousCloud
        (
            const word& cloudName,
            const volScalarField& rho,
            const volVectorField& U,
            const dimensionedVector& g,
            const SLGThermo& thermo,
            bool readFields = true
        );

        //- Copy constructor with new name
        ReactingHeterogeneousCloud
        (
            ReactingHeterogeneousCloud<CloudType>& c, const word& name
        );

        //- Copy constructor with new name - creates bare cloud
        ReactingHeterogeneousCloud
        (
            const fvMesh& mesh,
            const word& name,
            const ReactingHeterogeneousCloud<CloudType>& c
        );

        //- Construct and return clone based on (this) with new name
        virtual autoPtr<Cloud<parcelType>> clone(const word& name)
        {
            return autoPtr<Cloud<parcelType>>
            (
                new ReactingHeterogeneousCloud(*this, name)
            );
        }

        //- Construct and return bare clone based on (this) with new name
        virtual autoPtr<Cloud<parcelType>> cloneBare(const word& name) const
        {
            return autoPtr<Cloud<parcelType>>
            (
                new ReactingHeterogeneousCloud(this->mesh(), name, *this)
            );
        }


    //- Destructor
    virtual ~ReactingHeterogeneousCloud() = default;


    // Member Functions

        // Access

            //- Return a reference to the cloud copy
            inline const ReactingHeterogeneousCloud& cloudCopy() const;

            //- Return progress variable dimension
            inline label nF() const;


        // Sub-models

            //- Return reference to model
            inline const HeterogeneousReactingModel
            <
                ReactingHeterogeneousCloud<CloudType>
            >& heterogeneousReaction() const;


            inline HeterogeneousReactingModel
            <
                ReactingHeterogeneousCloud<CloudType>
            >& heterogeneousReaction();


        // Cloud evolution functions

            //- Set parcel thermo properties
            void setParcelThermoProperties
            (
                parcelType& parcel,
                const scalar lagrangianDt
            );

            //- Check parcel properties
            void checkParcelProperties
            (
                parcelType& parcel,
                const scalar lagrangianDt,
                const bool fullyDescribed
            );

            //- Store the current cloud state
            void storeState();

            //- Reset the current cloud to the previously stored state
            void restoreState();

            //- Evolve the cloud
            void evolve();


        // Mapping

            //- Remap the cells of particles corresponding to the
            //  mesh topology change with a default tracking data object
            virtual void autoMap(const mapPolyMesh&);


        // I-O

            //- Print cloud information
            void info();

            //- Read particle fields as objects from the obr registry
            virtual void readObjects(const objectRegistry& obr);

            //- Write the field data for the cloud
            virtual void writeFields() const;

            //- Write particle fields as objects into the obr registry
            virtual void writeObjects(objectRegistry& obr) const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#include "ReactingHeterogeneousCloudI.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
    #include "ReactingHeterogeneousCloud.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
