/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2016 OpenFOAM Foundation
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

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class CloudType>
inline const Foam::SprayCloud<CloudType>&
Foam::SprayCloud<CloudType>::cloudCopy() const
{
    return *cloudCopyPtr_;
}


template<class CloudType>
inline const Foam::AtomizationModel<Foam::SprayCloud<CloudType>>&
Foam::SprayCloud<CloudType>::atomization() const
{
    return atomizationModel_;
}


template<class CloudType>
inline Foam::AtomizationModel<Foam::SprayCloud<CloudType>>&
Foam::SprayCloud<CloudType>::atomization()
{
    return *atomizationModel_;
}


template<class CloudType>
inline const Foam::BreakupModel<Foam::SprayCloud<CloudType>>&
Foam::SprayCloud<CloudType>::breakup() const
{
    return breakupModel_;
}


template<class CloudType>
inline Foam::BreakupModel<Foam::SprayCloud<CloudType>>&
Foam::SprayCloud<CloudType>::breakup()
{
    return *breakupModel_;
}


template<class CloudType>
inline Foam::scalar Foam::SprayCloud<CloudType>::averageParcelMass() const
{
    return averageParcelMass_;
}


template<class CloudType>
inline Foam::scalar Foam::SprayCloud<CloudType>::penetration
(
    const scalar fraction
) const
{
    if ((fraction < 0) || (fraction > 1))
    {
        FatalErrorInFunction
            << "fraction should be in the range 0 < fraction < 1"
            << exit(FatalError);
    }

    const label nParcel = this->size();
    const globalIndex globalParcels(nParcel);
    const label nTotParcel = globalParcels.totalSize();

    if (nTotParcel == 0)
    {
        return 0;
    }

    // lists of parcels mass and distance from initial injection point
    List<scalar> mass(nParcel);
    List<scalar> dist(nParcel);

    scalar mTotal = 0;
    {
        label i = 0;
        for (const parcelType& p : *this)
        {
            scalar m = p.nParticle()*p.mass();
            scalar d = mag(p.position() - p.position0());
            mTotal += m;

            mass[i] = m;
            dist[i] = d;
            ++i;
        }
    }
    // Total mass across all processors
    reduce(mTotal, sumOp<scalar>());

    scalar distance = 0;
    globalParcels.gatherInplace(mass);
    globalParcels.gatherInplace(dist);

    if (Pstream::master())
    {
        if (nTotParcel == 1)
        {
            distance = dist[0];
        }
        else
        {
            // Distances - sored into ascending order
            // Masses - leave unsorted

            const SortList<scalar> sortedDist(dist);

            const scalar mLimit = fraction*mTotal;
            const labelList& indices = sortedDist.indices();

            if (mLimit > (mTotal - mass[indices.last()]))
            {
                distance = sortedDist.last();
            }
            else
            {
                // assuming that 'fraction' is generally closer to 1 than 0,
                // loop through in reverse distance order
                const scalar mThreshold = (1.0 - fraction)*mTotal;
                scalar mCurrent = 0;
                label i0 = 0;

                forAllReverse(indices, i)
                {
                    label indI = indices[i];

                    mCurrent += mass[indI];

                    if (mCurrent > mThreshold)
                    {
                        i0 = i;
                        break;
                    }
                }

                if (i0 == indices.size() - 1)
                {
                    distance = sortedDist.last();
                }
                else
                {
                    // linearly interpolate to determine distance
                    scalar alpha = (mCurrent - mThreshold)/mass[indices[i0]];
                    distance =
                    (
                        sortedDist[i0]
                      + alpha*(sortedDist[i0+1] - sortedDist[i0])
                    );
                }
            }
        }
    }

    Pstream::broadcast(distance);

    return distance;
}


// ************************************************************************* //
