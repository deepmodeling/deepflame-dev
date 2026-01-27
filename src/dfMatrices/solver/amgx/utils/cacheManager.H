/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2020 OpenCFD Ltd.
    Copyright (C) 2020 Simone Bna
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
    Foam::cacheManager

Description
    The manager class for the caching of matrix and preconditioner.

    An Example is:

    \verbatim
    petsc
    {
        ...
        caching
        {
            matrix
            {
                update  always;  // (default: always)
            }

            preconditioner
            {
                update  periodic;

                periodicCoeffs
                {
                    frequency   3;
                }
            }
        }
    }
    \endverbatim

SourceFiles
    cacheManager.C

\*---------------------------------------------------------------------------*/

#ifndef foamCacheManager_H
#define foamCacheManager_H

//#include "Enum.H"
//#include "clockValue.H"
//#include "dictionary.H"
#include "NamedEnum.H"
#include "dictionary.H"
#include <chrono>

typedef std::chrono::high_resolution_clock::duration value_type;
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace AmgxUtils
{

/*---------------------------------------------------------------------------*\
                           Class Caching Declaration
\*---------------------------------------------------------------------------*/

struct Caching
{
    // Public Data Types

        //- Caching update types
        //enum updateTypes
        //{
        //   Always,         //!< "always" update every time-step [default]
        //   Periodic,       //!< "periodic" update at given period
        //   Adaptive,       //!< "adaptive" update scheme
        //   Never,          //!< "never" update (or "none")
        //};


        //- Names for the update types
        //static const Enum<updateTypes> updateTypeNames_;
        //static const NamedEnum<updateTypes> updateTypeNames_;
        //static const Foam::NamedEnum<updateTypes, 4> updateTypeNames_;

    // Member Data

        //- Caching update
        //updateTypes updateType_;

        //- Cache update frequency (for Periodic)
        int updateFreq_;
        word updateType_;
        //- Elapsed time (s) for current iteration
        double timeIter;
        int iter_;
        //- Elapsed time (s) for first iteration
        double time0;

        //- Elapsed time (s) for second iteration
        double time1;

        //- Timer value
        //clockValue timer_;
        value_type timer_;

    // Constructors

        //- Default construct. Always update (no caching).
        Caching()
        :
           updateType_{"always"},
           updateFreq_{1},
           timeIter{0},
           time0{0},
           time1{0}
        {timer_ = value_type::zero();}

    //member functions
        //- The caching type
        word type() const noexcept
        {
            return updateType_;
        }

        //- True if caching type is Adaptive
        bool isAdaptive() const noexcept
        {
            return "adaptive" == updateType_;
        }

        //- True if caching type is Periodic
        bool isPeriodic() const noexcept
        {
            return "periodic" == updateType_;
        }

        void beginTimes()
        {
            // timer_.update();
            //Returns a duration representing the amount of time between *this and the clock's epoch.
            timer_ = std::chrono::high_resolution_clock::now().time_since_epoch();
        }
        void init(const word& key, const dictionary& dict)
        {   
            iter_ = 0;
            dictionary cacheDict = dict.subOrEmptyDict(key);
            updateType_ = cacheDict.lookupOrDefault("update",word("always"));

            if ("periodic" == updateType_)
            {
                dictionary coeffs(cacheDict.subOrEmptyDict("periodicCoeffs"));
                updateFreq_ = coeffs.lookupOrDefault("frequency", 1);
            }
    }
 


        //- Update timing for iterations
    void updateTimes()
    {
        // Elapsed timing interval (s)
        // timeIter_ = timer_.elapsed();
        value_type timer_tmp;
        timer_tmp = std::chrono::high_resolution_clock::now().time_since_epoch() - timer_;
        timeIter = (double(timer_tmp.count()) * value_type::period::num) / value_type::period::den;

        if (iter_ == 0)
        {
            time0 = timeIter;
        } 
        else if (iter_ == 1)
        {
            time1 = timeIter;
        }
    }

        //- Check if an update is required
        bool needsUpdate()
        {
            // Default: Always update
            bool need = true;

            // switch (updateType_)
            // {
                // case "never":
                if (updateType_ == "never")
                {
                    need = false;
                    // break;
                    return need;
                }

                // case "always":
                if (updateType_ == "always")
                {
                    // break;
                    return need;
                }

                // case "periodic":
                if (updateType_ == "periodic")
                {
                    if
                    (
                        updateFreq_ <= 1
                     || (iter_ % updateFreq_) == 0
                    )
                    {
                        iter_ = 0;
                    }
                    else
                    {
                        need = false;
                    }
                    // break;
                    return need;
                }

                // case "adaptive":
                if (updateType_ == "adaptive")
                {
                    if (iter_ > 3)  // Need at least three iterations
                    {
                        const double ratio0 = time0 / timeIter;

                        const double ratio1 = time1 / timeIter;

                        const int nsteps =
                            min(1e5, ratio0 * (1. / mag(1. - ratio1 + 1e-6)));

                        if (iter_ >= nsteps)
                        {
                            iter_ = 0;
                        }
                        else
                        {
                            need = false;
                        }
                    }
                    // break;
                    return need;
                }
            // }

            // return need;
        }    
};

} // End namespace PetscUtils


/*---------------------------------------------------------------------------*\
                     Class cacheManager Declaration
\*---------------------------------------------------------------------------*/

class cacheManager
{
    // Private Data

        //- The current (relative) iterations for matrix
        label miter;

        //- The current (relative) iterations for preconditioner
        label piter;

        AmgxUtils::Caching matrixCaching;
        //PetscUtils::Caching precondCaching;


public:

    // Constructors

        //- Default construct
        cacheManager()
        :
            miter{0},
            piter{0}
        {}


    //- Destructor
    ~cacheManager() = default;


    void init(const dictionary& dict)
    {
        matrixCaching.init("matrix", dict);
        //precondCaching.init("preconditioner", dict);
    }


    // Member Functions

    bool needsMatrixUpdate()
    {
        return matrixCaching.needsUpdate();
    }

    //bool needsPrecondUpdate()
    //{
    //    return precondCaching.needsUpdate();
    //}

    void eventBegin()
    {
        //if (precondCaching.isAdaptive())
        //{
            // Begin timing interval
            //precondCaching.timer_.update();
        //    precondCaching.timer_=std::chrono::high_resolution_clock::now().time_since_epoch();
        //}
    }

    void eventEnd()
    {
        //if (precondCaching.isPeriodic())
        //{
            // Elapsed timing interval (s)
        //    precondCaching.updateTimes();

        //    if (piter == 0)
        //    {
        //        precondCaching.time0 = precondCaching.timeIter;
        //    }
        //    else if (piter == 1)
        //    {
        //       precondCaching.time1 = precondCaching.timeIter;
        //    }
        //}

        //if
        //(
        //    precondCaching.isPeriodic()
        // || precondCaching.isAdaptive()
        //)
        //{
        //    ++piter;
        //}
        if
        (
            matrixCaching.isPeriodic()
        )
        {
            ++miter;
        }
    }


// private:

//     bool needsUpdate(const PetscUtils::Caching& caching, label& iter) const
//     {
//         // Default: Always update
//         bool need = true;

//         switch (caching.updateType_)
//         {
//             case PetscUtils::Caching::Never:
//             {
//                 need = false;
//                 break;
//             }

//             case PetscUtils::Caching::Always:
//             {
//                 break;
//             }

//             case PetscUtils::Caching::Periodic:
//             {
//                 if
//                 (
//                     caching.updateFreq_ <= 1
//                  || (iter % caching.updateFreq_) == 0
//                 )
//                 {
//                     iter = 0;
//                 }
//                 else
//                 {
//                     need = false;
//                 }
//                 break;
//             }

//             case PetscUtils::Caching::Adaptive:
//             {
//                 if (iter > 3) // we need at least three times
//                 {
//                     const double ratio0 =
//                         caching.time0 / caching.timeIter;

//                     const double ratio1 =
//                         caching.time1 / caching.timeIter;

//                     const int nsteps =
//                         min(1e5, ratio0 * (1. / mag(1. - ratio1 + 1e-6)));

//                     if (iter >= nsteps)
//                     {
//                         iter = 0;
//                     }
//                     else
//                     {
//                         need = false;
//                     }
//                 }
//                 break;
//             }
//         }

//         return need;
//     }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
