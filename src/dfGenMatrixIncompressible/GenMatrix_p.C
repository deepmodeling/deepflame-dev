#include "GenFvMatrix.H"
#include "multivariateGaussConvectionScheme.H"
#include "gaussConvectionScheme.H"
#include "snGradScheme.H"
#include "linear.H"
#include "orthogonalSnGrad.H"

#include "pimpleControl.H"
#include "fvOptions.H"
#include "turbulentFluidThermoModel.H"

#include "incompressibleTwoPhaseMixture.H"
#include "immiscibleIncompressibleTwoPhaseMixture.H"
#include "turbulentTransportModel.H"

namespace Foam{

tmp<fvVectorMatrix>
GenMatrix_p(

){
































    tmp<fvScalarMatrix> tfvm
    (
        new fvScalarMatrix
        (
            (fvm::laplacian(rAUf, p_rgh))
        == 
            (fvc::div(phiHbyA))
        )
    );

    return tfvm;



}


}