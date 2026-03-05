/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2024 DeepFlame
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of DeepFlame.

    DeepFlame is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    DeepFlame is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with DeepFlame.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "dfHybridInterfaceFvPatchField.H"
#include "volFields.H"

// * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * * * //

template<class Type>
const Foam::mappedPatchBase&
Foam::dfHybridInterfaceFvPatchField<Type>::mapper() const
{
    if (!isA<mappedPatchBase>(this->patch().patch()))
    {
        FatalErrorInFunction
            << "dfHybridInterface boundary condition requires the underlying "
            << "polyPatch to be of type 'mappedPatch' or 'mappedWall'." << nl
            << "    Patch: " << this->patch().name() << nl
            << "    Patch type: " << this->patch().patch().type() << nl
            << exit(FatalError);
    }

    return refCast<const mappedPatchBase>(this->patch().patch());
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class Type>
Foam::dfHybridInterfaceFvPatchField<Type>::
dfHybridInterfaceFvPatchField
(
    const fvPatch& p,
    const DimensionedField<Type, volMesh>& iF
)
:
    mixedFvPatchField<Type>(p, iF),
    mode_("provider"),
    fieldName_(iF.name())
{
    this->refValue() = Zero;
    this->refGrad() = Zero;
    this->valueFraction() = 0.0;
}


template<class Type>
Foam::dfHybridInterfaceFvPatchField<Type>::
dfHybridInterfaceFvPatchField
(
    const fvPatch& p,
    const DimensionedField<Type, volMesh>& iF,
    const dictionary& dict
)
:
    mixedFvPatchField<Type>(p, iF),
    mode_(dict.lookup("mode")),
    fieldName_(dict.lookupOrDefault<word>("fieldName", iF.name()))
{
    // Validate mode
    if (mode_ != "provider" && mode_ != "receiver")
    {
        FatalIOErrorInFunction(dict)
            << "Invalid mode '" << mode_ << "' for dfHybridInterface BC." << nl
            << "    Valid modes: provider, receiver" << nl
            << exit(FatalIOError);
    }

    if (mode_ == "provider")
    {
        // Provider (zeroGradient): initialise face values from internal field.
        fvPatchField<Type>::operator=(this->patchInternalField());
        this->refValue() = this->patchInternalField();
        this->refGrad() = Zero;
        this->valueFraction() = 0.0;
    }
    else
    {
        // Receiver (fixedValue): on restart the "value" entry holds the
        // true coupled field from the previous time step
        if (dict.found("value"))
        {
            fvPatchField<Type>::operator=
            (
                Field<Type>("value", dict, p.size())
            );
        }
        else
        {
            fvPatchField<Type>::operator=(this->patchInternalField());
        }
        this->refValue() = *this;
        this->refGrad() = Zero;
        this->valueFraction() = 1.0;
    }
}


template<class Type>
Foam::dfHybridInterfaceFvPatchField<Type>::
dfHybridInterfaceFvPatchField
(
    const dfHybridInterfaceFvPatchField<Type>& ptf,
    const fvPatch& p,
    const DimensionedField<Type, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    mixedFvPatchField<Type>(ptf, p, iF, mapper),
    mode_(ptf.mode_),
    fieldName_(iF.name())
{}


template<class Type>
Foam::dfHybridInterfaceFvPatchField<Type>::
dfHybridInterfaceFvPatchField
(
    const dfHybridInterfaceFvPatchField<Type>& ptf
)
:
    mixedFvPatchField<Type>(ptf),
    mode_(ptf.mode_),
    fieldName_(ptf.fieldName_)
{}


template<class Type>
Foam::dfHybridInterfaceFvPatchField<Type>::
dfHybridInterfaceFvPatchField
(
    const dfHybridInterfaceFvPatchField<Type>& ptf,
    const DimensionedField<Type, volMesh>& iF
)
:
    mixedFvPatchField<Type>(ptf, iF),
    mode_(ptf.mode_),
    fieldName_(iF.name())
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class Type>
void Foam::dfHybridInterfaceFvPatchField<Type>::updateCoeffs()
{
    if (this->updated())
    {
        return;
    }

    if (mode_ == "provider")
    {
        // ----- Provider mode: zeroGradient -----
        // Extrapolate internal field to the boundary
        this->refValue() = this->patchInternalField();
        this->refGrad() = Zero;
        this->valueFraction() = 0.0;
    }
    else // mode_ == "receiver"
    {
        // ----- Receiver mode: fixedValue from neighbour -----
        const mappedPatchBase& mpp = this->mapper();

        // Access the neighbour mesh and patch
        const fvMesh& nbrMesh =
            refCast<const fvMesh>(mpp.sampleMesh());

        const label samplePatchi = mpp.samplePolyPatch().index();

        const fvPatch& nbrPatch =
            nbrMesh.boundary()[samplePatchi];

        // Use the actual field name from the internal field,
        // not the stored fieldName_
        const word& lookupName = this->internalField().name();

        // Look up the target field on the neighbour patch
        typedef GeometricField<Type, fvPatchField, volMesh> fieldType;

        if (nbrMesh.foundObject<fieldType>(lookupName))
        {
            const fvPatchField<Type>& nbrField =
                nbrPatch.lookupPatchField<fieldType, Type>(lookupName);

            // Get the neighbour's internal cell values adjacent to the
            // patch.  This is robust regardless of what BC type the
            // neighbour has: if neighbour is "provider" (zeroGradient),
            // the boundary values equal the internal values anyway.
            tmp<Field<Type>> tnbrIntFld =
                nbrField.patchInternalField();

            // Map from neighbour patch layout to our patch layout
            // (handles parallel redistribution and non-conformal meshes)
            mpp.distribute(tnbrIntFld.ref());

            // Apply as fixedValue
            this->refValue() = tnbrIntFld();
            this->refGrad() = Zero;
            this->valueFraction() = 1.0;
        }
        else
        {
            // Field not found in neighbour region — fall back to
            // zero-gradient (extrapolate from own internal field).
            // This happens for template fields like Ydefault.
            WarningInFunction
                << "Field '" << lookupName
                << "' not found in neighbour region '"
                << nbrMesh.name() << "'. "
                << "Falling back to zeroGradient on patch "
                << this->patch().name() << endl;

            this->refValue() = this->patchInternalField();
            this->refGrad() = Zero;
            this->valueFraction() = 0.0;
        }

    }

    mixedFvPatchField<Type>::updateCoeffs();
}


template<class Type>
void Foam::dfHybridInterfaceFvPatchField<Type>::write(Ostream& os) const
{
    mixedFvPatchField<Type>::write(os);
    writeEntry(os, "mode", mode_);
    writeEntry(os, "fieldName", this->internalField().name());
}


// ************************************************************************* //
