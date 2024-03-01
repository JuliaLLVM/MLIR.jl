struct Attribute
    attribute::API.MlirAttribute
end

"""
    Attribute()

Returns an empty attribute.
"""
Attribute() = Attribute(API.mlirAttributeGetNull())

Base.convert(::Core.Type{API.MlirAttribute}, attribute::Attribute) = attribute.attribute

"""
    parse(::Core.Type{Attribute}, str; context=context())

Parses an attribute. The attribute is owned by the context.
"""
Base.parse(::Core.Type{Attribute}, str; context::Context=context) = Attribute(API.mlirAttributeParseGet(context, str))

"""
    ==(a1, a2)

Checks if two attributes are equal.
"""
Base.:(==)(a::Attribute, b::Attribute) = API.mlirAttributeEqual(a, b)

"""
    context(attribute)

Gets the context that an attribute was created with.
"""
context(attr::Attribute) = Context(API.mlirAttributeGetContext(attr))

"""
    type(attribute)

Gets the type of this attribute.
"""
type(attr::Attribute) = Type(API.mlirAttributeGetType(attr))

"""
    typeid(attribute)

Gets the type id of the attribute.
"""
typeid(attr::Attribute) = TypeID(API.mlirAttributeGetTypeID(attr))

"""
    isaffinemap(attr)

Checks whether the given attribute is an affine map attribute.
"""
isaffinemap(attr::Attribute) = API.mlirAttributeIsAAffineMap(attr)

"""
    Attribute(affineMap)

Creates an affine map attribute wrapping the given map. The attribute belongs to the same context as the affine map.
"""
Attribute(map::AffineMap) = Attribute(API.mlirAffineMapAttrGet(map))

"""
    AffineMap(attr)

Returns the affine map wrapped in the given affine map attribute.
"""
AffineMap(attr::Attribute) = AffineMap(API.mlirAffineMapAttrGetValue(attr))

"""
    isarray(attr)

Checks whether the given attribute is an array attribute.
"""
isarray(attr::Attribute) = API.mlirAttributeIsAArray(attr)

"""
    Attribute(elements; context=context())

Creates an array element containing the given list of elements in the given context.
"""
Attribute(attrs::Vector{Attribute}; context::Context=context()) = Attribute(API.mlirArrayAttrGet(context, length(attrs), pointer(attrs)))

"""
    isdict(attr)

Checks whether the given attribute is a dictionary attribute.
"""
isdict(attr::Attribute) = API.mlirAttributeIsADictionary(attr)

"""
    Attribute(elements; context=context())

Creates a dictionary attribute containing the given list of elements in the provided context.
"""
function Attribute(attrs::Dict; context::Context=context())
    attrs = map(splat(NamedAttribute), attrs)
    Attribute(API.mlirDictionaryAttrGet(context, length(attrs), pointer(attrs)))
end

"""
    isfloat(attr)

Checks whether the given attribute is a floating point attribute.
"""
isfloat(attr::Attribute) = API.mlirAttributeIsAFloat(attr)

"""
    Attribute(float; context=context(), location=Location(), check=false)

Creates a floating point attribute in the given context with the given double value and double-precision FP semantics.
If `check=true`, emits appropriate diagnostics on illegal arguments.
"""
function Attribute(f::T; context::Context=context(), location::Location=Location(), check::Bool=false) where {T<:AbstractFloat}
    if check
        Attribute(API.mlirFloatAttrDoubleGetChecked(location, Type(T), Float64(f)))
    else
        Attribute(API.mlirFloatAttrDoubleGet(context, Type(T), Float64(f)))
    end
end

"""
    Float64(attr)

Returns the value stored in the given floating point attribute, interpreting the value as double.
"""
function Base.Float64(attr::Attribute)
    @assert isfloat(attr) "attribute $(attr) is not a floating point attribute"
    API.mlirFloatAttrGetValueDouble(attr)
end

"""
    isinteger(attr)

Checks whether the given attribute is an integer attribute.
"""
isinteger(attr::Attribute) = API.mlirAttributeIsAInteger(attr)

"""
    Attribute(int)

Creates an integer attribute of the given type with the given integer value.
"""
Attribute(i::T) where {T<:Integer} = Attribute(API.mlirIntegerAttrGet(Type(T), Int64(i)))

# TODO mlirIntegerAttrGetValueInt

"""
    Int(attr)

Returns the value stored in the given integer attribute, assuming the value is of signed type and fits into a signed 64-bit integer.
"""
function Base.Int(attr::Attribute)
    @assert isinteger(attr) "attribute $(attr) is not an integer attribute"
    API.mlirIntegerAttrGetValueInt(attr)
end

"""
    UInt(attr)

Returns the value stored in the given integer attribute, assuming the value is of unsigned type and fits into an unsigned 64-bit integer.
"""
function Base.UInt(attr::Attribute)
    @assert isinteger(attr) "attribute $(attr) is not an integer attribute"
    API.mlirIntegerAttrGetValueUInt(attr)
end

"""
    isbool(attr)

Checks whether the given attribute is a bool attribute.
"""
isbool(attr::Attribute) = API.mlirAttributeIsABool(attr)

"""
    Attribute(value; context=context())

Creates a bool attribute in the given context with the given value.
"""
Attribute(b::Bool; context::Context=context()) = Attribute(API.mlirBoolAttrGet(context, b))

"""
    Bool(attr)

Returns the value stored in the given bool attribute.
"""
function Base.Bool(attr::Attribute)
    @assert isbool(attr) "attribute $(attr) is not a boolean attribute"
    API.mlirBoolAttrGetValue(attr)
end

"""
    isintegerset(attr)

Checks whether the given attribute is an integer set attribute.
"""
isintegerset(attr::Attribute) = API.mlirAttributeIsAIntegerSet(attr)

"""
    isopaque(attr)

Checks whether the given attribute is an opaque attribute.
"""
isopaque(attr::Attribute) = API.mlirAttributeIsAOpaque(attr)

"""
    OpaqueAttribute(dialectNamespace, dataLength, data, type; context=context())

Creates an opaque attribute in the given context associated with the dialect identified by its namespace.
The attribute contains opaque byte data of the specified length (data need not be null-terminated).
"""
OpaqueAttribute(namespace, data, type; context::Context=context) = Attribute(API.mlirOpaqueAttrGet(context, namespace, length(data), data, type))

"""
    mlirOpaqueAttrGetDialectNamespace(attr)

Returns the namespace of the dialect with which the given opaque attribute is associated. The namespace string is owned by the context.
"""
function namespace(attr::Attribute)
    @assert isopaque(attr) "attribute $(attr) is not an opaque attribute"
    String(API.mlirOpaqueAttrGetDialectNamespace(attr))
end

"""
    data(attr)

Returns the raw data as a string reference. The data remains live as long as the context in which the attribute lives.
"""
function data(attr::Attribute)
    @assert isopaque(attr) "attribute $(attr) is not an opaque attribute"
    String(API.mlirOpaqueAttrGetData(attr)) # TODO return as Base.CodeUnits{Int8,String}? or as a Vector{Int8}? or Pointer?
end

"""
    isstring(attr)

Checks whether the given attribute is a string attribute.
"""
isstring(attr::Attribute) = API.mlirAttributeIsAString(attr)

"""
    Attribute(str; context=context())

Creates a string attribute in the given context containing the given string.
"""
Attribute(str::AbstractString; context::Context=context()) = Attribute(API.mlirStringAttrGet(context, str))

"""
    Attribute(type, str)

Creates a string attribute in the given context containing the given string. Additionally, the attribute has the given type.
"""
function Attribute(type::Type, str::AbstractString)
    Attribute(API.mlirStringAttrTypedGet(type, str))
end

"""
    String(attr)

Returns the attribute values as a string reference. The data remains live as long as the context in which the attribute lives.
"""
function Base.String(attr::Attribute)
    @assert isstring(attr) "attribute $(attr) is not a string attribute"
    String(API.mlirStringAttrGetValue(attr))
end

"""
    issymbolref(attr)

Checks whether the given attribute is a symbol reference attribute.
"""
issymbolref(attr::Attribute) = API.mlirAttributeIsASymbolRef(attr)

"""
    SymbolRefAttribute(symbol, references; context=context())

Creates a symbol reference attribute in the given context referencing a symbol identified by the given string inside a list of nested references.
Each of the references in the list must not be nested.
"""
SymbolRefAttribute(symbol::String, references::Vector{Attribute}; context::Context=context()) = Attribute(API.mlirSymbolRefAttrGet(context, symbol, length(references), pointer(references)))

"""
    rootref(attr)

Returns the string reference to the root referenced symbol. The data remains live as long as the context in which the attribute lives.
"""
function rootref(attr::Attribute)
    @assert issymbolref(attr) "attribute $(attr) is not a symbol reference attribute"
    String(API.mlirSymbolRefAttrGetRootReference(attr))
end

"""
    leafref(attr)

Returns the string reference to the leaf referenced symbol. The data remains live as long as the context in which the attribute lives.
"""
function leafref(attr::Attribute)
    @assert issymbolref(attr) "attribute $(attr) is not a symbol reference attribute"
    String(API.mlirSymbolRefAttrGetLeafReference(attr))
end

"""
    nnestedrefs(attr)

Returns the number of references nested in the given symbol reference attribute.
"""
function nnestedrefs(attr::Attribute)
    @assert issymbolref(attr) "attribute $(attr) is not a symbol reference attribute"
    API.mlirSymbolRefAttrGetNumNestedReferences(attr)
end

"""
    isflatsymbolref(attr)

Checks whether the given attribute is a flat symbol reference attribute.
"""
isflatsymbolref(attr::Attribute) = API.mlirAttributeIsAFlatSymbolRef(attr)

"""
    FlatSymbolRefAttribute(ctx, symbol)

Creates a flat symbol reference attribute in the given context referencing a symbol identified by the given string.
"""
FlatSymbolRefAttribute(symbol::String; context::Context=context()) = Attribute(API.mlirFlatSymbolRefAttrGet(context, symbol))

"""
    flatsymbol(attr)

Returns the referenced symbol as a string reference. The data remains live as long as the context in which the attribute lives.
"""
function flatsymbol(attr::Attribute)
    @assert isflatsymbolref(attr) "attribute $(attr) is not a flat symbol reference attribute"
    String(API.mlirFlatSymbolRefAttrGetValue(attr))
end

"""
    istype(attr)

Checks whether the given attribute is a type attribute.
"""
istype(attr::Attribute) = API.mlirAttributeIsAType(attr)

"""
    Attribute(type)

Creates a type attribute wrapping the given type in the same context as the type.
"""
Attribute(type::Type) = Attribute(API.mlirTypeAttrGet(type))

"""
    Type(attr)

Returns the type stored in the given type attribute.
"""
Type(attr::Attribute) = Type(API.mlirTypeAttrGetValue(attr))

"""
    isunit(attr)

Checks whether the given attribute is a unit attribute.
"""
isunit(attr::Attribute) = API.mlirAttributeIsAUnit(attr)

"""
    UnitAttribute(; context=context())

Creates a unit attribute in the given context.
"""
UnitAttribute(; context::Context=context()) = Attribute(API.mlirUnitAttrGet(context))

"""
    iselements(attr)

Checks whether the given attribute is an elements attribute.
"""
iselements(attr::Attribute) = API.mlirAttributeIsAElements(attr)

# TODO mlirElementsAttrGetValue
# TODO mlirElementsAttrIsValidIndex

"""
    isdenseelements(attr)

Checks whether the given attribute is a dense elements attribute.
"""
isdenseelements(attr::Attribute) = API.mlirAttributeIsADenseElements(attr)
isdenseintelements(attr::Attribute) = API.mlirAttributeIsADenseIntElements(attr)
isdensefloatelements(attr::Attribute) = API.mlirAttributeIsADenseFPElements(attr)

"""
    DenseElementsAttribute(shapedType, elements)

Creates a dense elements attribute with the given Shaped type and elements in the same context as the type.
"""
function DenseElementsAttribute(shaped_type::Type, elements::AbstractArray)
    @assert isshaped(shaped_type) "type $(shaped_type) is not a shaped type"
    Attribute(API.mlirDenseElementsAttrGet(shaped_type, length(elements), pointer(elements)))
end

# TODO mlirDenseElementsAttrRawBufferGet

"""
    fill(attr, shapedType)

Creates a dense elements attribute with the given Shaped type containing a single replicated element (splat).
"""
function Base.fill(attr::Attribute, shaped_type::Type)
    @assert isshaped(shaped_type) "type $(shaped_type) is not a shaped type"
    Attribute(API.mlirDenseElementsAttrSplatGet(attr, shaped_type))
end

function Base.fill(value::Bool, shaped_type::Type)
    @assert isshaped(shaped_type) "type $(shaped_type) is not a shaped type"
    API.mlirDenseElementsAttrBoolSplatGet(value, shaped_type)
end

function Base.fill(value::UInt8, shaped_type::Type)
    @assert isshaped(shaped_type) "type $(shaped_type) is not a shaped type"
    API.mlirDenseElementsAttrUInt8SplatGet(value, shaped_type)
end

function Base.fill(value::Int8, shaped_type::Type)
    @assert isshaped(shaped_type) "type $(shaped_type) is not a shaped type"
    API.mlirDenseElementsAttrInt8SplatGet(value, shaped_type)
end

function Base.fill(value::UInt32, shaped_type::Type)
    @assert isshaped(shaped_type) "type $(shaped_type) is not a shaped type"
    API.mlirDenseElementsAttrUInt32SplatGet(value, shaped_type)
end

function Base.fill(value::Int32, shaped_type::Type)
    @assert isshaped(shaped_type) "type $(shaped_type) is not a shaped type"
    API.mlirDenseElementsAttrInt32SplatGet(value, shaped_type)
end

function Base.fill(value::UInt64, shaped_type::Type)
    @assert isshaped(shaped_type) "type $(shaped_type) is not a shaped type"
    API.mlirDenseElementsAttrUInt64SplatGet(value, shaped_type)
end

function Base.fill(value::Int64, shaped_type::Type)
    @assert isshaped(shaped_type) "type $(shaped_type) is not a shaped type"
    API.mlirDenseElementsAttrInt64SplatGet(value, shaped_type)
end

function Base.fill(value::Float32, shaped_type::Type)
    @assert isshaped(shaped_type) "type $(shaped_type) is not a shaped type"
    API.mlirDenseElementsAttrFloatSplatGet(value, shaped_type)
end

function Base.fill(value::Float64, shaped_type::Type)
    @assert isshaped(shaped_type) "type $(shaped_type) is not a shaped type"
    API.mlirDenseElementsAttrDoubleSplatGet(value, shaped_type)
end

function Base.fill(value::Union{Attribute,Bool,UInt8,Int8,UInt32,Int32,UInt64,Int64,Float32,Float64}, shape)
    shaped_type = TensorType(length(shape), collect(shape), Type(typeof(value)))
    Base.fill(value, shaped_type)
end

"""
    Attribute(array::AbstractArray)

Creates a dense elements attribute with the given shaped type from elements of a specific type. Expects the element type of the shaped type to match the data element type.
"""
function Attribute(values::AbstractVector{Bool})
    shaped_type = TensorType(ndims(values), collect(size(values)), Type(Bool))
    Attribute(API.mlirDenseElementsAttrBoolGet(shaped_type, length(values), pointer(values)))
end

function Attribute(values::AbstractArray{UInt8})
    shaped_type = TensorType(ndims(values), collect(size(values)), Type(UInt8))
    Attribute(API.mlirDenseElementsAttrUInt8Get(shaped_type, length(values), pointer(values)))
end

function Attribute(values::AbstractArray{Int8})
    shaped_type = TensorType(ndims(values), collect(size(values)), Type(Int8))
    Attribute(API.mlirDenseElementsAttrInt8Get(shaped_type, length(values), pointer(values)))
end

function Attribute(values::AbstractArray{UInt16})
    shaped_type = TensorType(ndims(values), collect(size(values)), Type(UInt16))
    Attribute(API.mlirDenseElementsAttrUInt16Get(shaped_type, length(values), pointer(values)))
end

function Attribute(values::AbstractArray{Int16})
    shaped_type = TensorType(ndims(values), collect(size(values)), Type(Int16))
    Attribute(API.mlirDenseElementsAttrInt16Get(shaped_type, length(values), pointer(values)))
end

function Attribute(values::AbstractArray{UInt32})
    shaped_type = TensorType(ndims(values), collect(size(values)), Type(UInt32))
    Attribute(API.mlirDenseElementsAttrUInt32Get(shaped_type, length(values), pointer(values)))
end

function Attribute(values::AbstractArray{Int32})
    shaped_type = TensorType(ndims(values), collect(size(values)), Type(Int32))
    Attribute(API.mlirDenseElementsAttrInt32Get(shaped_type, length(values), pointer(values)))
end

function Attribute(values::AbstractArray{UInt64})
    shaped_type = TensorType(ndims(values), collect(size(values)), Type(UInt64))
    Attribute(API.mlirDenseElementsAttrUInt64Get(shaped_type, length(values), pointer(values)))
end

function Attribute(values::AbstractArray{Int64})
    shaped_type = TensorType(ndims(values), collect(size(values)), Type(Int64))
    Attribute(API.mlirDenseElementsAttrInt64Get(shaped_type, length(values), pointer(values)))
end

function Attribute(values::AbstractArray{Float32})
    shaped_type = TensorType(ndims(values), collect(size(values)), Type(Float32))
    Attribute(API.mlirDenseElementsAttrFloatGet(shaped_type, length(values), pointer(values)))
end

function Attribute(values::AbstractArray{Float64})
    shaped_type = TensorType(ndims(values), collect(size(values)), Type(Float64))
    Attribute(API.mlirDenseElementsAttrDoubleGet(shaped_type, length(values), pointer(values)))
end

# TODO mlirDenseElementsAttrBFloat16Get

function Attribute(values::AbstractArray{Float16})
    shaped_type = TensorType(ndims(values), collect(size(values)), Type(Float16))
    Attribute(API.mlirDenseElementsAttrFloat16Get(shaped_type, length(values), pointer(values)))
end

"""
    Attribute(array::AbstractArray{String})

Creates a dense elements attribute with the given shaped type from string elements.
"""
function Attribute(values::AbstractArray{String})
    # TODO may fail because `Type(String)` is not defined
    shaped_type = TensorType(ndims(values), collect(size(values)), Type(String))
    Attribute(API.mlirDenseElementsAttrStringGet(shaped_type, length(values), pointer(values)))
end

"""
    Base.reshape(attr, shapedType)

Creates a dense elements attribute that has the same data as the given dense elements attribute and a different shaped type. The new type must have the same total number of elements.
"""
function Base.reshape(attr::Attribute, shape)
    @assert isdenseelements(attr) "attribute $(attr) is not a dense elements attribute"
    @assert length(attr) == prod(shape) "new shape $(shape) has a different number of elements than the original attribute"
    element_type = eltype(type(attr))
    shaped_type = TensorType(length(shape), collect(shape), element_type)
    Attribute(API.mlirDenseElementsAttrReshape(attr, shaped_type))
end

"""
    issplat(attr)

Checks whether the given dense elements attribute contains a single replicated value (splat).
"""
function issplat(attr::Attribute)
    @assert isdenseelements(attr) "attribute $(attr) is not a dense elements attribute"
    API.mlirDenseElementsAttrIsSplat(attr) # TODO Base.allequal?
end

# TODO mlirDenseElementsAttrGetRawData

"""
    isopaqueelements(attr)

Checks whether the given attribute is an opaque elements attribute.
"""
isopaqueelements(attr::Attribute) = API.mlirAttributeIsAOpaqueElements(attr)

"""
    issparseelements(attr)

Checks whether the given attribute is a sparse elements attribute.
"""
issparseelements(attr::Attribute) = API.mlirAttributeIsASparseElements(attr)

# TODO mlirSparseElementsAttribute
# TODO mlirSparseElementsAttrGetIndices
# TODO mlirSparseElementsAttrGetValues

function Base.length(attr::Attribute)
    if isarray(attr)
        API.mlirArrayAttrGetNumElements(attr)
    elseif isdict(attr)
        API.mlirDictionaryAttrGetNumElements(attr)
    elseif iselements(attr)
        API.mlirElementsAttrGetNumElements(attr)
    end
end

function Base.getindex(attr::Attribute, i)
    if isarray(attr)
        Attribute(API.mlirArrayAttrGetElement(attr, i))
    elseif isdict(attr)
        if i isa String
            Attribute(API.mlirDictionaryAttrGetElementByName(attr, i))
        else
            NamedAttribute(API.mlirDictionaryAttrGetElement(attr, i))
        end
    elseif isdenseelements(attr)
        elem_type = julia_type(eltype(type(attr)))
        if elem_type isa Bool
            API.mlirDenseElementsAttrGetBoolValue(attr, i)
        elseif elem_type isa Int8
            API.mlirDenseElementsAttrGetInt8Value(attr, i)
        elseif elem_type isa UInt8
            API.mlirDenseElementsAttrGetUInt8Value(attr, i)
        elseif elem_type isa Int16
            API.mlirDenseElementsAttrGetInt16Value(attr, i)
        elseif elem_type isa UInt16
            API.mlirDenseElementsAttrGetUInt16Value(attr, i)
        elseif elem_type isa Int32
            API.mlirDenseElementsAttrGetInt32Value(attr, i)
        elseif elem_type isa UInt32
            API.mlirDenseElementsAttrGetUInt32Value(attr, i)
        elseif elem_type isa Int64
            API.mlirDenseElementsAttrGetInt64Value(attr, i)
        elseif elem_type isa UInt64
            API.mlirDenseElementsAttrGetUInt64Value(attr, i)
        elseif elem_type isa Float32
            API.mlirDenseElementsAttrGetFloatValue(attr, i)
        elseif elem_type isa Float64
            API.mlirDenseElementsAttrGetDoubleValue(attr, i)
        elseif elem_type isa String # TODO does this case work?
            String(API.mlirDenseElementsAttrGetStringValue(attr, i))
        else
            throw("unsupported element type $(elem_type)")
        end
    end
end

function Base.getindex(attr::Attribute)
    @assert isdenseelements(attr) "attribute $(attr) is not a dense elements attribute"
    @assert issplat(attr) "attribute $(attr) is not splatted (more than one different elements)"
    elem_type = julia_type(eltype(type(attr)))
    if elem_type isa Bool
        API.mlirDenseElementsAttrGetBoolSplatValue(attr)
    elseif elem_type isa Int8
        API.mlirDenseElementsAttrGetInt8SplatValue(attr)
    elseif elem_type isa UInt8
        API.mlirDenseElementsAttrGetUInt8SplatValue(attr)
    elseif elem_type isa Int16
        API.mlirDenseElementsAttrGetInt16SplatValue(attr)
    elseif elem_type isa UInt16
        API.mlirDenseElementsAttrGetUInt16SplatValue(attr)
    elseif elem_type isa Int32
        API.mlirDenseElementsAttrGetInt32SplatValue(attr)
    elseif elem_type isa UInt32
        API.mlirDenseElementsAttrGetUInt32SplatValue(attr)
    elseif elem_type isa Int64
        API.mlirDenseElementsAttrGetInt64SplatValue(attr)
    elseif elem_type isa UInt64
        API.mlirDenseElementsAttrGetUInt64SplatValue(attr)
    elseif elem_type isa Float32
        API.mlirDenseElementsAttrGetFloatSplatValue(attr)
    elseif elem_type isa Float64
        API.mlirDenseElementsAttrGetDoubleSplatValue(attr)
    elseif elem_type isa String # TODO does this case work?
        String(API.mlirDenseElementsAttrGetStringSplatValue(attr))
    else
        throw("unsupported element type $(elem_type)")
    end
end

function Base.show(io::IO, attribute::Attribute)
    print(io, "Attribute(#= ")
    c_print_callback = @cfunction(print_callback, Cvoid, (API.MlirStringRef, Any))
    ref = Ref(io)
    API.mlirAttributePrint(attribute, c_print_callback, ref)
    print(io, " =#)")
end

struct NamedAttribute
    named_attribute::API.MlirNamedAttribute
end

"""
    NamedAttribute(name, attr)

Associates an attribute with the name. Takes ownership of neither.
"""
function NamedAttribute(name, attribute; context=context(attribute))
    @assert !mlirIsNull(attribute.attribute)
    name = API.mlirIdentifierGet(context, name)
    NamedAttribute(API.mlirNamedAttributeGet(name, attribute))
end

Base.convert(::Core.Type{API.MlirAttribute}, named_attribute::NamedAttribute) = named_attribute.named_attribute
