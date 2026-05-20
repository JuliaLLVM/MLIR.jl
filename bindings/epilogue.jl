#### EPILOGUE

# MlirStringRef is a non-owning reference to a string,
# we thus need to ensure that the Julia string remains alive
# over the use. For that we use the cconvert/unsafe_convert mechanism
# for foreign-calls. The returned value of the cconvert is rooted across
# foreign-call.
Base.cconvert(::Core.Type{MlirStringRef}, s::Union{Symbol,String}) = s
function Base.cconvert(::Core.Type{MlirStringRef}, s::AbstractString)
    return Base.cconvert(MlirStringRef, String(s)::String)
end

# Directly create `MlirStringRef` instead of adding an extra ccall.
function Base.unsafe_convert(
    ::Core.Type{MlirStringRef}, s::Union{Symbol,String,AbstractVector{UInt8}}
)
    p = Base.unsafe_convert(Ptr{Cchar}, s)
    return MlirStringRef(p, sizeof(s))
end

Base.String(str::MlirStringRef) = Base.unsafe_string(pointer(str.data), str.length)
Base.String(str::MlirIdentifier) = String(mlirIdentifierStr(str))

function print_callback(str::MlirStringRef, userdata)
    data = unsafe_wrap(Array, Base.convert(Ptr{Cchar}, str.data), str.length; own=false)
    write(userdata isa Base.RefValue ? userdata[] : userdata, data)
    return Cvoid()
end

##### END OF EPILOGUE
