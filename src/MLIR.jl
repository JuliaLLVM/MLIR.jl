module MLIR

using Preferences
using ScopedValues
using MLIR_jll: MLIR_jll

"""
    Session{P,V}

A object for statically encoding the path to _libMLIR_ and its version on its type parameters.
This is required by the FFI bindings to be able to switch _libMLIR_ in a way amenable to `ccall`.
"""
struct Session{P,V} end
Session(path, version) = Session{Symbol(path)}(version)
Session{P}(version::Integer) = Session{Symbol(path),version}()
Session{P}(version::VersionNumber) = Session{Symbol(path),Int(version.major)}()
@inline mlir_path(@specialize(_::Type{<:Session{P}})) where {P} = String(P)
@inline mlir_version(@specialize(_::Type{Session{P,V}})) where {P,V} = VersionNumber(V)

const MLIR_VERSION = ScopedValue(
    VersionNumber(@load_preference("MLIR_VERSION", Base.libllvm_version_string))
)
const MLIR_C_PATH = ScopedValue(@load_preference("MLIR_C_PATH", MLIR_jll.mlir_c))

const MLIR_VERSION_MIN = v"14"
const MLIR_VERSION_MAX = v"21"

struct MLIRException <: Exception
    msg::String
end

Base.showerror(io::IO, err::MLIRException) = print(io, err.msg)

include("API/API.jl")
include("IR/IR.jl")
include("Dialects/Dialects.jl")

end # module MLIR
