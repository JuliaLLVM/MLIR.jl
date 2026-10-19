module MLIR

using Preferences
using ScopedValues
using MLIR_jll: MLIR_jll, mlir_c

"""
    Session{P,V}

A object for statically encoding the path to _libMLIR_ and its version on its type parameters.
This is required by the FFI bindings to be able to switch _libMLIR_ in a way amenable to `ccall`.
"""
struct Session{P,V} end
Session(path, version) = Session{Symbol(path)}(version)
Session{P}(version::Integer) = Session{P,version}()
Session{P}(version::VersionNumber) = Session{P,Int(version.major)}()

"""
    mlir_path(::Session)

Return the path to the `libMLIR` encoded in the [`Session`](@ref).
"""
function mlir_path end

mlir_path(::S) where {S<:Session} = mlir_path(S)
@inline mlir_path(@specialize(_::Type{<:Session{P}})) where {P} = P


"""
    mlir_version(::Session)

Return the version of the `libMLIR` encoded in the [`Session`](@ref).
"""
function mlir_version end

mlir_path(::S) where {S<:Session} = mlir_path(S)
@inline mlir_version(@specialize(_::Type{Session{P,V}})) where {P,V} = VersionNumber(V)

# TODO is this type-stable? if not, would a generated function work here?
function default_session()
    path = @load_preference("MLIR_C_PATH", mlir_c)
    version = VersionNumber(@load_preference("MLIR_VERSION", Base.libllvm_version_string))
    return Session(path, version.major)
end

# The auto-generated `@ccall mlir_c.xxx(...)` sites in `src/API/N/libMLIR_h.jl`
# bind their library path directly to `MLIR_jll`'s `mlir_c` product — a
# JLLWrappers-managed `::String` global that's safe to use as a bare
# identifier in `@ccall` across every Julia version.
#
# This replaces the previous `MLIR_C_PATH = ScopedValue(@load_preference(...))`
# (introduced in #71) which used the form `@ccall (MLIR_C_PATH[]).xxx(...)`.
# Julia 1.13 tightened the `@ccall` macro to require the library form be a
# bare identifier, `Symbol`, or literal `String` — the Ref-deref form now
# errors with "expected Symbol, got Expr" at parse time.
#
# Users who want a custom `libMLIR-C.so` can override `MLIR_jll`'s artifact
# via Preferences (`Override.toml`) or by `Pkg.develop`-ing it; the
# `@load_preference("MLIR_C_PATH", ...)` indirection is no longer needed.

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
