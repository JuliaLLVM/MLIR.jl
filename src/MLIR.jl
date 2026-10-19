module MLIR

using Preferences
using ScopedValues
using MLIR_jll: MLIR_jll, mlir_c

const MLIR_VERSION = ScopedValue(
    VersionNumber(@load_preference("MLIR_VERSION", Base.libllvm_version_string))
)

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
