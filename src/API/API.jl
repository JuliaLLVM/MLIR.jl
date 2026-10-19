module API

using ..MLIR: MLIRException, MLIR_VERSION_MIN, MLIR_VERSION_MAX, mlir_version, default_session

include("Types.jl")
using .Types

let
    for dir in Base.Filesystem.readdir(joinpath(@__DIR__))
        isdir(joinpath(@__DIR__, dir)) || continue

        modname = Symbol(:v, dir)
        version_major = parse(Int, dir)
        
        # generate versioned API modules
        #! format: off
        @eval module $modname
            using ...MLIR: mlir_c
            using ..API.Types
            include(joinpath(@__DIR__, $dir, "libMLIR_h.jl"))
        end

        is_mlir_api = (name) -> name ∉ [nameof(mod), :eval, :include] && !startswith(string(name), '#')
        for funcname in Iterators.filter(is_mlir_api, names(mod; all=true))
            if !Base.isdefined(API, funcname)
                @eval begin
                    # fallback for MLIR versions that do not have such API function for that version
                    function $funcname(session::Session, args...; kwargs...)
                        error("API.$funcname not defined for version $(mlir_version(session))")
                    end

                    # some sugar if you code by hand
                    function $funcname(args...; session = default_session(), kwargs...)
                        return $funcname(session, args...; kwargs...)
                    end
                end
            end

            # dispatch to appropiate module based on major version of MLIR
            @eval function $funcname(session::Session{P,$version_major}, args...; kwargs...) where {P}
                return $modname.$funcname(session, args...; kwargs...)
            end
        end
        #! format: on
    end
end

function print_callback(str::API.MlirStringRef, userdata)
    data = unsafe_wrap(Array, Base.convert(Ptr{Cchar}, str.data), str.length; own=false)
    write(userdata isa Base.RefValue ? userdata[] : userdata, data)
    return Cvoid()
end

end # module API
