module vcix

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, context, IndexType
import ..Dialects: operandsegmentsizes, resultsegmentsizes

"""
`v_iv`

Binary VCIX operation with an immediate second operand.

Correponds to:
|Mnemonic|funct6|vm|rs2|rs1|funct3|rd|Destination|Sources|
|--|--|--|--|--|--|--|--|--|
|sf.vc.v.iv|0010--|0|vs2|simm|011|vd|vector vd| simm[4:0]  vector vs2|
"""
function v_iv(vs2::Value, vl=nothing::Union{Nothing, Value}; res::IR.Type, opcode, imm, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[vs2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("opcode", opcode), NamedAttribute("imm", imm), ]
    !isnothing(vl) && push!(operands, vl)
    
    IR.create_operation(
        "vcix.v.iv", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end

"""
`v_sv`

Binary VCIX operation with an integer scalar, or floating pointer scalar or
vector second operand.

Correponds to:
|Mnemonic|funct6|vm|rs2|rs1|funct3|rd|Destination| Sources|
|--|--|--|--|--|--|--|--|--|--|
|sf.vc.v.vv|0010--|0|vs2|vs1|000|vd|vector vd|vector vs1, vector vs|
|sf.vc.v.xv|0010--|0|vs2|xs1|100|vd|vector vd|scalar xs1, vector vs2|
|sf.vc.v.fv|0010--|0|vs2|fs1|101|vd|vector vd|scalar fs1, vector vs2|
"""
function v_sv(vs2::Value, op::Value, vl=nothing::Union{Nothing, Value}; res::IR.Type, opcode, location=Location())
    op_ty_results = IR.Type[res, ]
    operands = Value[vs2, op, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[NamedAttribute("opcode", opcode), ]
    !isnothing(vl) && push!(operands, vl)
    
    IR.create_operation(
        "vcix.v.sv", location;
        operands, owned_regions, successors, attributes,
        results=op_ty_results,
        result_inference=false
    )
end
end # vcix
