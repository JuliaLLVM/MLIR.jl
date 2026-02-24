import ..IR: NamedAttribute

operandsegmentsizes(segments) = NamedAttribute("operand_segment_sizes", Int32.(segments))
resultsegmentsizes(segments) = NamedAttribute("result_segment_sizes", Int32.(segments))
