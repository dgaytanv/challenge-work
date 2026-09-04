read_verilog t.v
synth_design -top topv -part xcvu13p-flga2577-2-e -mode out_of_context
report_utilization
puts "VIVADO_SYNTH_OK"
exit
