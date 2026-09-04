open_project -reset lictest
set_top topf
add_files t.cpp
open_solution -reset -flow_target vivado sol1
set_part {xcvu13p-flga2577-2-e}
create_clock -period 5 -name default
csynth_design
exit
