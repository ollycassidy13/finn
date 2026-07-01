if {$argc < 4} {
    puts "usage: postroute_physopt_stitched_ip_ooc.tcl <src_routed_dcp> <src_xdc> <out_dir> <clk_ns> ?phys_directive? ?route_directive? ?final_phys_directive?"
    exit 2
}

set src_routed_dcp [lindex $argv 0]
set src_xdc [lindex $argv 1]
set out_dir [lindex $argv 2]
set clk_ns [lindex $argv 3]
set phys_directive "AggressiveExplore"
set route_directive ""
set final_phys_directive ""

if {$argc > 4} {
    set phys_directive [lindex $argv 4]
}
if {$argc > 5} {
    set route_directive [lindex $argv 5]
}
if {$argc > 6} {
    set final_phys_directive [lindex $argv 6]
}

file mkdir $out_dir

open_checkpoint $src_routed_dcp
reset_timing
if {[file exists $src_xdc]} {
    read_xdc $src_xdc
}

set ap_clk_port [get_ports ap_clk]
set ap_clk_existing [get_clocks -quiet -of_objects $ap_clk_port]
if {[llength $ap_clk_existing] > 0} {
    reset_timing
}
create_clock -name ap_clk -period $clk_ns $ap_clk_port

set ap_clk2x_ports [get_ports -quiet ap_clk2x]
if {[llength $ap_clk2x_ports] > 0} {
    set ap_clk2x_existing [get_clocks -quiet -of_objects $ap_clk2x_ports]
    if {[llength $ap_clk2x_existing] > 0} {
        reset_timing
        create_clock -name ap_clk -period $clk_ns $ap_clk_port
    }
    create_clock -name ap_clk2x -period [expr {$clk_ns / 2.0}] $ap_clk2x_ports
}

if {$phys_directive eq "" || $phys_directive eq "Default"} {
    phys_opt_design
} else {
    phys_opt_design -directive $phys_directive
}

if {$route_directive eq "" || $route_directive eq "Default"} {
    route_design
} else {
    route_design -directive $route_directive
}

if {$final_phys_directive ne "" && $final_phys_directive ne "None"} {
    if {$final_phys_directive eq "Default"} {
        phys_opt_design
    } else {
        phys_opt_design -directive $final_phys_directive
    }
}

report_utilization -file "$out_dir/ooc_utilization.rpt"
report_timing_summary -file "$out_dir/ooc_timing.rpt"
report_route_status -file "$out_dir/ooc_route_status.rpt"
report_power -file "$out_dir/ooc_power.rpt"

set fp [open "$out_dir/ooc_metadata.txt" w]
puts $fp "clk_period_ns=$clk_ns"
puts $fp "vivado_version=[version -short]"
puts $fp "phys_directive=$phys_directive"
puts $fp "route_directive=$route_directive"
puts $fp "final_phys_directive=$final_phys_directive"
puts $fp "source_routed_dcp=$src_routed_dcp"
close $fp

write_checkpoint -force "$out_dir/finn_design_routed.dcp"
exit
