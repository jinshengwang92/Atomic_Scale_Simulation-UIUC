mol delete all
display resetview
mol load xyz "traj.xyz"
set molid [molinfo top]

mol active $molid
mol delrep 0 $molid
mol representation VDW
mol addrep $molid
mol color red
mol modcolor 0 $molid Molecule

set sel [atomselect $molid all]
$sel set radius 0.1

animate speed 0.9
animate forward
