reset
#==========================================================
set terminal epslatex standalone color dl 3 lw 2 size 10cm,7cm
set terminal epslatex "Times-Roman" 11
set output 'temp.tex'

set multiplot layout 2,1
set tmargin 1
set bmargin 4
#==========================================================
set style line 1 lc rgb  '#21618C' lw 1.5 ps 1.5 pt 7
set style line 2 lc rgb  '#A93226' lw 1.5 ps 1.5 pt 7
set style line 3 lc rgb  '#186A3B' lw 1.5 ps 1.5 pt 7
set style line 11 lc rgb '#21618C' lw 1.5 dt 3 ps 1.5 pt 7
set style line 12 lc rgb '#A93226' lw 1.5 dt 3 ps 1.5 pt 7
set style line 13 lc rgb '#186A3B' lw 1.5 dt 3 ps 1.5 pt 7

et = 8.5851
ehe = 7.4054
edd = 1.9527

#set xrange[2:14]

set key spacing 1.2
set key top right 
set ytics format '%g'
set lmargin 8
#==========================================================
set nogrid 
#unset xlabel
set xtics format '%g'

#==========================================================
set xlabel "$E_{\\textrm{\\scriptsize c.m.}}$ [MeV]"
set ylabel "$\\delta$ [Deg]"
#==========================================================
#plot 'diagPH_1-1.dat' u ($1):($3>0?$3:$3) ls 1 w l t '$\delta_{11}$',\
#	 'diagPH_2-2.dat' u ($1+et-ehe):3 ls 2 w l t '$\delta_{22}$',\
#	 'diagPH_3-3.dat' u ($1+et-edd):3 ls 3 w l t '$\delta_{33}$',\
#	 'scdiagPH_1-1.dat' u ($1):($3>0?$3:$3) ls 11 w l t '',\
#	 'scdiagPH_2-2.dat' u ($1+et-ehe):3     ls 12 w l t '',\
#	 'scdiagPH_3-3.dat' u ($1+et-edd):3     ls 13 w l t ''

plot 'atom-atom_phases.dat' u ($1):3 ls 1 w l t 'atom-atom $J^{\pi}=0^+$',\
	'dimer-dimer_phases.dat' u ($2):3 ls 3 w l t 'dimer-dimer $J^{\pi}=0^+$',\
	'tp-dd-mixing_phases.dat' u ($1):3 ls 11 w l t 'mixing $J^{\pi}=0^+$',\
	'trimer-atom_phases.dat' u ($1):3 ls 2 w l t 'trimer-atom $J^{\pi}=0^+$'

set xlabel "$E^{\\textrm{\\scriptsize match}}_{\\textrm{\\scriptsize c.m.}}$ [MeV]"
set ylabel "scattering length [fm]"
set key top left
#set tics font "Times-Roman,20"
set ytics 0.,1.,5    format "$%g$" nomirror
set yrange[0.:5]
set autoscale y
#set bmargin 4
##set size 1,0.5

plot 'a_ratio_4.0.dat' u ($1):($2) ls 1 w l t 'atom-atom',\
     'a_ratio_4.0.dat' u ($3):($4) ls 3 w l t 'dimer-dimer',\
     'a_ratio_4.0.dat' u ($7):($8) ls 11 w l t 'transition',\
     'a_ratio_4.0.dat' u ($5):($6) ls 2 w l t 'trimer-atom'

#plot 'atom-atom_phases.dat' u ($1):4 ls 1 w l notitle 'atom-atom $J^{\pi}=0^+$',\
#	 'dimer-dimer_phases.dat' u ($1):4 ls 3 w l notitle 'dimer-dimer $J^{\pi}=0^+$'

#
unset multiplot
#==========================================================
set output
! latex temp.tex && dvipdf temp.dvi
! mv temp.pdf 0p_phases.pdf
! rm temp*.*
#==========================================================

