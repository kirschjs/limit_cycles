reset
#==========================================================
set terminal epslatex standalone color dl 3 lw 2 size 10cm,7cm
set terminal epslatex "Times-Roman" 11
set output 'temp.tex'

set multiplot layout 2,1
set tmargin 1
set bmargin 0
#==========================================================
set style line 1 lc rgb '#21618C' lw 2.5 ps 1.5 pt 7
set style line 2 lc rgb '#A93226' lw 2.5 ps 1.5 pt 7
set style line 3 lc rgb '#186A3B' lw 2.5 ps 1.5 pt 7
set style line 11 lc rgb '#21618C' lw 2.5 dt 3 ps 1.5 pt 7
set style line 12 lc rgb '#A93226' lw 2.5 dt 3 ps 1.5 pt 7
set style line 13 lc rgb '#186A3B' lw 2.5 dt 3 ps 1.5 pt 7

et = 8.5851
ehe = 7.4054
edd = 1.9527

set xrange[0:11]

set key spacing 1.2
set key bottom left 
set ytics format '%g'
set ytics -100,20,0
set lmargin 8
#==========================================================
set nogrid 
unset xlabel
set xtics format ''

#==========================================================
set ylabel "$\\delta(0^+)$ [Deg]"
#==========================================================
plot 'diagPH_1-1.dat' u ($1):($3>0?$3:$3) ls 1 w l t '$\delta_{11}$',\
	 'diagPH_2-2.dat' u ($1+et-ehe):3 ls 2 w l t '$\delta_{22}$',\
	 'diagPH_3-3.dat' u ($1+et-edd):3 ls 3 w l t '$\delta_{33}$',\
	 'scdiagPH_1-1.dat' u ($1):($3>0?$3:$3) ls 11 w l t '',\
	 'scdiagPH_2-2.dat' u ($1+et-ehe):3     ls 12 w l t '',\
	 'scdiagPH_3-3.dat' u ($1+et-edd):3     ls 13 w l t ''	 

set xlabel "$E_{\\textrm{c.m.}}$ [MeV]"
set tics font "Times-Roman,20"
set xtics format "$%g$"
set yrange[-100:100]
set ytics -110,30,110
set bmargin 4
#set size 1,0.5
plot 'diagPH_2-3.dat' u ($1+et-ehe):($3<0?$3:$3) ls 2 w l t '$\delta_{23}$',\
     'diagPH_1-2.dat' u ($1):($3<0?$3:$3) ls 3 w l t '$\delta_{12}$',\
	 'diagPH_1-3.dat' u ($1):($3<0?$3:$3) ls 1 w l t '$\delta_{13}$'


unset multiplot
#==========================================================
set output
! latex temp.tex && dvipdf temp.dvi
! mv temp.pdf 0p_phases.pdf
! rm temp*.*
#==========================================================

