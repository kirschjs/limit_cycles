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

plot 'np_phases.dat' u ($1):3 ls 1 w l t 'n-p $J^{\pi}=0^+$',\
	'd-d_phases.dat' u ($2):3 ls 3 w l t 'd-d $J^{\pi}=0^+$',\
	't-p_phases.dat' u ($1):3 ls 11 w l t 't-p $J^{\pi}=0^+$',\
	'he3-n_phases.dat' u ($1):3 ls 2 w l t 'he3-n $J^{\pi}=0^+$'

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
     'a_ratio_4.0.dat' u ($3):($4) ls 3 w l t 'd-d',\
     'a_ratio_4.0.dat' u ($7):($8) ls 11 w l t 'he3-n',\
     'a_ratio_4.0.dat' u ($5):($6) ls 2 w l t 't-p'


#
unset multiplot
#==========================================================
set output
! latex temp.tex && dvipdf temp.dvi
! mv temp.pdf 0p_phases.pdf
! rm temp*.*
#==========================================================

