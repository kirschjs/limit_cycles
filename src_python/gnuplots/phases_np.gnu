reset
#==========================================================
set terminal epslatex standalone color dl 3 lw 2 size 10cm,12cm
set terminal epslatex "Times-Roman" 11
set output 'temp.tex'

set size 1,1.8
set multiplot layout 3,1

et = 8.5851	
ehe = 7.4054
edd = 1.9527

set key spacing 1.
set key at screen 1, graph 1

set rmargin 14
set lmargin 8

set ytics format '%g'
#==========================================================
set size 1,0.6
set origin 0,1.15
set tmargin 0
set bmargin 4

set label sprintf('$\Lambda=%s\,\textrm{fm}^{-1}$',lambda) at graph 0.1,graph 0.1

set nogrid 

set xtics format '%g'

set xlabel "$E_{\\textrm{\\scriptsize c.m.}}$ [MeV]"
set ylabel "$\\delta$ [Deg]"

plot sprintf('np3s_phases_%s.dat',lambda)  u ($1):3 lc rgb  '#21618C' lw 1.5 ps 1.5 pt 7 w l  t 'np ${}^3S_1$',\
	 sprintf('np1s_phases_%s.dat',lambda)  u ($1):3 lc rgb  '#21618C' lw 1.5 ps 0.5 pt 1 w lp t 'np ${}^1S_0$',\
	 sprintf('d-d_phases_%s.dat',lambda)   u ($2):3 lc rgb  '#A93226' lw 1.5 ps 1.5 pt 7 w l  t 'd-d',\
	 sprintf('dq-dq_phases_%s.dat',lambda) u ($2):3 lc rgb  '#A93226' lw 1.5 ps 0.5 pt 8 w lp t 'dq-dq',\
	 sprintf('t-p_phases_%s.dat',lambda)   u ($2):3 lc rgb  '#186A3B' lw 1.5 ps 1.5 pt 7 w l  t 't-p',\
	 sprintf('he3-n_phases_%s.dat',lambda) u ($2):3 lc rgb  '#186A3B' lw 1.5 ps 0.5 pt 8 w lp t 'he3-n'
#==========================================================
set size 1,0.6
set origin 0,0.55
set tmargin 0
set bmargin 4

set xlabel "$E^{\\textrm{\\scriptsize match}}_{\\textrm{\\scriptsize c.m.}}$ [MeV]"
set ylabel "scattering length [fm]"

set ytics 0.,1.,12    format "$%g$" nomirror
set yrange[0.:9]
set xrange[0.:1.05]

plot ratfile u ($1):($2) lc rgb  '#21618C' lw 1.5 ps 1.5 pt 7 w l  t 'np ${}^3S_1$',\
	 ratfile u ($1):($3) lc rgb  '#21618C' lw 1.5 ps 0.5 pt 1 w lp t 'np ${}^1S_0$',\
     ratfile u ($1):($4) lc rgb  '#A93226' lw 1.5 ps 1.5 pt 7 w l  t 'd-d',\
     ratfile u ($1):($5) lc rgb  '#A93226' lw 1.5 ps 0.5 pt 8 w lp t 'dq-dq',\
     ratfile u ($1):($6) lc rgb  '#186A3B' lw 1.5 ps 1.5 pt 7 w l  t 't-p',\
     ratfile u ($1):($7) lc rgb  '#186A3B' lw 1.5 ps 0.5 pt 8 w lp t 'he3-n'


#==========================================================
set size 1,0.5
set origin 0,0.0
set tmargin 0
set bmargin 4

set xlabel "$E^{\\textrm{\\scriptsize match}}_{\\textrm{\\scriptsize c.m.}}$ [MeV]"
set ylabel ""

set ytics 0.,.1,1.2    format "$%g$" nomirror
set yrange[0.2:1.2]
set autoscale x

plot ratfile u ($1):($4/$2) lc rgb  '#21618C' lw 1.5 ps 1.5 pt 7 w l   t '$a_{dd}/a_{np}$',\
	 ratfile u ($1):($4/$2) lc rgb  '#21618C' lw 1.5 ps 0.5 pt 1 w lp  t '$a_{dqdq}/a_{np}$'

#
unset multiplot
#==========================================================
set output
! latex temp.tex && dvipdf temp.dvi
! mv temp.pdf 0p_phases.pdf
! rm temp*.*
#==========================================================

