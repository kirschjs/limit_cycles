reset
#==========================================================
set terminal epslatex standalone color dl 3 lw 2 size 10cm,7cm
set terminal epslatex "Times-Roman" 11
set output 'temp.tex'
set size 1.0,1.0
set multiplot layout 1,1
#==========================================================
mevfm = 197.33
th    = -2.25
mn    = 938
#==========================================================
set key spacing 2.4
set key top right #at graph 0.99, graph 0.6
#unset key
set tics font "Times-Roman,20"
#set y2tics format "$%g$"
set ytics format "$%g$" nomirror
set format x '$%g$'
set lmargin 12
set bmargin 4
set logscale y 10
set style rect 

#==========================================================
set nogrid 
set border linewidth 1.
set style line 1 lc rgb '#0060ad' lt 1 lw 2 pt 7 pi -0 ps 1.0 #blueish
set style line 2 lc rgb '#00008b' lt 1 lw 2 pt 6 pi -0 ps 1.5
set style line 3 lc rgb '#a2142f' lt 1 lw 2 pt 7 pi -0 ps 1.0 #redish
set style line 5 lc rgb '#f03232' lt 1 lw 2 pt 6 pi -0 ps 1.5
set style line 4 lc rgb '#006400' lt 1 lw 2 pt 6 pi -0 ps 1.0

set pointintervalbox 3
#==========================================================
set xlabel '\Large 3-body LEC~[MeV]'
#set y2label '$\langle D|\hat{V}^{(1)}|D\rangle$~[MeV]'
#==========================================================
set style fill transparent solid 0.3
#==========================================================
set size 1,1
#set yrange[0.2:4.6]
set y2range[-1:2]
#set label '\Large $B(3,{}^2S_{1/2})$'  at graph 0.2,graph 0.36 front
#set label '\Large $B(4,0^+)$'          at graph 0.2,graph 0.81 front
#set label '$B(D)$'      at graph 0.2,graph 0.22 front
#set label '$B(T)$'      at graph 0.03,graph 0.35 front
#set label '\Large\framebox{3NF$=0$}' at graph 0.06,graph 0.86 front
set title ''
set ylabel '\Large Energy~[MeV]'
set origin 0,0
plot '/home/kirscher/kette_repo/limit_cycles/systems/3_B2-05_B3-8.00/4.00/123/spect/B3_of_TNI.dat' u ($1):($6<th?-$6:1/0) with linespoints ls 1 title '$B^{(0)}$' axes x1y1,\
     '/home/kirscher/kette_repo/limit_cycles/systems/3_B2-05_B3-8.00/4.00/123/spect/B3_of_TNI.dat' u ($1):($5<th?-$5:1/0) with linespoints ls 3 title '$B^{(1)}$' axes x1y1,\
     '/home/kirscher/kette_repo/limit_cycles/systems/3_B2-05_B3-8.00/4.00/123/spect/B3_of_TNI.dat' u ($1):($4<th?-$4:1/0) with linespoints ls 4 title '$B^{(2)}$' axes x1y1
#set ylabel '\Large $\Lambda^2\,C^{(1)}_2~$'
#set size 1,1
#set origin 1,0
#plot 'NLO_LECs.dat'  u ($1*mevfm/1000):($2/($1*mevfm)**3*($1*mevfm)**2) with linespoints ls 1 title '$C_2^{(1)}(^3S_1)$' axes x1y1,\
#	 'NLO_LECs.dat'  u ($1*mevfm/1000):(-$3/($1*mevfm)**3*($1*mevfm)**2) with linespoints ls 3 title '$C_2^{(1)}(^1S_0)$' axes x1y1,\
#	 'NLO_LECs.dat'  u ($1*mevfm/1000):($6) with linespoints ls 7 title '$B(D)$' axes x1y2
unset multiplot
#==========================================================
set output
! latex temp.tex && dvipdf temp.dvi
! mv temp.pdf B3_of_D.pdf
! rm temp*.*
#==========================================================
