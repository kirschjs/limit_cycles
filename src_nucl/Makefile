# hexpresso
F77     =      mpifort 
FFLAGS  =       -Ofast -ffpe-summary='none' -fallow-argument-mismatch # -mcmodel=medium 
LIBS    = -L -lgfortran -L/usr/lib/x86_64-linux-gnu/lapack -llapack -lblas
#LIBS    = -L${MKLROOT} -llapack -lblas  -lm -ldl

#theogate
#F77     =       ifort
#FFLAGS  =       -Ofast -ffpe-summary='none'  -mcmodel=large
#LIBS    = -L${MKLROOT} -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5

# ohne -g und -traceback erzeugt spoleinv runtime error in smatest,rrandb subroutinen wenn auf die variable nbess zugegriffen wird;
# mit den optionen tritt der fehler nicht auf;

.f.o:
	$(F77) $(FFLAGS) -c -o $*.o $<


clean:
	rm -f *.o core

KOBER_eft-cib: KOBER_eft-cib.o par/KOBER_EFT
	$(F77) $(FFLAGS) -o KOBER_eft-cib.exe KOBER_eft-cib.o $(LIBS)

LUDW_eft-cib: LUDW_eft-cib.o par/LUCN_EFT
	$(F77) $(FFLAGS) -o LUDW_eft-cib.exe LUDW_eft-cib.o $(LIBS)

QUAFL_N_eft-cib: QUAFL_N_eft-cib.o par/QUAF_EFT
	$(F77) $(FFLAGS) -o QUAFL_N_eft-cib.exe QUAFL_N_eft-cib.o

QUAFL_N_eft-cib_pop: QUAFL_N_eft-cib_pop.o par/QUAF_EFT
	$(F77) $(FFLAGS) -o QUAFL_N_eft-cib_pop.exe QUAFL_N_eft-cib_pop.o

DROBER_eft-cib: DROBER_eft-cib.o par/DROBER
	$(F77) $(FFLAGS) -o DROBER_eft-cib.exe DROBER_eft-cib.o $(LIBS)

DRLUD_eft-cib: DRLUD_eft-cib.o par/DRLUD
	$(F77) $(FFLAGS) -o DRLUD_eft-cib.exe DRLUD_eft-cib.o $(LIBS)

DRQUA_N_eft-cib: DRQUA_N_eft-cib.o par/DRQUA_AK
	$(F77) $(FFLAGS) -o DRQUA_N_eft-cib.exe DRQUA_N_eft-cib.o $(LIBS)

DRQUA_N_eft-cib_pop: DRQUA_N_eft-cib_pop.o par/DRQUA_AK
	$(F77) $(FFLAGS) -o DRQUA_N_eft-cib_pop.exe DRQUA_N_eft-cib_pop.o $(LIBS)

DR2END_eft-cib: DR2END_eft-cib.o par/DR2END_EFT
	$(F77) $(FFLAGS) -o DR2END_eft-cib.exe DR2END_eft-cib.o $(LIBS)

DR2END_eft-cib_pop: DR2END_eft-cib_pop.o par/DR2END_EFT
	$(F77) $(FFLAGS) -o DR2END_eft-cib_pop.exe DR2END_eft-cib_pop.o $(LIBS)

TDR2END_eft-cib: TDR2END_eft-cib.o par/DR2END_EFT
	$(F77) $(FFLAGS) -o TDR2END_eft-cib.exe TDR2END_eft-cib.o $(LIBS)

TDR2END_eft-cib_pop: TDR2END_eft-cib_pop.o par/DR2END_EFT
	$(F77) $(FFLAGS) -o TDR2END_eft-cib_pop.exe TDR2END_eft-cib_pop.o $(LIBS)

KOBER_v18-uix: KOBER_v18-uix.o par/KOBER
	$(F77) $(FFLAGS) -o KOBER_v18-uix.exe KOBER_v18-uix.o $(LIBS)

LUDW_v18-uix: LUDW_v18-uix.o par/LUCN
	$(F77) $(FFLAGS) -o LUDW_v18-uix.exe LUDW_v18-uix.o $(LIBS)

QUAFL_N_v18-uix: QUAFL_N_v18-uix.o par/QUAF
	$(F77) $(FFLAGS) -o QUAFL_N_v18-uix.exe QUAFL_N_v18-uix.o

QUAFL_N_v18-uix_pop: QUAFL_N_v18-uix_pop.o par/QUAF
	$(F77) $(FFLAGS) -o QUAFL_N_v18-uix_pop.exe QUAFL_N_v18-uix_pop.o

DROBER_v18-uix: DROBER_v18-uix.o par/DROBER
	$(F77) $(FFLAGS) -o DROBER_v18-uix.exe DROBER_v18-uix.o $(LIBS)

DRLUD_v18-uix: DRLUD_v18-uix.o par/DRLUD
	$(F77) $(FFLAGS) -o DRLUD_v18-uix.exe DRLUD_v18-uix.o $(LIBS)

DRQUA_N_v18-uix: DRQUA_N_v18-uix.o par/DRQUA_AK
	$(F77) $(FFLAGS) -o DRQUA_N_v18-uix.exe DRQUA_N_v18-uix.o $(LIBS)

DRQUA_N_v18-uix_pop: DRQUA_N_v18-uix_pop.o par/DRQUA_AK
	$(F77) $(FFLAGS) -o DRQUA_N_v18-uix_pop.exe DRQUA_N_v18-uix_pop.o $(LIBS)

DR2END_v18-uix: DR2END_v18-uix.o par/DR2END_AK
	$(F77) $(FFLAGS) -o DR2END_v18-uix.exe DR2END_v18-uix.o $(LIBS)

DR2END_v18-uix_pop: DR2END_v18-uix_pop.o par/DR2END_AK
	$(F77) $(FFLAGS) -o DR2END_v18-uix_pop.exe DR2END_v18-uix_pop.o $(LIBS)	

TDR2END_v18-uix: TDR2END_v18-uix.o par/DR2END_AK
	$(F77) $(FFLAGS) -o TDR2END_v18-uix.exe TDR2END_v18-uix.o $(LIBS)

TDR2END_v18-uix_pop: TDR2END_v18-uix_pop.o par/DR2END_AK
	$(F77) $(FFLAGS) -o TDR2END_v18-uix_pop.exe TDR2END_v18-uix_pop.o $(LIBS)

QUAFL_M_v18-uix: QUAFL_M_v18-uix.o par/QUAF
	$(F77) $(FFLAGS) -o QUAFL_M_v18-uix.exe QUAFL_M_v18-uix.o

DRQUA_M_v18-uix: DRQUA_M_v18-uix.o par/DRQUA_AK
	$(F77) $(FFLAGS) -o DRQUA_M_v18-uix.exe DRQUA_M_v18-uix.o $(LIBS)

S-POLE_PdP: S-POLE_PdP.o par/verma coul.o
	$(F77) $(FFLAGS) -o S-POLE_PdP.exe S-POLE_PdP.o coul.o $(LIBS)

S-POLE_zget: S-POLE_zget.o par/verma coul.o
	$(F77) $(FFLAGS) -o S-POLE_zget.exe S-POLE_zget.o coul.o $(LIBS)

all: KOBER_eft-cib LUDW_eft-cib QUAFL_N_eft-cib QUAFL_N_eft-cib_pop DROBER_eft-cib DRLUD_eft-cib DRQUA_N_eft-cib DR2END_eft-cib DR2END_eft-cib_pop KOBER_v18-uix LUDW_v18-uix QUAFL_N_v18-uix QUAFL_N_v18-uix_pop DROBER_v18-uix DRLUD_v18-uix DRQUA_N_v18-uix DRQUA_N_v18-uix_pop TDR2END_v18-uix TDR2END_v18-uix_pop DR2END_v18-uix DR2END_v18-uix_pop QUAFL_M_v18-uix DRQUA_M_v18-uix S-POLE_PdP S-POLE_zget DRQUA_N_eft-cib_pop TDR2END_eft-cib TDR2END_eft-cib_pop clean
