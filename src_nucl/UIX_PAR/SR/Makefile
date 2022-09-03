CC          = mpicc
#F77         = mpif90
F77  =ifort
#COPTS       = -O2 -Wall -ggdb3
COPTS       = -O2 
FOPTS       = -O3 -xW 
#FOPTS       = -O3 -tpp7 -xW -g -traceback
#FOPTS       = -O3 -g -traceback
#MAKRO       = -DPC_FORTRAN -DDEBUG_INT -DDEBUG_DOUBLE
#MAKRO       = -DPC_FORTRAN -DDEBUG_INT 
#MAKRO       = -DPC_FORTRAN 
MAKRO       = -DSUN_FORTRAN  
RM          = /bin/rm -f
#LIBS        = -lm -lf2c
#LIBS        = -L$(INTEL_F_HOME)/lib -Wl,-rpath,$(INTEL_F_HOME)/lib -lifcore -lsvml -lm 
LIBS        =  -L/opt/intel_fc_81-023/lib -lifcore -limf -Xlinker -rpath -Xlinker /opt/intel_fc_81-023/lib -Xlinker -rpath -Xlinker /opt/intel_cc_81-024/lib -lsvml
#LIBS        = -lm 
#MPI         = -lmpi

#    "$@" expands to the target; "$?" expands to the dependency list
CFLAGS      = -o $@ $? $(PRECFLAGS)

.c.o:
	$(CC) $(COPTS) $(MAKRO) -c -o $*.o $< $(MPIINC)

.f.o:
	$(F77) $(FOPTS) -c -o $*.o $< 

.F.o:
	$(F77) $(FOPTS) $(MAKRO) -c -o $*.o $< $(MPIINC)

tni_quaf.o: tni_quaf.F par/DRQUA_AK Makefile

tni_rechner.o: tni_rechner.F par/DRQUA_AK Makefile

mpi_comm.o: mpi_comm.c comm.h Makefile

pvm_comm.o: pvm_comm.c comm.h Makefile

haupt.o: haupt.c comm.h Makefile

zeit.o: zeit.c Makefile

change: change.o Makefile
	$(F77) $(FOPTS) $(MAKRO) -o change change.o 

SAMMEL: SAMMEL.o Makefile
	$(F77) $(FOPTS) $(MAKRO) -o SAMMEL SAMMEL.o 

aufsammel.exe: aufsammel.o Makefile
	$(F77) $(FOPTS) -o aufsammel.exe aufsammel.o

DRQUA_AK_N.o: DRQUA_AK_N.F par/DRQUA_AK Makefile

DRQUA_AK_N.exe: DRQUA_AK_N.o
	$(F77) $(FOPTS) $(MAKRO) -o DRQUA_AK_N.exe DRQUA_AK_N.o $(LIBS)

mpi_tniq: haupt.o tni_quaf.o tni_rechner.o mpi_comm.o  zeit.o 
	$(CC) $(COPTS) $(MAKRO) -o $@ haupt.o tni_quaf.o tni_rechner.o mpi_comm.o zeit.o  $(MPIINC) $(LIBS) $(MPI) 

pvm_tniq: haupt.o tni_quaf.o tni_rechner.o pvm_comm.o zeit.o
	$(CC) $(COPTS) $(MAKRO) -o $@ haupt.o tni_quaf.o tni_rechner.o pvm_comm.o zeit.o $(LIBS) $(PVM) 

quaakquaak.exe: quaakquaak.o Makefile
	$(CC) $(COPTS) -o $@ quaakquaak.o

clean:
	$(RM) *.o *.exe pvm_tniq mpi_tniq *~
