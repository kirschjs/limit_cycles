CC          = mpicc
F77         = mpifort

FOPTS  =       -Ofast -ffpe-summary=none -fallow-argument-mismatch -mcmodel=medium 

#COPTS       = -O2 -Wall -ggdb3
#COPTS       = -O2
# option -xW generates code exclusivly for P4 processors
#FOPTS       = -O2 -xW -mcmodel medium -shared-intel
#MAKRO       = -DPC_FORTRAN 
#-DDEBUG_INT -DDEBUG_DOUBLE
#RM          = /bin/rm -f
#LIBS    =       -L/home/hmh/lib -L/opt/intel_mkl_61/lib -leispack -lsu2 -lmkl -lmkl_lapack -lmkl_lapack32 -lguide -lblas
#    "$@" expands to the target; "$?" expands to the dependency list
CFLAGS      = -o $@ $? $(PRECFLAGS)

.c.o:
	$(CC) $(COPTS) $(MAKRO) -c -o $*.o $< 

.f.o:
	$(F77) $(FOPTS) -c -o $*.o $< 

.F.o:
	$(F77) $(FOPTS) $(MAKRO) -c -o $*.o $< 

mpi_quaf_n_v18-uix.o: mpi_quaf_n_v18-uix.F par.h Makefile
mpi_quaf_m_v18-uix.o: mpi_quaf_m_v18-uix.F par.h Makefile	
mpi_quaf_eft.o: mpi_quaf_eft.F par-eft.h Makefile
mpi_quaf_n_eft-cib.o: mpi_quaf_n_eft-cib.F par-cib.h Makefile

mpi_quaf_n_v18-uix: mpi_quaf_n_v18-uix.o 
	$(F77) $(FOPTS) $(MAKRO) -o $@ mpi_quaf_n_v18-uix.o   $(LIBS) $(MPI)

mpi_quaf_n_eft-cib: mpi_quaf_n_eft-cib.o 
	$(F77) $(FOPTS) $(MAKRO) -o $@ mpi_quaf_n_eft-cib.o   $(LIBS) $(MPI) 

mpi_quaf_m_v18-uix: mpi_quaf_m_v18-uix.o 
	$(F77) $(FOPTS) $(MAKRO) -o $@ mpi_quaf_m_v18-uix.o   $(LIBS) $(MPI) 

mpi_quaf_eft: mpi_quaf_eft.o 
	$(F77) $(FOPTS) $(MAKRO) -o $@ mpi_quaf_eft.o   $(LIBS) $(MPI) 

mpi_quaf_n_v18-uix_m1: mpi_quaf_n_v18-uix_m1.o 
	$(F77) $(FOPTS) $(MAKRO) -o $@ mpi_quaf_n_v18-uix_m1.o   $(LIBS) $(MPI) 

mpi_quaf_m_v18-uix_m1: mpi_quaf_m_v18-uix_m1.o 
	$(F77) $(FOPTS) $(MAKRO) -o $@ mpi_quaf_m_v18-uix_m1.o   $(LIBS) $(MPI) 

mpi_quaf_eft_m1: mpi_quaf_eft_m1.o 
	$(F77) $(FOPTS) $(MAKRO) -o $@ mpi_quaf_eft_m1.o   $(LIBS) $(MPI)

sammel_v18-uix: sammel_v18-uix.o
	$(F77) $(FOPTS) $(MAKRO) -o $@ sammel_v18-uix.o $(LIBS)

sammel_eft-cib: sammel_eft-cib.o
	$(F77) $(FOPTS) $(MAKRO) -o $@ sammel_eft-cib.o $(LIBS)

clean:
	$(RM) *.o *.exe pvm_tniq *~

all: mpi_quaf_n_eft-cib mpi_quaf_n_v18-uix mpi_quaf_n_v18-uix_m1 mpi_quaf_m_v18-uix mpi_quaf_m_v18-uix_m1 mpi_quaf_eft mpi_quaf_eft_m1 sammel_eft-cib sammel_v18-uix clean
