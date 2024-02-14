CC          = mpiicx
F77         = mpiifx

FOPTS  =       -Ofast -mcmodel=medium #-ffpe-summary=none -fallow-argument-mismatch 

CFLAGS      = -o $@ $? $(PRECFLAGS)

.c.o:
	$(CC) $(COPTS)  -c -o $*.o $< 

.f.o:
	$(F77) $(FOPTS) -c -o $*.o $< 

.F.o:
	$(F77) $(FOPTS)  -c -o $*.o $< 

mpi_quaf_n_v18-uix.o: mpi_quaf_n_v18-uix.F par.h Makefile
mpi_quaf_m_v18-uix.o: mpi_quaf_m_v18-uix.F par.h Makefile	
mpi_quaf_eft.o: mpi_quaf_eft.F par-eft.h Makefile
mpi_quaf_n_eft-cib.o: mpi_quaf_n_eft-cib.F par-cib.h Makefile

mpi_quaf_n_v18-uix: mpi_quaf_n_v18-uix.o 
	$(F77) $(FOPTS)  -o $@ mpi_quaf_n_v18-uix.o   $(LIBS) $(MPI)

mpi_quaf_n_eft-cib: mpi_quaf_n_eft-cib.o 
	$(F77) $(FOPTS)  -o $@ mpi_quaf_n_eft-cib.o   $(LIBS) $(MPI) 

mpi_quaf_m_v18-uix: mpi_quaf_m_v18-uix.o 
	$(F77) $(FOPTS)  -o $@ mpi_quaf_m_v18-uix.o   $(LIBS) $(MPI) 

mpi_quaf_eft: mpi_quaf_eft.o 
	$(F77) $(FOPTS)  -o $@ mpi_quaf_eft.o   $(LIBS) $(MPI) 

mpi_quaf_n_v18-uix_m1: mpi_quaf_n_v18-uix_m1.o 
	$(F77) $(FOPTS)  -o $@ mpi_quaf_n_v18-uix_m1.o   $(LIBS) $(MPI) 

mpi_quaf_m_v18-uix_m1: mpi_quaf_m_v18-uix_m1.o 
	$(F77) $(FOPTS)  -o $@ mpi_quaf_m_v18-uix_m1.o   $(LIBS) $(MPI) 

mpi_quaf_eft_m1: mpi_quaf_eft_m1.o 
	$(F77) $(FOPTS)  -o $@ mpi_quaf_eft_m1.o   $(LIBS) $(MPI)

sammel_v18-uix: sammel_v18-uix.o
	$(F77) $(FOPTS)  -o $@ sammel_v18-uix.o $(LIBS)

sammel_eft-cib: sammel_eft-cib.o
	$(F77) $(FOPTS)  -o $@ sammel_eft-cib.o $(LIBS)

clean:
	$(RM) *.o *.exe pvm_tniq *~

all: mpi_quaf_n_eft-cib mpi_quaf_n_v18-uix mpi_quaf_n_v18-uix_m1 mpi_quaf_m_v18-uix mpi_quaf_m_v18-uix_m1 mpi_quaf_eft mpi_quaf_eft_m1 sammel_eft-cib sammel_v18-uix clean
