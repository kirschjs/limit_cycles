CC          = mpiicx
F77         = mpiifx

LIBS        = -I${MKLROOT}/include/mkl/intel64/lp64 -qmkl=parallel
FOPTS       = -Ofast -mcmodel=medium #-ffpe-summary=none -fallow-argument-mismatch #

CFLAGS      = -o $@ $? $(PRECFLAGS)

.c.o:
	$(CC) $(COPTS)  -c -o $*.o $< 

.f.o:
	$(F77) $(FOPTS) -c -o $*.o $< 

.F.o:
	$(F77) $(FOPTS)  -c -o $*.o $<

clean:
	rm -f *.o core


mpi_drqua_n_v18-uix.o:  mpi_drqua_n_v18-uix.F  par.h Makefile
mpi_drqua_m_v18-uix.o:  mpi_drqua_m_v18-uix.F  par.h Makefile
mpi_drqua_n_eft-cib.o:  mpi_drqua_n_eft-cib.F  par-cib.h Makefile
mpi_drqua_m_eft-cib.o:  mpi_drqua_m_eft-cib.F  par-cib.h Makefile


mpi_drqua_n_v18-uix: mpi_drqua_n_v18-uix.o
	$(F77) $(FFLAGS)  -o mpi_drqua_n_v18-uix mpi_drqua_n_v18-uix.o $(LIBS) $(MPI)

mpi_drqua_m_v18-uix: mpi_drqua_m_v18-uix.o
	$(F77) $(FFLAGS)  -o mpi_drqua_m_v18-uix mpi_drqua_m_v18-uix.o $(LIBS) $(MPI)

mpi_drqua_n_eft-cib: mpi_drqua_n_eft-cib.o
	$(F77) $(FFLAGS)  -o mpi_drqua_n_eft-cib mpi_drqua_n_eft-cib.o $(LIBS) $(MPI)

mpi_drqua_m_eft-cib: mpi_drqua_m_eft-cib.o
	$(F77) $(FFLAGS)  -o mpi_drqua_m_eft-cib mpi_drqua_m_eft-cib.o $(LIBS) $(MPI)

drsammel_v18-uix: drsammel_v18-uix.o  par.h
	$(F77) $(FOPTS)  -o $@ drsammel_v18-uix.o $(LIBS)

drsammel_eft-cib: drsammel_eft-cib.o  par-cib.h
	$(F77) $(FOPTS)  -o $@ drsammel_eft-cib.o $(LIBS)

all: mpi_drqua_n_v18-uix mpi_drqua_m_v18-uix mpi_drqua_n_eft-cib mpi_drqua_m_eft-cib drsammel_v18-uix drsammel_eft-cib clean
