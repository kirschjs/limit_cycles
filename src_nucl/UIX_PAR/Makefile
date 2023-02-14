CC          = mpicc
F77         = mpifort
#MAKRO       = -DSUN_FORTRAN
#MAKRO       = -DPC_FORTRAN
#LIBS        = -parallel -static -L/opt/intel/mkl52sp1/lib/32 -lmkl_lapack -lmkl_p4
FOPTS       = -Ofast -ffpe-summary=none -fallow-argument-mismatch #-mcmodel=medium 
# -mcmodel=large
#LIBS        = -L/home/kirscher/lib/intel/mkl/10.0.1.014/lib/em64t -lmkl_em64t -lguide

LIBS    = -L -lgfortran #-static-libgfortran
CFLAGS      = -o $@ $? $(PRECFLAGS)

.c.o:
	$(CC) $(COPTS) $(MAKRO) -c -o $*.o $< 

.f.o:
	$(F77) $(FOPTS) -c -o $*.o $< 

.F.o:
	$(F77) $(FOPTS) $(MAKRO) -c -o $*.o $<

clean:
	rm -f *.o core


mpi_drqua_n_v18-uix.o:  mpi_drqua_n_v18-uix.F  par.h Makefile
mpi_drqua_m_v18-uix.o:  mpi_drqua_m_v18-uix.F  par.h Makefile
mpi_drqua_n_eft-cib.o:  mpi_drqua_n_eft-cib.F  par-cib.h Makefile
mpi_drqua_m_eft-cib.o:  mpi_drqua_m_eft-cib.F  par-cib.h Makefile


mpi_drqua_n_v18-uix: mpi_drqua_n_v18-uix.o
	$(F77) $(FFLAGS) $(MAKRO) -o mpi_drqua_n_v18-uix mpi_drqua_n_v18-uix.o $(LIBS) $(MPI)

mpi_drqua_m_v18-uix: mpi_drqua_m_v18-uix.o
	$(F77) $(FFLAGS) $(MAKRO) -o mpi_drqua_m_v18-uix mpi_drqua_m_v18-uix.o $(LIBS) $(MPI)

mpi_drqua_n_eft-cib: mpi_drqua_n_eft-cib.o
	$(F77) $(FFLAGS) $(MAKRO) -o mpi_drqua_n_eft-cib mpi_drqua_n_eft-cib.o $(LIBS) $(MPI)

mpi_drqua_m_eft-cib: mpi_drqua_m_eft-cib.o
	$(F77) $(FFLAGS) $(MAKRO) -o mpi_drqua_m_eft-cib mpi_drqua_m_eft-cib.o $(LIBS) $(MPI)

drsammel_v18-uix: drsammel_v18-uix.o  par.h
	$(F77) $(FOPTS) $(MAKRO) -o $@ drsammel_v18-uix.o $(LIBS)

drsammel_eft-cib: drsammel_eft-cib.o  par-cib.h
	$(F77) $(FOPTS) $(MAKRO) -o $@ drsammel_eft-cib.o $(LIBS)

all: mpi_drqua_n_v18-uix mpi_drqua_m_v18-uix mpi_drqua_n_eft-cib mpi_drqua_m_eft-cib drsammel_v18-uix drsammel_eft-cib clean