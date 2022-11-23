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


mpi_quaf_v6.o:  mpi_quaf_v6.F  par.h Makefile

mpi_drqua_i.o:  mpi_drqua_i.F  par.h Makefile

mpi_drqua_uix.o:  mpi_drqua_uix.F  par.h Makefile

mpi_quaf_v6: mpi_quaf_v6.o
	$(F77) $(FFLAGS) $(MAKRO) -o mpi_quaf_v6_l mpi_quaf_v6.o $(LIBS) $(MPI)

mpi_drqua_i: mpi_drqua_i.o
	$(F77) $(FFLAGS) $(MAKRO) -o mpi_drqua_i_l mpi_drqua_i.o $(LIBS) $(MPI)

mpi_drqua_uix: mpi_drqua_uix.o
	$(F77) $(FFLAGS) $(MAKRO) -o mpi_drqua_uix mpi_drqua_uix.o $(LIBS) $(MPI)

mpi_drqua_v7: mpi_drqua_v7.o
	$(F77) $(FFLAGS) $(MAKRO) -o mpi_drqua_v7 mpi_drqua_v7.o $(LIBS) $(MPI)

sammel: sammel.o
	$(F77) $(FOPTS) $(MAKRO) -o $@ sammel.o $(LIBS)

SAMMEL-uix: SAMMEL-uix.o  par.h
	$(F77) $(FOPTS) $(MAKRO) -o $@ SAMMEL-uix.o $(LIBS)

all: mpi_drqua_uix SAMMEL-uix clean
