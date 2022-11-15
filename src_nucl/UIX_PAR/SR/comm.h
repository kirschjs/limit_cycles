#ifndef COMM_H
#define COMM_H

#include <stdlib.h>

#define MAX_PROCESSOR_NAME 255

#ifdef PC_FORTRAN

#define end_comm end_comm__
#define abort_comm abort_comm__
#define init_comm init_comm__
#define machine_size machine_size__
#define machine_id machine_id__
#define receive_double receive_double__
#define receive_int receive_int__
#define receive_long receive_long__
#define send_int send_int__
#define send_double send_double__
#define send_long send_long__

#endif

#ifdef SUN_FORTRAN

#define end_comm end_comm_
#define abort_comm abort_comm_
#define init_comm init_comm_
#define machine_size machine_size_
#define machine_id machine_id_
#define receive_double receive_double_
#define receive_int receive_int_
#define receive_long receive_long_
#define send_int send_int_
#define send_double send_double_
#define send_long send_long_

#endif

#ifdef CRAY_FORTRAN

#define end_comm END_COMM
#define abort_comm ABORT_COMM
#define init_comm INIT_COMM
#define machine_size MACHINE_SIZE
#define machine_id MACHINE_ID
#define receive_double RECEIVE_DOUBLE
#define receive_int RECEIVE_INT
#define receive_long RECEIVE_LONG
#define send_int SEND_INT
#define send_double SEND_DOUBLE
#define send_long SEND_LONG

#endif

void end_comm();
void abort_comm( int code);
void init_comm( int argc, char *argv[]);

char* machine_name(int *len);
void machine_size( int* );
void machine_id( int* );
void send_int( int *data, int *len, int *dest);
void send_double( double *data, int *len, int *dest);
void send_long( long *data, int *len, int *dest);
void receive_int( int *data, int *len, int *source, int *flag);
void receive_long( long *data, int *len, int *source, int *flag);
void receive_double( double *data, int *len, int *source, int *flag);

#endif
