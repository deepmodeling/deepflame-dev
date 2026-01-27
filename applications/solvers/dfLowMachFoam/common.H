#pragma once

#if WM_LABEL_SIZE == 32
#define MPI_LABEL MPI_INT
#elif WM_LABEL_SIZE == 64
#define MPI_LABEL MPI_LONG
#else
#error "WM_LABEL_SIZE must be 32 or 64 !!!"
#endif