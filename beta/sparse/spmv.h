#include <complex>
using namespace std;

#ifndef SPMV_H
#define SPMV_H

void zspmv(const int nrows,const int * IA,const int * JA,const double * data,
             const complex<double> * vec,complex<double> * out,const int nthr);

#endif
