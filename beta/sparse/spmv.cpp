#include <complex>
#include <iostream>
#include "omp.h"
#include "spmv.h"
using namespace std;

void zspmv(const int nrows,const int * IA,const int * JA,const double * data,
             const complex<double> * vec,complex<double> * out,const int nthr){
    int i,j,j0,jf;
    #pragma omp parallel for \
        private(i,j,j0,jf) \
        shared(IA,JA,data,vec,out) schedule(static) \
        num_threads(nthr)
    for (i=0; i<nrows; i++){
        j0 = IA[i];
        jf = IA[i+1];
        for (j=j0; j<jf; j++){
            out[i] += data[j]*vec[JA[j]];
        }
    }
    return;
}
