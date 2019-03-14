/*
Terrence Alsup
HPC Spring 2019
March 13, 2019

Debugged code for Open MP.
The main change was to declare the variable sum outside of the functions.  This
is because the reduction requires a shared variable, however, each thread was
individually setting sum.
*/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define VECLEN 100

float a[VECLEN], b[VECLEN];
float sum; // Declare the variable outside the

float dotprod ()
{
int i,tid;
//float sum; //Variable is declared outside the method now.

tid = omp_get_thread_num();
#pragma omp for reduction(+:sum)
  for (i=0; i < VECLEN; i++)
    {
    sum = sum + (a[i]*b[i]);
    printf("  tid= %d i=%d\n",tid,i);
    }
}


int main (int argc, char *argv[]) {
int i;
//float sum; Variable is declared outside of main now.

for (i=0; i < VECLEN; i++)
  a[i] = b[i] = 1.0 * i;
sum = 0.0;

#pragma omp parallel shared(sum)
  dotprod();

printf("Sum = %f\n",sum);

}
