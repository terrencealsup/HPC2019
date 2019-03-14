/*
Terrence Alsup
HPC Spring 2019
March 13, 2019

Debugged code using valgrind.
The only changes were:
1. Change n to (n+1) to allocate space for x[0] to x[n].
2. Change delete [] x to free(x) since malloc was used.
*/

# include <cstdlib>
# include <iostream>

using namespace std;

int main ( );
void f ( int n );

//****************************************************************************80

int main ( )

//****************************************************************************80
//
//  Purpose:
//
//    MAIN is the main program for TEST01.
//
//  Discussion:
//
//    TEST01 calls F, which has a memory "leak".  This memory leak can be
//    detected by VALGRID.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    18 May 2011
//
{
  int n = 10;

  cout << "\n";
  cout << "TEST01\n";
  cout << "  C++ version.\n";
  cout << "  A sample code for analysis by VALGRIND.\n";

  f ( n );
//
//  Terminate.
//
  cout << "\n";
  cout << "TEST01\n";
  cout << "  Normal end of execution.\n";

  return 0;
}
//****************************************************************************80

void f ( int n )

//****************************************************************************80
//
//  Purpose:
//
//    F computes N+1 entries of the Fibonacci sequence.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    18 May 2011
//
{
  int i;
  int *x;

  // Change n to (n+1) in the line below because we need to allocate space for
  // x[i] with i = 0,...,10.
  x = ( int * ) malloc ( (n+1) * sizeof ( int ) );

  x[0] = 1;
  cout << "  " << 0 << "  " << x[0] << "\n";

  x[1] = 1;
  cout << "  " << 1 << "  " << x[1] << "\n";

  for ( i = 2; i <= n; i++ )
  {
    x[i] = x[i-1] + x[i-2];
    cout << "  " << i << "  " << x[i] << "\n";
  }

  // Change delete [] x to free(x) since the space for x was allocated with
  // malloc.
  free(x);

  return;
}
