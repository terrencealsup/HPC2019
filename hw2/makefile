CXX = g++
CXXFLAGS = -O3 -march=native -fopenmp -std=c++11

all: val_test01_solved val_test02_solved MMult1 omp_solved2 omp_solved3 omp_solved4 omp_solved5 omp_solved6 jacobi2D-omp gs2D-omp

val_test01_solved: val_test01_solved.cpp
	$(CXX) $(CXXFLAGS) val_test01_solved.cpp -o val_test01_solved

val_test02_solved: val_test02_solved.cpp
	$(CXX) $(CXXFLAGS) val_test02_solved.cpp -o val_test02_solved

MMult1: MMult1.cpp
	$(CXX) $(CXXFLAGS) MMult1.cpp -o MMult1

omp_solved2: omp_solved2.c
	$(CXX) $(CXXFLAGS) omp_solved2.c -o omp_solved2

omp_solved3: omp_solved3.c
	$(CXX) $(CXXFLAGS) omp_solved3.c -o omp_solved3

omp_solved4: omp_solved4.c
	$(CXX) $(CXXFLAGS) omp_solved4.c -o omp_solved4

omp_solved5: omp_solved5.c
	$(CXX) $(CXXFLAGS) omp_solved5.c -o omp_solved5

omp_solved6: omp_solved6.c
	$(CXX) $(CXXFLAGS) omp_solved6.c -o omp_solved6

jacobi2D-omp: jacobi2D-omp.cpp
	$(CXX) $(CXXFLAGS) jacobi2D-omp.cpp -o jacobi2D-omp

gs2D-omp: gs2D-omp.cpp
	$(CXX) $(CXXFLAGS) gs2D-omp.cpp -o gs2D-omp

clean:
	-rm -f val_test01_solved val_test02_solved MMult1 omp_solved2 omp_solved3 omp_solved4 omp_solved5 omp_solved6 jacobi2D-omp gs2D-omp

.PHONY: clean
