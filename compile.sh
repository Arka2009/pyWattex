#!/bin/bash


g++ -DTESTCVXBIN --std=c++11 -I${HOME}/anaconda3/include/python3.6m cvxoptimizernlopt.cpp -o textcvxbin -lnlopt -lm
g++ -c -fPIC --std=c++11 -I${HOME}/anaconda3/include/python3.6m cvxoptimizernlopt.cpp -o testcvxopt.o
g++ -shared -Wall -Werror -Wl,--export-dynamic testcvxopt.o -lboost_python36 -lnlopt -lm -o testcvxopt.so