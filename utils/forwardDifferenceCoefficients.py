#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Simple script to compute the coefficients of the forward difference approximation
import numpy as np
import sympy as sp
import argparse


parser = argparse.ArgumentParser(description="Compute forward difference coefficients.")
parser.add_argument('order', type=int, help='Order of the forward difference approximation')
ARG = parser.parse_args()

def forward_difference_coefficients(n):
    # Define symbols
    f0 = sp.symbols("f0")
    equations = []
    dsymbols = []
    fsymbols = [f0]
    for i in range(n):
        dsymbols.append(sp.symbols("d%d"%(i+1)))
        fsymbols.append(sp.symbols("f%d"%(i+1)))
    for i in range(n):
        equation = f0 - fsymbols[i+1]
        #length is i+1
        for j in range(n):
            if j==0:
                equation = equation + (i+1) * dsymbols[0]
            else:
                equation += dsymbols[j]*(i+1)**(j+1)/sp.factorial(j+1)
        equations.append(equation)
    
#    print(equations)
    solution = sp.solve(equations, dsymbols)
#    print(solution)
    d1_coefficients = [solution[d].expand() for d in dsymbols]
#    print(d1_coefficients)
    # Collect the coefficients of f0, f1, ..., fn for d1
    coefficients = []
    for coef in d1_coefficients:
        coeffs_dict = sp.collect(coef, fsymbols, evaluate=False)
        coefficients.append([coeffs_dict.get(f, 0) for f in fsymbols])
    
    return coefficients[0]  # Coefficients for d1

# Example usage:
order = ARG.order
coefficients = forward_difference_coefficients(order)
print(f"Coefficients for forward difference of order {order}: {coefficients}")

