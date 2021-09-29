#!/usr/bin/env python3.7

# Copyright 2021, Gurobi Optimization, LLC

# This example formulates and solves the following simple QP model:
#  minimize
#      x^2 + x*y + y^2 + y*z + z^2 + 2 x
#  subject to
#      x + 2 y + 3 z >= 4
#      x +   y       >= 1
#      x, y, z non-negative
#
# It solves it once as a continuous model, and once as an integer model.

import gurobipy as gp
from gurobipy import GRB

# Create a new model
m = gp.Model("qp")

# Create variables
x = m.addVar(ub=1.0, name="x")
y = m.addVar( ub=1.0, name="y" )
z = m.addVar(  ub=1.0, name="z" )

# Set objective: x^2 + x*y + y^2 + y*z + z^2 + 2 x
obj = -0.3 * x**2 + x*y + y**2 + y*z + z**2 + 2*x
m.setObjective(obj)

# Add constraint: x + 2 y + 3 z <= 4
m.addConstr(x + 2 * y + 3 * z >= 4, "c0")

# Add constraint: x + y >= 1
m.addConstr(x + y >= 1, "c1")
m.params.NonConvex = 2

m.optimize()

for v in m.getVars():
    print('%s %g' % (v.varName, v.x))

print('Obj: %g' % obj.getValue())

x.vType = GRB.INTEGER
y.vType = GRB.INTEGER
z.vType = GRB.INTEGER

m.optimize()

for v in m.getVars():
    print('%s %g' % (v.varName, v.x))

print('Obj: %g' % obj.getValue())