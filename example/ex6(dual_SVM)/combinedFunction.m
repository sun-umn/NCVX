function [f,f_grad,ci,ci_grad,ce,ce_grad] = combinedFunction(X,parameters)
%   combinedFunction: (example_mat/ex5)
%       Defines objective and inequality constraint functions, with their
%       respective gradients, for soft margin SVM optimization problem:
%
%           f(w,b,xi) = 0.5 || w ||^2 + C \sum_{n=1}^N xi_n,
%           s.t., yn(<w,xn>+b) >= 1-xi_n, xi_n >= O
%
%       GRANSO's convention is that the inequality constraints must be less
%       than or equal to zero to be satisfied.  For equality constraints,
%       GRANSO's convention is that they must be equal to zero.
%
%   USAGE:
%       [f,f_grad,ci,ci_grad,ce,ce_grad] = ...
%                               combinedFunction(X,parameters);
% 
%   INPUT:
%       X                   struct of optimization variables
%                           X.w is a vector (real-valued matrix, p by 1)
%                           X.xi is a vector (real-valued matrix, N by 1)
%                           X.b is a vector (real-valued matrix, N by 1)
%  
%   OUTPUT:
%       f                   value of the objective function at x
%                           scalar real value
% 
%       f_grad              struct of gradient of the objective function at x.
%                           f_grad.w is a vector (real-valued matrix, 1 by p)
%                           f_grad.xi is a vector (real-valued matrix, 1 by N)
%                           f_grad.b is a scalar (real-valued matrix, 1 by 1)
% 
%       ci                  struct of value of the inequality constraint at x.
%                           c.c1 is a vector (real-valued matrix, N by 1)
%                           c.c2 is a vector (real-valued matrix, N by 1)
% 
%       ci_grad             struct of gradient of the inequality constraint at x.
%                           c.c1.w is a real-valued matrix, p by N
%                           c.c1.xi is a real-valued matrix, N by N
%                           c.c1.b is a real-valued matrix, N by 1
%                           c.c2.w is a real-valued matrix, p by N
%                           c.c2.xi is a real-valued matrix, N by N
%                           c.c2.b is a real-valued matrix, N by 1
% 
%       ce                  value of the equality constraint(s) at x.
%                           [], since there aren't any equality constraints
%
%       ce_grad             gradient(s) of the equality constraint(s) at x.
%                           [], since there aren't any equality constraints
%
%

C = parameters.C; % regularaization parameter
x = parameters.data_x;
y = parameters.label_r;
N = parameters.N;

% input variable, matirx form
alpha = X.alpha;

% objective function
yalphax = y .* alpha .* x; % N by p
f = 0.5 * sum(sum(yalphax * yalphax')) - sum(alpha);

% gradient of objective function, matrix form
f_grad.alpha = sum((yalphax * yalphax') ./ alpha, 2) - 1;

% inequality constraint, matrix form
ci.c1 = -alpha;
ci.c2 = alpha - C;

% gradient of inequality constraint, matrix form
ci_grad.c1.alpha = -speye(N);
ci_grad.c2.alpha = speye(N);

% equality constraint 
ce = sum(y .* alpha);
ce_grad = y;

end