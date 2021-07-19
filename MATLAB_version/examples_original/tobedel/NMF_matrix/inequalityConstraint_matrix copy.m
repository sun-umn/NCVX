function [ci,ci_grad] = inequalityConstraint_matrix(X)
 
    U = X.U;
    V = X.V;

    ci(1).U = -U;
    ci(1).V = zeros(2);
    ci(2).U = zeros(2);
    ci(2).V = -V;
    
    
    ci_grad(1).U = -[1 1; 1 1];
    ci_grad(1).V = [0 0; 0 0];
    
    ci_grad(2).U = [0 0; 0 0];
    ci_grad(2).V = -[1 1; 1 1];
    
%     ci = -x;
%     ci_grad = -eye(8);
end