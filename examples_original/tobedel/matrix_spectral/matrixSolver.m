function [f_vec,f_grad_vec,ci_vec,ci_grad_vec,ce_vec,ce_grad_vec] = matrixSolver(x,inputVarMap)

% x is vector
% output all vector: for granso package
% transform user provided matrix form into vector form

inputVar = keys(inputVarMap); % e.g., U,V
varDim = values(inputVarMap);
curIdx = 0;

for idx = 1:length(keys(inputVarMap))
    % current variable, e.g., U
    tmpVar = inputVar{1,idx};
    % corresponding dimension of the variable, e.g, 2 by 2
    tmpDim1 = varDim{1,idx}(1);
    tmpDim2 = varDim{1,idx}(2);
    % reshape vector input x in to matrix variables, e.g, X.U, X.V
    curIdx = curIdx + 1;
    X.(tmpVar) = reshape(x(curIdx:curIdx+tmpDim1*tmpDim2-1),tmpDim1,tmpDim2);
    curIdx = curIdx+tmpDim1*tmpDim2-1;
end

% calculate total number of scalar variables
nvar = 0;
for idx = 1:length(varDim)
    curDim = varDim(idx);
    nvar = nvar + curDim{1,1}(1)*curDim{1,1}(2);
end



[f,f_grad] = objectiveFunction_matrix(X);
[ci,ci_grad] = inequalityConstraint_matrix(X);

% obj function is scalar form
f_vec = f; 

f_grad_vec = zeros(nvar,1);
curIdx = 0;
for idx = 1:length(keys(inputVarMap))
    % current variable, e.g., U
    tmpVar = inputVar{1,idx};
    % corresponding dimension of the variable, e.g, 2 by 2
    tmpDim1 = varDim{1,idx}(1);
    tmpDim2 = varDim{1,idx}(2);
    
    
    % reshape vector input x in to matrix variables, e.g, X.U, X.V
    curIdx = curIdx + 1;
    f_grad_vec(curIdx:curIdx+tmpDim1*tmpDim2-1) = f_grad.(tmpVar)(:);
    curIdx = curIdx+tmpDim1*tmpDim2-1;
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nconstr = 0;
arrConstr = fieldnames(ci);
% get # of constraints
for iconstr = 1: length(arrConstr)
    % current constraint, e.g., c1, c2
    tmpconstr = arrConstr{iconstr};
    constrMatrix = ci.(tmpconstr);   
    
    nconstr = nconstr + length(constrMatrix(:));
end
    
ci_vec = zeros(nconstr,1);

curIdx = 0;
for iconstr = 1: length(arrConstr)
    % current constraint, e.g., c1, c2
    tmpconstr = arrConstr{iconstr};
    constrMatrix = ci.(tmpconstr);   
    curIdx = curIdx+1;
    
    ci_vec(curIdx:curIdx+length(constrMatrix(:))-1) = constrMatrix(:);
    curIdx = curIdx+length(constrMatrix(:))-1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ci_grad_vec = zeros(nvar,nconstr);
colIdx = 0;
% iterate column
for iconstr = 1: length(arrConstr)
    % current constraint, e.g., c1, c2
    tmpconstr = arrConstr{iconstr};
    constrMatrix = ci.(tmpconstr);
    % initialize
    rowIdx = 0;
    colIdx = colIdx+1;
    
    % iterate row: variables
    for idx = 1:length(keys(inputVarMap))
        % current variable, e.g., U
        tmpVar = inputVar{1,idx};
        ciGradMatrix = ci_grad.(tmpconstr).(tmpVar);
        
        
        % corresponding dimension of the variable, e.g, 2 by 2
        tmpDim1 = varDim{1,idx}(1);
        tmpDim2 = varDim{1,idx}(2);

        % reshape vector input x in to matrix variables, e.g, X.U, X.V
        rowIdx = rowIdx + 1; 
        
        ci_grad_vec(rowIdx:rowIdx+tmpDim1*tmpDim2-1,colIdx:colIdx+length(constrMatrix(:))-1) = ciGradMatrix;
        rowIdx = rowIdx +tmpDim1*tmpDim2-1;
         
    end
    colIdx = colIdx+length(constrMatrix(:))-1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% X.U = reshape(x(1:4),2,2);
% X.V = reshape(x(5:8),2,2);

% f_vec = f; 
% f_grad_vec = [f_grad.U(:);f_grad.V(:)];

% ci_vec = [ci.c1(:) ; ci.c2(:)];
% ci_grad_vec = diag([ci_grad(1).U(:);ci_grad(1).V(:)] + [ci_grad(2).U(:);ci_grad(2).V(:)]);


ce_vec = [];
ce_grad_vec = [];

end