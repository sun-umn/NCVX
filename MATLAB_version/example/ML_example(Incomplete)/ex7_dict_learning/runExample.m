function soln = runExample()
%   runExample: (examples/ex7)
%       Run GRANSO on a 2-variable nonsmooth Rosenbrock objective function,
%       subject to simple bound constraints, with GRANSO's default
%       parameters.
%    
%       Read this source code.
%   
%       This tutorial example shows:
%
%           - how to call GRANSO using objective and constraint functions
%             defined in .m files 
%       
%           - how to set GRANSO's inputs when there aren't any 
%             equality constraint functions (which also applies when there
%             aren't any inequality constraints)
%
%   USAGE:
%       soln = runExample();
% 
%   INPUT: [none]
%   
%   OUTPUT:
%       soln        GRANSO's output struct
%
%   See also objectiveFunction, inequalityConstraint. 

%% specify input variables 
% key: input variables
var = {'U','V'};
n_samples=100;
n_components=15;
n_features=20;

% value: dimension. e.g., 3 by 2 => [3,2]
dim = {[n_features,n_components],[n_components,n_samples]};
var_dim_map =  containers.Map(var, dim);

% calculate total number of scalar variables
nvar = 0;
for idx = 1:length(dim)
    curDim = dim(idx);
    nvar = nvar + curDim{1,1}(1)*curDim{1,1}(2);
end

% random init
rng('default');
U = rand(n_features,n_components); % dictionary
V = rand(n_components,n_samples); % dictionary

Y = U*V;

parameters.n_samples = n_samples;
parameters.n_components = n_components;
parameters.n_features = n_features;
parameters.Y = Y;
parameters.alpha = 1; % init value from scikit learn

rng(2021);
opts.x0 = rand(n_features * n_components+n_components*n_samples,1);

% opts.quadprog_opts.QPsolver = 'qpalm';
opts.quadprog_opts.QPsolver = 'quadprog';



%% call mat2vec to enable GRANSO using matrix input
tic
combined_fn = @(x) mat2vec(x,var_dim_map,nvar, parameters);
soln = granso(nvar,combined_fn,opts);
toc

U = soln.final.x(1:n_features * n_components);
V = soln.final.x(n_features * n_components + 1:end);
U = reshape(U,[n_features,n_components]);
V = reshape(V,[n_components,n_samples]);
f = .5*norm(Y - U*V, 'fro')^2

end