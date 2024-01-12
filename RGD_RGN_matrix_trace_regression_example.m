clear all
tic;
rng('default');
rng(2021);
p1 = 100;
p2 = 100;
r = 3;
r_input = 10;
n = 10*r_input*p1;
sig = 0;
lambda = 3;

S = diag(repelem(lambda,r));
U = randn(p1,r);
[U,~,~] = svds(U,r);
V = randn(p2,r);
[V,~,~] = svds(V,r);

X = U*S*(V');

A = randn(n,p1,p2);
A = tensor(A);
A1 = tenmat(A,1);
eps = sig * randn(n,1);
y = A1.data*(X(:)) + eps;

% good initialization
tildeX = reshape(y'*A1.data,[p1,p2])/n;
[U0,Sigma0,V0] = svds(tildeX, r_input);
hatX = U0 * Sigma0 * V0';
iter_max = 30;
iter_max_GD = 300;
tol = 1e-13;
succ_tol = 1e-13;
retra_type = 'svd';
[RGN_error,succ_tag] = RGN_matrix_trace_regression(A, y, r_input, p1, p2, hatX,X, iter_max, tol, succ_tol, retra_type);
RGD_error = RGD_matrix_trace_regression(A1.data, y, r_input, p1, p2, hatX, X, iter_max_GD,tol);
array2table(RGN_error)
array2table(RGD_error)
time = toc;
time



