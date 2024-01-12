% RGD for scalar-on-tensor regression


clear;
rng('default');
rng(2018);
p = 30;
r = 3;
r_use = 10;
n = floor(p^(3/2)) * r * 10;
sig = 0;
t_max = 300;
tol = 1e-13;
succ_tol = 1e-13;
init = 'good';
p1 = p; p2 = p; p3 = p;
r1 = r; r2 = r; r3 = r;
r1_use = r_use; r2_use = r_use; r3_use = r_use;
S = tensor(randn(r1, r2, r3));
E1 = randn(p1, r1);
E2 = randn(p2, r2);
E3 = randn(p3, r3);
[U1,~,~] = svds(E1,r1);
[U2,~,~] = svds(E2,r2);
[U3,~,~] = svds(E3,r3);
A = tensor(randn(p1, p2, p3,n));
U = {U1, U2, U3};
X = ttm(S, U, [1:3]);
eps = sig * randn(n,1);
A_mat = tenmat(A, 4);
A_mat = A_mat.data;
y = A_mat * X(:) + eps;


if strcmp(init,'good') 
    % initialization
    W = reshape(tensor(y' * A_mat),[p1, p2, p3])/n;
    init_result = hosvd(W,norm(W),'ranks',[r1_use,r2_use,r3_use],'sequential',false,'verbosity',0);
    Xt = ttm(init_result.core, init_result.u,[1:3]);
    Ut = init_result.u;
end

RGD_error = RGD_scalar_tensor_regression(A_mat,y,Xt, Ut,X, U, p1, p2, p3,r1_use,r2_use,r3_use, t_max, succ_tol );
array2table(RGD_error)



