% Order-3 scalar-on-tensor regression via RGD.
% Output: iteration, time and error
% Inputs: A_mat: n-by-p1p2p3 matrix, each row is a vectorization of the covariate tensor; y: observation; X0: initialization of X; U0: initialization of loads
% X: true parameter of interest; U: true loading factors of X; p1,p2,p3: dimensions; r1,r2,r3: input ranks;
% iter_max: the maximum number of iteration; succ_tol: stopping criteria

function [ error_matrix ] = RGD_scalar_tensor_regression( A_mat,y,X0, U0,X, U, p1, p2, p3, r1,r2,r3, iter_max, succ_tol)
% Here we use HOSVD as the retraction.
Xt = X0;
Ut = U0;
St = ttm(Xt,{Ut{1}', Ut{2}' Ut{3}'});
Ut_perp = cell(3,1);
Vt = cell(3,1);
for i = 1:3
    Ut_perp{i} = null(Ut{i}');
    [Vt{i},~] = qr(double(tenmat(St,i))',0);
end
%Ut_err = sinq_norm(Ut{1},U{1}, Inf);
Xt_err = norm(tensor(Xt) - X)/norm(X);

error_matrix = [0, Xt_err, 0];
tic;
for iter = 1:iter_max    
    haty = A_mat * Xt(:);
    Z = reshape( (haty - y)' * A_mat, [p1, p2, p3] );
    Z = tensor(Z);
    grad_core = ttm( ttm( Z, {Ut{1}',Ut{2}', Ut{3}'} ), Ut, [1:3]  );
    Wt1 = kron( Ut{3}, Ut{2} ) * Vt{1};
    Wt2 = kron( Ut{3}, Ut{1} ) * Vt{2};
    Wt3 = kron( Ut{2}, Ut{1} ) * Vt{3};
    grad_arm1 = tensor( Ut_perp{1} * Ut_perp{1}' * double(tenmat(Z,1)) * (Wt1) * (Wt1'), [p1, p2, p3]  );
    grad_arm2 = permute( tensor( Ut_perp{2} * Ut_perp{2}' * double(tenmat(Z,2)) * (Wt2) * (Wt2'), [p2, p1, p3] ) , [2,1,3]  );
    grad_arm3 = permute( tensor( Ut_perp{3} * Ut_perp{3}' * double(tenmat(Z,3)) * (Wt3) * (Wt3'), [p3, p1, p2] ) , [2,3,1]  ); % [2,3,1] means put dim 2 to be mode 1, dim 3 to be mode 2, dim 1 to be mode 3
    grad = grad_core + grad_arm1 + grad_arm2 + grad_arm3;
    alpha_t = (norm(grad))^2 / ( norm(A_mat* grad(:)) )^2;
    tildeXt = Xt - alpha_t * grad;
    Xt = hosvd(tildeXt,norm(tildeXt),'ranks',[r1,r2,r3],'sequential',true,'verbosity',0);
    Ut = Xt.u;
    St = Xt.core;
    for i = 1:3
        Ut_perp{i} = null(Ut{i}');
        [Vt{i},~] = qr(double(tenmat(St,i))',0);
    end
    Xt = tensor(Xt);
    Xt_err_new = norm(Xt - X)/norm(X);
    if (Xt_err_new > Xt_err & i >= 100) || (Xt_err - Xt_err_new)/Xt_err < 1e-5
        break
    else
        Xt_err = Xt_err_new;
    end
    time = toc;
    iter_result = [iter, Xt_err, time];
    error_matrix = vertcat(error_matrix, iter_result);
    if Xt_err < succ_tol || Xt_err > 50
            break
    end
end
end