% RGD for order-4 tensor-on-vector regression
% Output: iteration, time and error
% Input: A: n-by-p1 covariate matrix; Y: observed tensors; X0: initialization; X: true parameter of interest; U: true loadings of X
% p1-p4: dimensions; r1_use-r4_use: input ranks; iter_max: iteration number; succ_tol: successful stopping criteria.

function [ error_matrix ] = RGD_tensor_vector_regression( A,Y,X0, U0,X, U, p1, p2, p3, p4, r1_use,r2_use,r3_use, r4_use, iter_max, succ_tol)
% Here we use HOSVD as the retraction.
Xt = X0; 
Ut = U0;
St = ttm(Xt,{Ut{1}', Ut{2}', Ut{3}', Ut{4}'});
Ut_perp = cell(4,1);
Vt = cell(4,1);
for i = 1:4
    Ut_perp{i} = null(Ut{i}');
    [Vt{i},~] = qr(double(tenmat(St,i))',0);
end   
Xt_err = norm(tensor(Xt) - X)/norm(X);
error_matrix = [0, Xt_err, 0];
tic;
for iter = 1:iter_max % Here we are operating on a order-4 parameter tensor, so it does takes more time.
    Z = ttm((ttm(Xt,A,1) - Y),A',1);
    ZU12 = ttm(Z,{Ut{1}', Ut{2}'},[1,2]);
    ZU34 = ttm(Z,{Ut{3}', Ut{4}'},[3,4]);
    grad_core = ttm( ttm( ZU12, { Ut{3}', Ut{4}'}, [3,4] ), Ut, [1:4]  );
    
    B1 = Ut_perp{1} * Ut_perp{1}' * double( tenmat(ttm(ZU34,{Ut{2}'},2 ), 1 ) ) * Vt{1};
    B2 = Ut_perp{2} * Ut_perp{2}' * double( tenmat(ttm(ZU34,{Ut{1}'},1 ), 2 ) ) * Vt{2};
    B3 = Ut_perp{3} * Ut_perp{3}' * double( tenmat(ttm(ZU12,{Ut{4}'},4 ), 3 ) ) * Vt{3};
    B4 = Ut_perp{4} * Ut_perp{4}' * double( tenmat(ttm(ZU12,{Ut{3}'},3 ), 4 ) ) * Vt{4};
    
    V1_tensor = tensor(Vt{1}', [r1_use,r2_use,r3_use,r4_use]);
    V2_tensor = permute( tensor( Vt{2}', [r2_use, r1_use, r3_use, r4_use] ), [2,1,3,4]  );
    V3_tensor = permute( tensor( Vt{3}', [r3_use, r1_use, r2_use, r4_use] ), [2,3,1,4]  );
    V4_tensor = permute( tensor( Vt{4}', [r4_use, r1_use, r2_use, r3_use] ), [2,3,4,1]  );
    grad_arm1 = ttm( V1_tensor, { B1, Ut{2}, Ut{3}, Ut{4} }, [1:4] );
    grad_arm2 = ttm( V2_tensor, { Ut{1}, B2, Ut{3}, Ut{4} }, [1:4] );
    grad_arm3 = ttm( V3_tensor, { Ut{1}, Ut{2}, B3, Ut{4} }, [1:4] );
    grad_arm4 = ttm( V4_tensor, { Ut{1}, Ut{2}, Ut{3}, B4 }, [1:4] );
    
    grad = grad_core + grad_arm1 + grad_arm2 + grad_arm3 + grad_arm4;
    alpha_t = (norm(grad))^2 / ( norm( ttm(grad,A,1) ) )^2;
    tildeXt = Xt - alpha_t * grad;
    Xt = hosvd(tildeXt,norm(tildeXt),'ranks',[r1_use,r2_use,r3_use, r4_use],'sequential',true,'verbosity',0);
    Ut = Xt.u;
    St = Xt.core;
    for i = 1:4
        Ut_perp{i} = null(Ut{i}');
        [Vt{i},~] = qr(double(tenmat(St,i))',0);
    end
    Xt = tensor(Xt);
    Xt_err_new = norm(Xt - X)/norm(X);
    if Xt_err_new > Xt_err || (Xt_err - Xt_err_new)/Xt_err < 1e-5
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