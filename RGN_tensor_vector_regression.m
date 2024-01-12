% RGN for order-4 tensor-on-vector regression
% Output: iteration, time and error
% Input: A: n-by-p1 covariate matrix; Y: observed tensors; X0: initialization; X: true parameter of interest; U: true loadings of X
% p1-p4: dimensions; r1_use-r4_use: input ranks; iter_max: iteration number; succ_tol: successful stopping criteria.


function [ error_matrix ] = RGN_tensor_vector_regression( A,Y,X0, U0,X, U, p1, p2, p3, p4, r1_use,r2_use,r3_use, r4_use, iter_max, succ_tol)
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
for iter = 1:iter_max
    W1 = kron( kron(Ut{4}, Ut{3}), Ut{2} ) * Vt{1};
    W2 = kron( kron(Ut{4}, Ut{3}), Ut{1} ) * Vt{2};
    W3 = kron( kron(Ut{4}, Ut{2}), Ut{1} ) * Vt{3};
    W4 = kron( kron(Ut{3}, Ut{2}), Ut{1} ) * Vt{4};
    YA1 = ttm(Y, {A', Ut{2}', Ut{3}', Ut{4}'}, [1:4] );
    YA2 = ttm(Y, { (Ut{1}' * A'), Ut{3}', Ut{4}' }, [1,3,4] );
    YA3 = ttm(Y, { (Ut{1}' * A'), Ut{2}', Ut{4}' }, [1,2,4] );
    YA4 = ttm(Y, { (Ut{1}' * A'), Ut{2}', Ut{3}' }, [1:3] );
    
    core_middle = Ut{1}' * double(tenmat(YA1,1)) - Ut{1}' * (A') * A * Ut_perp{1} * Ut_perp{1}' / (A' * A) * double(tenmat(YA1,1)) * Vt{1} * Vt{1}';
    core = (Ut{1}' * (A') * A * Ut{1}) \ core_middle;
    core = tensor( core, [r1_use,r2_use,r3_use,r4_use] );
    core_tensor = ttm(core, Ut, [1:4]);
    
    arm1 = Ut_perp{1} * Ut_perp{1}' / (A' * A) * double(tenmat(YA1,1)) * Vt{1} * (W1');
    arm2 = Ut_perp{2} * Ut_perp{2}' * double(tenmat(YA2,2)) * Vt{2} /( Vt{2}' * kron(eye(r4_use*r3_use), (Ut{1}' * (A') * A * Ut{1}) ) * Vt{2} ) * W2';
    arm3 = Ut_perp{3} * Ut_perp{3}' * double(tenmat(YA3,3)) * Vt{3} /( Vt{3}' * kron(eye(r4_use*r2_use), (Ut{1}' * (A') * A * Ut{1}) ) * Vt{3} ) * W3';
    arm4 = Ut_perp{4} * Ut_perp{4}' * double(tenmat(YA4,4)) * Vt{4} /( Vt{4}' * kron(eye(r3_use*r2_use), (Ut{1}' * (A') * A * Ut{1}) ) * Vt{4} ) * W4';
    arm1_tensor = tensor(arm1, [p1,p2,p3,p4]);
    arm2_tensor = permute(tensor(arm2, [p2,p1,p3,p4]), [2,1,3,4] );
    arm3_tensor = permute(tensor(arm3, [p3,p1,p2,p4]), [2,3,1,4] );
    arm4_tensor = permute(tensor(arm4, [p4,p1,p2,p3]), [2,3,4,1] );
    tildeXt = core_tensor + arm1_tensor + arm2_tensor + arm3_tensor;
    Xt = hosvd(tildeXt,norm(tildeXt),'ranks',[r1_use,r2_use,r3_use, r4_use],'sequential',true,'verbosity',0);
    Ut = Xt.u;
    St = Xt.core;
    for i = 1:4
        Ut_perp{i} = null(Ut{i}');
        [Vt{i},~] = qr(double(tenmat(St,i))',0);
    end
    Xt = tensor(Xt);
    Xt_err = norm(Xt - X)/norm(X);
    time = toc;
    iter_result = [iter, Xt_err, time];
    error_matrix = vertcat(error_matrix, iter_result);
    if Xt_err < succ_tol || Xt_err > 50
            break
    end    
end
end