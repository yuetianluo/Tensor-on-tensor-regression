% In this code, we study the property of RGD in the rank overspecified
% tensor on vector regression.


clear;
tic;
rng('default');
rng(2018);
p = 30;
r = 3;
r_use = 10;
sig_total = [0, 1e-6,1e-2];
t_max = 10;
succ_tol = 1e-13;
init = 'good';
retra_type = 'hosvd';
ratio_candidate = [2,4];
final_result = [];

for ratio = ratio_candidate
    for sig = sig_total
        p1 = p; p2 = p; p3 = p; p4 = p;
        r1 = r; r2 = r; r3 = r; r4 = r;
        r1_use = r_use; r2_use = r_use; r3_use = r_use; r4_use = r_use;
        S = tensor(randn(r1, r2, r3, r4));
        E1 = randn(p1, r1);
        E2 = randn(p2, r2);
        E3 = randn(p3, r3);
        E4 = randn(p4, r4);
        [U1,~,~] = svds(E1,r1);
        [U2,~,~] = svds(E2,r2);
        [U3,~,~] = svds(E3,r3);
        [U4,~,~] = svds(E4,r4);
        lambda_min = 10000;
        for k = 1:4
            sigma_min = svds( double(tenmat(S,k)), 1, 'smallest' );
            lambda_min = min(lambda_min, sigma_min);
        end
        U = {U1, U2, U3, U4};
        X = ttm(S, U, [1:4]);
        n = floor(p^(2) * ratio/(lambda_min^2));
        A = randn(n, p1);
        eps = sig * randn(n,p2, p3, p4);
        Y = ttm(X,A,1) + eps;
        if strcmp(init,'good') 
             % initialization
           W = ttm(Y,A',1) ;
           init_result = hosvd(W,norm(W),'ranks',[r1_use,r2_use,r3_use, r4_use],'sequential',false,'verbosity',0);      
        end
        Xt = ttm(init_result.core, init_result.u,[1:4]);
        Ut = init_result.u;
        RGN_error = RGN_tensor_vector(A,Y,Xt, Ut,X, U, p1, p2, p3, p4,r1_use,r2_use,r3_use,r4_use, t_max, succ_tol);
        nrow_error = size(RGN_error,1);
        RGN_error = horzcat( RGN_error, repelem( ratio, nrow_error )', repelem( sig, nrow_error )' );
        final_result = vertcat(final_result, RGN_error);
    end
end
toc