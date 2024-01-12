% In this code, we study the property of RGD in the rank overspecified
% scalar on tensor regression with different rank and sample size input.

clear;
core_num = 2;
parpool('local', core_num)
exp_time = 100;
n_candidate = [500,750,1000,1500,2000,3000,4000,5000,6000,7000,8000];
r_use_cand = [3,6,9,12,15];
tic;
rng('default');
rng(2018);
p = 30;
r = 3;
sig = 1e-6;
t_max = 300;
succ_tol = 1e-13;
init = 'good';
retra_type = 'hosvd';

final_result = [];

for r_use = r_use_cand
    for n = n_candidate
        round_result = [];
        parfor i = 1:exp_time
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
            RGD_error = RGD_scalar_tensor(A_mat,y,Xt, Ut,X, U, p1, p2, p3,r1_use,r2_use,r3_use, t_max, succ_tol);
            nrow_error = size(RGD_error,1);
            round_result = vertcat( round_result, RGD_error(nrow_error,:) );
        end       
        per_result_mean = mean(round_result,1);
        per_result_sd = std(round_result,1);
        disp("inner loop result")
        per_result = [n, r_use, per_result_mean, per_result_sd]
        final_result = vertcat(final_result, per_result); 
    end
end

toc
