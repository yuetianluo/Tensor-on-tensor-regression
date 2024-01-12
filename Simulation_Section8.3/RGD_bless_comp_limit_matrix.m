% In this code, we study the successful recovery in overparameterized matrix trace regression. 

clear;
core_num = 2;
parpool('local', core_num)
exp_time = 100;
n_candidate = 200:100:3000;
n_min = min(n_candidate);
n_max = max(n_candidate);
r_use_cand = [1,2,3,4,5,6,7,8];

tic;
rng('default');
rng(2018);
p = 100;
r = 1;
retra_type = 'svd';
init = 'good';
succ_tol = 1e-2;
tol = 1e-2;
sig = 0;
iter_max = 500;

final_result = [];
n_star = 0;
for r_use = r_use_cand
    for n = n_candidate
        if n < n_star
            continue
        end
        round_result = [];
        parfor i = 1:exp_time
            p1 = p; p2 = p;
            r1 = r; r2 = r; 
            r1_use = r_use; r2_use = r_use;
            S = randn(r1, r2);
            E1 = randn(p1, r1);
            E2 = randn(p2, r2);
            [U1,~,~] = svds(E1,r1);
            [U2,~,~] = svds(E2,r2);
            A = tensor(randn(n,p1, p2));
            A1 = tenmat(A,1);
            X = U1*S*(U2');

            eps = sig * randn(n,1);
            y = A1.data*(X(:)) + eps;

            if strcmp(init,'good') 
            % initialization
                tildeX = reshape(y'*A1.data,[p1,p2])/n;
                [U0,Sigma0,V0] = svds(tildeX, r_use);
                X0 = U0 * Sigma0 * V0';
            end
            RGD_error = RGD_scalar_matrix(A1.data, y, r_use, p1, p2, U0, V0, Sigma0, X, iter_max,tol);
            nrow_error = size(RGD_error,1);
            round_result = vertcat( round_result, RGD_error(nrow_error,:) );
        end       
        per_result_mean = median(round_result,1);
        per_result_sd = std(round_result,1);
        disp("inner loop result");
        per_result = [n, r_use, per_result_mean, per_result_sd]
        final_result = vertcat(final_result, per_result); 
        if per_result_mean(2) < succ_tol
            n_star = n;
            break
        end
    end
end
toc
