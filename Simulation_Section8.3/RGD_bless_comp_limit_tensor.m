% In this code, we study the property of RGD in the rank overspecified
% scalar on tensor regression to demonstrate the blessing of computational limits in the overparameterized tensor regression problems.

clear;
server = 1;
  
core_num = 2;
parpool('local', core_num)
exp_time = 100;
n_candidate = 800:100:3500;
n_min = min(n_candidate);
n_max = max(n_candidate); 
r_use_cand = [1,2,3,4,5,6,7,8];

tic;
rng('default');
rng(2018);
p = 90;
r = 1;
sig = 0;
t_max = 500;
tol = 1e-2;

succ_tol = 1e-2;
init = 'good';
retra_type = 'hosvd';

final_result = [];
my_ndims = @(x)(isvector(x) + ~isvector(x) * ndims(x));
outprod = @(u, v)bsxfun(@times, u, permute(v, circshift(1:(my_ndims(u) + my_ndims(v)), [0, my_ndims(u)])));
n_star = 0;
for r_use = r_use_cand
    for n = n_candidate
        if n < n_star
            continue
        end
        round_result = [];
        parfor i = 1:exp_time
            p1 = p; p2 = p; p3 = p;
            r1 = r; r2 = r; r3 = r;
            r1_use = r_use; r2_use = r_use; r3_use = r_use;
            if r == 1
                S = randn(1);
                E1 = randn(p1, r1);
                E2 = randn(p2, r2);
                E3 = randn(p3, r3);
                [U1,~,~] = svds(E1,r1);
                [U2,~,~] = svds(E2,r2);
                [U3,~,~] = svds(E3,r3);
                U = {U1, U2, U3};
                X = outprod(outprod(U1,U2),U3);
                X = tensor(X);
            else               
                S = tensor(randn(r1, r2, r3));
                E1 = randn(p1, r1);
                E2 = randn(p2, r2);
                E3 = randn(p3, r3);
                [U1,~,~] = svds(E1,r1);
                [U2,~,~] = svds(E2,r2);
                [U3,~,~] = svds(E3,r3);  
                U = {U1, U2, U3};
                X = ttm(S, U, [1:3]);                
            end
            A = tensor(randn(p1, p2, p3,n));
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
            RGD_error = RGD_scalar_tensor(A_mat,y,Xt, Ut,X, U, p1, p2, p3,r1_use,r2_use,r3_use, t_max, tol );
            nrow_error = size(RGD_error,1);
            round_result = vertcat( round_result, RGD_error(nrow_error,:) );
        end       
        per_result_mean = median(round_result,1);
        per_result_sd = std(round_result,1);
        disp("inner loop result")
        per_result = [n, r_use, per_result_mean, per_result_sd]
        final_result = vertcat(final_result, per_result); 
        if per_result_mean(2) < succ_tol
            n_star = n;
            break
        end
    end
end

toc;

