% Riemannian gradient descent for Matrix trace regression
% And in this code, we output the use time and error after every iteration.
    % Input: A1: matrix form, each row is the vectorization of one covariate matrix, y: response, p1, p2: dimension of the parameter matrix, r: input rank, iter-max: iteration max number,
    % tol:tolerence to terminate
    % hatX:initialization, 
    % X: underlying parameter matrix
function [error_matrix] = RGD_matrix_trace_regression(A1, y, r, p1, p2, hatX, X, iter_max,tol)
    [U0,Sigma0,V0] = svds(hatX,r);
    Ut = U0;
    Ut_perp = null(Ut');
    Vt = V0;
    Vt_perp = null(Vt');
    Sigmat = Sigma0;
    rela_err = norm(hatX - X, 'fro')/norm(X, 'fro');
    error_matrix = [0, rela_err, 0];
    tic;
    for iter = 1:iter_max
        haty = A1*(hatX(:));
        % compute the gradient in Euclidian space
        Z = reshape((haty - y)' * A1, [p1, p2]);
        % Update on the manifold
        grad = Ut * Ut' * Z + Z * Vt * Vt' - Ut * Ut' * Z * Vt * Vt';
        eta = (norm(grad,'fro'))^2/(norm( A1 * (grad(:)), 'fro' ))^2;
        %eta = 0.00001;
        tildeX = hatX - eta * grad;
        [Ut,sigmat,Vt] = svds(tildeX,r);
        hatX = Ut * sigmat * Vt';
        Ut_perp = null(Ut');
        Vt_perp = null(Vt');
        rela_err_new = norm(hatX - X, 'fro')/norm(X,'fro');
        if (rela_err_new > rela_err & iter >= 100)  || (rela_err - rela_err_new)/rela_err < 1e-5
            break
        else
            rela_err = rela_err_new;
        end          
%        rela_err = norm(hatX - X, 'fro')/norm(X, 'fro');
        time = toc;
        iter_result = [iter, rela_err,time];
        error_matrix = vertcat(error_matrix, iter_result);
        if rela_err < tol || rela_err > 5
            break
        end
    end
end
