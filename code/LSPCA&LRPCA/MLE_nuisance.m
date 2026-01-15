function [gamma,lambda] = MLE_nuisance(X, Y, L, beta, gamma, option)
% input:
% An n×p data matrix X, 
% an n×q response matrix Y,
% a p × r orthogonal matrix L, 
% an r × q coefficient matrix β, 
% a scalar parameter γ
% option: 1: LSPCA; 2: LRPCA

% output:
% A scalar λ, 
% a scalar γ
    
    [n,p] = size(X);
    [n,q] = size(Y);
    [p,r] = size(L);

    if gamma > 0
        var_x = (norm(X, 'fro')^2 - norm(X*L, 'fro')^2)/n/(p-r);
    else
        var_x = norm(X, 'fro')^2/n/p;
    end

    alpha = max(norm(X*L, 'fro')^2/n/r - var_x, 0);
    gamma = 1 - sqrt(var_x/(var_x + alpha));

    if option == 1
        var_y = norm(Y-X*L*beta, 'fro')^2/n/q;
        lambda = var_y/var_x;
    else
        lambda = 0.5/var_x;
    end
end