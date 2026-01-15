function [Z, beta, L, lambda, gamma] = Alternating_algorithm(X, Y, L, lambda, gamma, option, fix)
% Input: 
% An n × p data matrix X, 
% an n × q response matrix Y , 
% a p × r orthogonal matrix L with columns given by the
% first r principal components of X, 
% lambda: hyperparameter
% gamma: hyperparameter
% option: 1: LSPCA; 2: LRPCA
% fix: 0: update lambda and gamma by MLE; 1: fix initial lambda and gamma

% Output:
% The n × r reduced data matrix Z∗, 
% the coefficients β∗, 
% a p × r orthogonal matrix L∗ such that Z∗ = XL∗
    
    [n, p] = size(X);
    [n, q] = size(Y);
    [p, r] = size(L);
    
    if option == 1
        beta = pinv(X*L)*Y;
    else
        beta = solve_LR(X*L, Y);
        % beta = solve_beta(X*L,Y);
        % beta = logistic_regression_bfgs(X*L, Y);
    end
    
    max_iter = 20;
    tol = 1E-6;
    k = 0;

    while k < max_iter
        k = k +1;
        % fprintf("k = %d\n", k);

        % update nuisance parameter using MLE
        if fix == 0
            [gamma, lambda] = MLE_nuisance(X, Y, L, beta, gamma, option);
        end
        
        % with beta fixed, solve for L
        problem.M = stiefelfactory(p, r);
        % problem.M = grassmannfactory(p, r);
        problem.cost = @(L) Cost(X,Y,L,beta,gamma,lambda, option);
        problem.egrad = @(L) Gradient(X,Y,L,beta,gamma,lambda, option);

        options.tolgradnorm = 1e-6;
        options.minstepsize = 1e-6;
        options.maxiter = 300;
        [L1, ~, ~] = conjugategradient(problem, L, options);

        if norm(L1 - L, 'fro') < tol
            break;
        end
        
        L = L1;
        % with L fixed, solve for beta
        if option == 1
            beta = pinv(X*L)*Y;
        else
            beta = solve_LR(X*L, Y);
            % beta = solve_beta(X*L,Y);
            % beta = logistic_regression_bfgs(X*L, Y);
        end
        
    end
    
    Z = X*L;
end