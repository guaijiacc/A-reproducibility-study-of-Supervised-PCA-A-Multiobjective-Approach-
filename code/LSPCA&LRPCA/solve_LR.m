function beta = solve_LR(Z, Y)
    % Solve logistic regression
    [n, k] = size(Y);
    beta_init = zeros(size(Z,2), k);
    options = optimset('Display', 'off');
    beta = fminsearch(@(b) lr_objective(reshape(b, size(Z,2), k), Z, Y), beta_init(:), options);
    beta = reshape(beta, size(Z,2), k);
end

function obj = lr_objective(beta, Z, Y)
    % Compute logistic regression objective
    logits = Z*beta;
    max_logits = max(logits, [], 2);
    exp_logits = exp(logits - max_logits); % Subtract max for stability
    M = exp_logits./sum(exp_logits, 2);
    obj = - sum(sum(Y.*log(M)));
end
