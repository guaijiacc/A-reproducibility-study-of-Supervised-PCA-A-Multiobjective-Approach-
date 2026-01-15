function beta = solve_beta(Z, Y)
% This function using gradient descent to solve multiclass logistic regression 

% input
% Z: train data matrix; size (n,r), n is smaple size, r is feature num
% Y: label matrix: size (n, q), n is sample size, q is class num
% each row of Y is a one-hot vector

% output
% beta matrix; size (r,q)

    loop1 = 0;
    max_iter1 = 200;
    max_iter2 = 20;
    [n,r] = size(Z);
    [n,q] = size(Y);
    beta = zeros(r,q);

    while loop1 < max_iter1
        loop1 = loop1 + 1;
        step = 10;
        loss = Loss(Z, Y, beta);
        grad = Grad(Z, Y, beta);

        if norm(grad, 'fro') < 1E-6
            disp("convergence reached");
            return;
        end
        
        loop2 = 0;
        while loop2 < max_iter2
            loop2 = loop2 + 1;
            beta1 = beta - step*grad;
            if Loss(Z, Y, beta1) < loss
                beta = beta1;
                break;
            end
            step = step/2;
        end

        if loop2 == max_iter2
            disp("max loop2 reached");
            return;
        end
      
        beta = beta1;
    end
    
    disp("max loop1 reached");
end

function loss = Loss(Z, Y, beta)
    logits = Z*beta;
    max_logits = max(logits, [], 2);
    exp_logits = exp(logits - max_logits); % Subtract max for stability
    M = exp_logits./sum(exp_logits, 2);

    % compute cross-entropy loss
    loss = - sum(sum(Y.*log(M)));
end

function grad = Grad(Z, Y, beta)
    logits = Z*beta;
    exp_logits = exp(logits);
    M = exp_logits./sum(exp_logits, 2);
    
    % compute gradient
    grad = Z' * ( -Y + M);
end