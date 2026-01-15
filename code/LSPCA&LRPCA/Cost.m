function cost = Cost(X,Y,L,beta,gamma,lambda, option)
% Input: 
% An n × p data matrix X, 
% an n × q response matrix Y , 
% a p × r orthogonal matrix L with columns given by the
% first r principal components of X, 
% option: 1: LSPCA; 2: LRPCA


% Output:
% cost

    % calculate loss, omitting consant term
    if option == 1
        loss = -2*trace(beta*Y'*X*L) + norm(X*L*beta, 'fro')^2;
    else
        logits = X*L*beta;
        max_logits = max(logits, [], 2);
        exp_logits = exp(logits - max_logits); % Subtract max for stability
        M = exp_logits./sum(exp_logits, 2);
    
        % compute cross-entropy loss
        loss = - sum(sum(Y.*log(M)));
    end
    
    % compute regularization term, omitting constant term
    regularization = - lambda*gamma*(2-gamma)*norm(X*L,'fro')^2;

    % compute cost
    cost = loss + regularization;
end