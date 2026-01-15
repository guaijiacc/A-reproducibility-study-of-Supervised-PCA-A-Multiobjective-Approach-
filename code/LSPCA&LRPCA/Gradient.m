function grad = Gradient(X,Y,L,beta,gamma,lambda, option)
% Input: 
% An n × p data matrix X, 
% an n × q response matrix Y , 
% a p × r orthogonal matrix L with columns given by the
% first r principal components of X, 
% option: 1: LSPCA; 2: LRPCA

% Output:
% the euclidean gradient of cost wrt L; size: p × r

    [n, p] = size(X);
    [n, q] = size(Y);
    [p, r] = size(L);
    
    % caculate grad of loss term
    if option == 1
        loss_grad = -2*X'*(Y-X*L*beta)*beta';
    else
        logits = X*L*beta;
        max_logits = max(logits, [], 2);
        exp_logits = exp(logits - max_logits); % Subtract max for stability
        M = exp_logits./sum(exp_logits, 2);
    
        % loss_grad = zeros(p,r);
        % for i = 1:n
        %     for j = 1:q
        %         loss_grad = loss_grad + (-Y(i,j) + M(i,j))*X(i,:)'*beta(:,j)';
        %     end
        % end
        loss_grad = X'*(-Y + M)*beta';
    end
    
    % calculate grad of regularization term
    regularization_grad = - 2*lambda*gamma*(2-gamma)*X'*X*L;

    grad = loss_grad + regularization_grad;
end