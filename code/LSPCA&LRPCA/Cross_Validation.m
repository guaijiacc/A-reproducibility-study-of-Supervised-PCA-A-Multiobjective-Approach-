function para = Cross_Validation(X, y, option, r_CV, l_CV);
% this function uses 10-fold cross-validation to minimize prediction error to tune hyperparameter
% input:
% X: normalized training features
% y: normalized training labels
% option: 1: linear regression; 2: logistic regression
% r_CV = 0; % 0: fixed r = 2; 1: choose r by CV
% l_CV = 0; % 0: choose lambda by MLE; 1: choose lambda by CV

% output:
% para = [r, lambda];
    
    if r_CV == 0
        R = 2:2;
    else
        R = 2 : min(10, size(X,2));
    end
    
    if l_CV == 0
        Lambda = [1];
    else
        Lambda = [0.001, 0.01, 0.1, 1, 10];
    end
    
    gamma = [1];

    n1 = size(R, 2);
    n2 = size(Lambda, 2);
    n3 = size(gamma, 2);

    R = reshape(reshape(R, 1,1,n1) + zeros(n3, n2, n1),[],1);
    Lambda = reshape(reshape(Lambda, 1, n2, 1) + zeros(n3, n2, n1),[],1);
    gamma = reshape(reshape(gamma, n3, 1, 1) + zeros(n3, n2, n1),[],1);

    Para = [R,Lambda,gamma];
    
    %% 10-fold cross validation
    num_folds = 10;
    n = size(X,1);
    rand_ind = randperm(n);
    dn = floor(n/num_folds);

    Loss = zeros(1,size(Para,1));
    parfor p = 1 : size(Para,1)
        r = Para(p,1);
        lambda = Para(p,2);
        gamma = Para(p,3);
        loss = zeros(1,num_folds);

        for f = 1:num_folds
            if f < num_folds
                val_ind = rand_ind((f-1)*dn + 1:f*dn);
            else
                val_ind = rand_ind((f-1)*dn + 1:end);
            end
            train_ind = setdiff(rand_ind, val_ind);

            X_train = X(train_ind, :);
            y_train = y(train_ind, :);
            X_val = X(val_ind, :);
            y_val = y(val_ind, :);
            
            [U, ~, ~] = svd(X_train');
            L = U(:,1:r);
            [~, beta, L, lambda, gamma] = Alternating_algorithm(X_train, y_train, L, lambda, gamma, option, l_CV);
            
            if option == 1               
                Z_val = X_val*L;
                y_val_pred = Z_val*beta;

                loss(f) = mean((y_val_pred - y_val).^2,"all");
            else
                logits_val = X_val*L*beta;
                max_logits_val = max(logits_val, [], 2);
                exp_logits = exp(logits_val - max_logits_val); % Subtract max for stability
                
                M = exp_logits./sum(exp_logits, 2);
                loss(f) = - sum(sum(y_val.*log(M)));
            end
        end

        Loss(p) = mean(loss);

        fprintf("para set %d finished\n", p);
    end

    [~, idx] = min(Loss);
    para = Para(idx, :);
end