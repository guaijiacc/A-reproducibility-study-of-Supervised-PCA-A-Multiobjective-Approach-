%% This is a unified code to run LSPCA and LRPCA in Ritchie et al, 2022
clear all;
close all;

% make sure manopt is downloaded and saved somewhere in your PC
addpath(genpath('/Users/bbai/Documents/manopt'));

tic;

path = "/Users/bbai/Documents/graduate study/EECS 553/project/data/";

name = ["Residential", "Barshan_A", "Music", "Ionosphere", "Colon", "Arcene"]; %dataset name
opt = [1,1,1,2,2,2]; % 1: for LSPCA 2: for LRPCA

%% change these parameter as needed
d = 4; % choose dataset
r_CV = 0; % 0: fixed r = 2; 1: choose r by CV
l_CV = 0; % 0: choose lambda by MLE; 1: choose lambda by CV
default = 0; % 0: do not use default r, lambda, gamma; 1: use default r, lambda, gamma
% note: when default = 1, the program will not be afftected by r_CV and l_CV

%% if setting default to be 1,change these default values as you want
if default == 1
    r0 = 2;
    lambda0 = 0.1;
    gamma0 = 1;
end

option = opt(d); % 1 for linear regression; 2 for logistic regression
dataset_start = 1;
dataset_end = 10;
Res = zeros(dataset_end - dataset_start + 1, 6);

% for num = dataset_start:dataset_end
parfor num = dataset_start:dataset_end % use parallel computation
    train_data = load(path + name(d) + "_train" + string(num) + ".mat");
    test_data = load(path + name(d) + "_test" + string(num) + ".mat");
    
    X_train = train_data.X_train;
    y_train = train_data.y_train;
    X_test = test_data.X_test;
    y_test = test_data.y_test;

    [n,p] = size(X_train);
    [n,q] = size(y_train);
    
    mu_x = mean(X_train, 1);
    std_x = std(X_train, 0, 1) + 1E-10;
    X_train = (X_train - mu_x)./std_x;
    X_test = (X_test - mu_x)./std_x;
    
    % C = 1;
    % X_train = (X_train - mu_x)./std_x/C;
    % X_test = (X_test - mu_x)./std_x/C;
    
    %% PCA
    if option == 1
        mu_y = mean(y_train, 1);
        std_y = std(y_train, 0, 1);
        y_train = (y_train - mu_y)./std_y;
        y_test = (y_test - mu_y)./std_y;
    else
        label = unique(y_train)';
        y_train = y_train == label;
        y_test = y_test == label;
    end
    

    if default == 1
        r = r0;
        lambda = lambda0;
        gamma = gamma0;

        % initialize L
        [U, S, V] = svd(X_train');
        L = U(:,1:r);

        %% run superviased PCA on train dataset
        [Z_train, beta, L, lambda, gamma] = Alternating_algorithm(X_train, y_train, L, lambda, gamma, option, 1);
    else
        if r_CV == 0 && l_CV == 0
            r = 2;  % fix r = 2
            lambda = 1; % initialize lambda
            gamma = 1; % initialize gamma
        else
            %% cross validation
            para = Cross_Validation(X_train, y_train, option, r_CV, l_CV);
            r = para(1);
            lambda = para(2);
            gamma = para(3);
        end
    
        % initialize L
        [U, S, V] = svd(X_train');
        L = U(:,1:r);
       
        %% run superviased PCA on train dataset
        [Z_train, beta, L, lambda, gamma] = Alternating_algorithm(X_train, y_train, L, lambda, gamma, option, l_CV);
    end
    
    if option == 1    
        % prediction
        y_train_pred = Z_train*beta;
        err_train = mean((y_train_pred - y_train).^2,"all");
    
        Z_test = X_test*L;
        y_test_pred = Z_test*beta;
        
        err_test = mean((y_test_pred - y_test).^2,"all");
        
        % vidualization
        set(gcf,'unit','inches','position',[4,10,5*q,10]); 
        t = tiledlayout(2,q);

        for i = 1:q
            ax_now = nexttile;
            plot(y_train(:,i), y_train_pred(:,i), 'o');

            hold on;
            plot(y_train(:,i), y_train(:,i));
            legend("train label" + string(i), "1:1", 'Location','northwest', 'Fontsize',14);
        end

        for i = 1:q
            ax_now = nexttile;
            plot(y_test(:,i), y_test_pred(:,i), 'o');
            hold on;
            plot(y_test(:,i), y_test(:,i));
            legend("test label" + string(i), "1:1", 'Location','northwest', 'Fontsize',14);
        end

        t.Padding = 'compact';
        t.TileSpacing = 'compact';
    else
        logits_train = X_train*L*beta;
        max_logits = max(logits_train, [], 2);
        y_train_pred = logits_train == max_logits;
        err_train = 1 - mean(diag(Delta(y_train, y_train_pred)));
        
        logits_test = X_test*L*beta;
        max_logits = max(logits_test, [], 2);
        y_test_pred = logits_test == max_logits;
        err_test = 1 - mean(diag(Delta(y_test, y_test_pred)));
    end
    VE = norm(Z_train,'fro')^2/norm(X_train,'fro')^2;
    Res(num, :) = [r, lambda, gamma, err_train, err_test, VE];
    fprintf("dataset %d finished\n", num);
end

toc;