%% this program draws n =100 realization of a r.v. X in R^4, ~ N(0,I4) to
%% generate Barshan A dataset in Barshan et al, 2011
clear all;
close all;

n = 100; % # of samples
p = 4; % # of features
mu = 0; % mean
sigma = 1; % sqrt of variance

X = normrnd(mu, sigma, p, n); 
noise = normrnd(mu,sigma,1,n);

y = X(1,:)./(0.5 + (X(2,:) + 1.5).^2) + (1 + X(2,:)).^2 + 0.5*noise;
% see Eq. (20) in Barshan et al, 2011 or Example 6.2 in Li et al, 2005

% asjust shape to match the definition in Ritchie et al, 2022.
noise = noise';
X = X';
y = y';

% randomly choose 80% of data as train set and 20% as test set
random_index = randperm(n);
X_train = X(1:round(0.8*n),:);
y_train = y(1:round(0.8*n),:);

X_test = X(round(0.8*n)+1:n,:);
y_test = y(round(0.8*n)+1:n,:);

save('Barshan_A.mat', 'X','y');
save('Barshan_A_train.mat', 'X_train','y_train');
save('Barshan_A_test.mat', 'X_test','y_test');