%% this program divides six datasets into train and test sets 10 times
clear all;
close all;

k = 2;
dataset = ["Residential", "Barshan_A", "Music", "Ionosphere", "Colon", "Arcene"];
data = load(dataset(k) + ".mat");
name = fieldnames(data);
X = data.(name{1});
y = data.(name{2});

n = size(X,1);
% randomly choose 80% of data as train set and 20% as test set
for i = 1:10
    random_index = randperm(n);
    train_index = random_index(1:round(0.8*n));
    test_index = random_index(round(0.8*n)+1:n);

    X_train = X(train_index,:);
    y_train = y(train_index,:);
    
    X_test = X(test_index,:);
    y_test = y(test_index,:);
    
    % save( dataset(k) + "_train"+ string(i) + ".mat", 'X_train','y_train');
    % save( dataset(k) + "_test"+ string(i) + ".mat", 'X_test','y_test');
end