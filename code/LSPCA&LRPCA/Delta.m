function K = Delta(x,y)
% Delta kernal function
% input: 
% x of size(m,q);
% y of size (n,q);
% output: 
% delta kernal matrix of size (m,n)
    
    [m,q] = size(x);
    [n,~] = size(y);
    D = reshape(x,m,1,q) - reshape(y,1,n,q);
    K = (sum(D.^2, 3) == 0)*1;
end