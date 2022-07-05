% Functia sigmoid = 1/(1+exp(-z)) 
% Outputul g este o valoare intre 0 si1
function g = sigmoid(z)
g = 1.0 ./ (1.0 + exp(-z));