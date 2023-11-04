function error = fun(pop, hiddennum, net, p_train, t_train)

%%  Neuron numbers
inputnum  = size(p_train, 1);  % input numbers
outputnum = size(t_train, 1);  % output numbers

%%  Get the optimal weight and bias
w1 = pop(1 : inputnum * hiddennum);
B1 = pop(inputnum * hiddennum + 1 : inputnum * hiddennum + hiddennum);
w2 = pop(inputnum * hiddennum + hiddennum + 1 : ...
    inputnum * hiddennum + hiddennum + hiddennum * outputnum);
B2 = pop(inputnum * hiddennum + hiddennum + hiddennum * outputnum + 1 : ...
    inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum);
 
%%  Plug the value into NN
net.Iw{1, 1} = reshape(w1, hiddennum, inputnum );
net.Lw{2, 1} = reshape(w2, outputnum, hiddennum);
net.b{1}     = reshape(B1, hiddennum, 1);
net.b{2}     = B2';

%%  Train the NN
net = train(net, p_train, t_train);

%%  Prediction result
t_sim1 = sim(net, p_train);

%%  Fitness value
error = sum(sqrt(sum((t_sim1 - t_train) .^ 2) ./ length(t_sim1)));