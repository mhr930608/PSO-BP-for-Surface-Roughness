%%  Clearing environmental variables
warning off           
close all            
clear                  
clc                     

%%  Load data
res  = xlsread('Surface_input.xlsx');

num_iterations = 10;
MAE_values = zeros(1, num_iterations);
MAPE_values = zeros(1, num_iterations);
MSE_values = zeros(1, num_iterations);

for k=1:num_iterations
%%  Split data into traning set and test set
% shuffle data
temp = randperm(86);
% 70% Traning set
P_train  =  res(temp(1: 61), 1:4)';
T_train  =  res(temp(1: 61), 5)';
M        =  size(P_train,2);        %No.of traning set
% 30% Test set                        
P_test   =  res(temp(62: end), 1:4)';
T_test   =  res(temp(62: end), 5)';
N        =  size(P_test,2);         %No.of test set

%%  Normalization of data to the range of (0,1)
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%%  Neural numbers in each layer
inputnum  = size(p_train, 1);  % Input layer numbers
hiddennum = 2*inputnum+1;      % Hidden layer numbers (input*2+1)
outputnum = size(t_train, 1);  % Output layer numbers

%%  Set up the BP neural network to be optimized by PSO
net = newff(p_train, t_train, hiddennum,{'logsig','purelin'},'trainlm');

%  Parameters of BP
net.trainParam.epochs     = 1000;      % training epochs
net.trainParam.goal       = 1e-5;      % error goal
net.trainParam.lr         = 0.01;      % learning rate
net.trainParam.showWindow =    0;      % close window

%%  Parameters initialization
c1      =  2.0;       % learning factor
c2      =  2.0;       % learning factor
maxgen  =   50;        % iteration times  
sizepop =   30;        % population size
wmax    =  0.9;        % inertia weight
wmin    =  0.4;
Vmax    =  1.0;        % max velocity
Vmin    = -1.0;        % min velocity
popmax  =  5.0;        % max position
popmin  = -5.0;        % min position

%% Initialization of particle location and velocity  
% Total nodes number 
numsum = inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum;

for i = 1 : sizepop
% randomly intialize the particle
    pop(i, :)  = 5*rands(1, numsum);  % Initialize the position
    V(i, :)    = rands(1, numsum);    % Initialize the velocity
% fitness value calculation
    fitness(i) = fun(pop(i, :), hiddennum, net, p_train, t_train);
end
 
% individual and global optimal
[fitnesszbest, bestindex] = min(fitness);
zbest = pop(bestindex, :);     % global optimal position
gbest = pop;                   % individual optimal position
fitnessgbest = fitness;        % individual optimal fitness value
BestFit = fitnesszbest;        % global optimal fitness value

%%  Iteration for the optimal value
for i = 1: maxgen
    for j = 1: sizepop
        
        % Update the velocity
        w=wmax-(wmax-wmin)*j/maxgen;        
        V(j, :) = w*V(j, :) + c1 * rand * (gbest(j, :) - pop(j, :)) + c2 * rand * (zbest - pop(j, :));
        V(j, (V(j, :) > Vmax)) = Vmax;
        V(j, (V(j, :) < Vmin)) = Vmin;
        
        % Update the position
        pop(j, :) = pop(j, :) +  0.5*V(j, :);
        pop(j, (pop(j, :) > popmax)) = popmax;
        pop(j, (pop(j, :) < popmin)) = popmin;
        
        % Adaptive mutation
        if rand >0.9
            pos = ceil(21*rand);
            pop(j, pos) = rand;    %10 percent chance of
            %randomly change a value 
        end
        
        % fitness value for the new particle
        fitness(j) = fun(pop(j, :), hiddennum, net, p_train, t_train);

    end

    % Update the individual and global optimal
    for j = 1 : sizepop
        % individual
        if fitness(j) < fitnessgbest(j)
            gbest(j, :) = pop(j, :);
            fitnessgbest(j) = fitness(j);
        end

        % global
        if fitness(j) < fitnesszbest
            zbest = pop(j, :);
            fitnesszbest = fitness(j);
        end

    end

    BestFit = [BestFit, fitnesszbest];    
end

%%  Get the optimal initial weight and bias
w1 = zbest(1 : inputnum * hiddennum);
B1 = zbest(inputnum * hiddennum + 1 : inputnum * hiddennum + hiddennum);
w2 = zbest(inputnum * hiddennum + hiddennum + 1 : inputnum * hiddennum ...
    + hiddennum + hiddennum * outputnum);
B2 = zbest(inputnum * hiddennum + hiddennum + hiddennum * outputnum + 1 : ...
    inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum);
%%  Plug the value into NN
net.Iw{1, 1} = reshape(w1, hiddennum, inputnum);
net.Lw{2, 1} = reshape(w2, outputnum, hiddennum);
net.b{1}     = reshape(B1, hiddennum, 1);
net.b{2}     = B2';

%%  Open the traning window 
net.trainParam.showWindow = 1;        

%%  Train the NN
net = train(net, p_train, t_train);

%%  Prediction result using PSO-BP
t_sim1 = sim(net, p_train);    %prediction result for training set
t_sim2 = sim(net, p_test );    %prediction result for test set

%  Data inverse-normalization
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);

%Relative error for PSO-BP
error_train = (T_train- T_sim1)./T_train;
error_test  = (T_test - T_sim2)./T_test; 

%Combine 
T=[T_train, T_test];
T_sim = [T_sim1, T_sim2];
%%  Statistics for PSO-BPNN
%  R2
R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test -  T_sim2)^2 / norm(T_test -  mean(T_test ))^2;
disp(['R2 for the training set£º', num2str(R1)])
disp(['R2 for the test set£º', num2str(R2)])

%  MAE (Mean absolute error) 
mae1 = sum(abs(T_sim1 - T_train), 2)' ./M  ;
mae2 = sum(abs(T_sim2 - T_test ), 2 )' ./N ;
avmae= sum(abs(T_sim - T ), 2 )' ./(M+N) ;
disp(['MAE for the training set£º', num2str(mae1)])
disp(['MAE for the test set£º', num2str(mae2)])
disp(['Total MAE for PSO-BP£º', num2str(avmae)])

%  MAPE (Mean absolute percentage error)
mape1       = sum(abs(error_train), 2)' ./ M;
mape2       = sum(abs(error_test ), 2)' ./ N;
avmape      = sum(abs((T-T_sim)./T), 2)' ./ (M+N);
disp(['MAPE for the training set£º', num2str(mape1)])
disp(['MAPE for the test set£º', num2str(mape2)])
disp(['Total MAPE for PSO-BP:',num2str(avmape)])

%  MSE
rmse1 = sum((T_train - T_sim1).^2, 2)' ./M ;
rmse2 = sum((T_test - T_sim2) .^2, 2)' ./N ;
avmse = sum((T - T_sim) .^2, 2)' ./(M+N) ;
disp(['MSE for the training set£º', num2str(rmse1)])
disp(['MSE for the test set£º', num2str(rmse2)])
disp(['Total MSE for PSO-BP£º', num2str(avmse)])

MAE_values(k) = avmae;
MAPE_values(k) = avmape;
MSE_values(k) = avmse;
end

% Calculating average value
mean_MAE = mean(MAE_values);
mean_MAPE = mean(MAPE_values);
mean_MSE = mean(MSE_values);
disp(['Average MAE: ', num2str(mean_MAE)]);
disp(['Average MAPE: ', num2str(mean_MAPE)]);
disp(['Average MSE: ', num2str(mean_MSE)]);
