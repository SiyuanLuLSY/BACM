function [YO,FP,FN,NumberofTestingData,TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy,TY]=ELM_BA(train_data,test_data, Elm_Type,NumberofHiddenNeurons);
% ============================================================   % 
% Files of the Matlab programs included in the book:             %
% Xin-She Yang, Nature-Inspired Metaheuristic Algorithms,        %
% Second Edition, Luniver Press, (2010).   www.luniver.com       %
% ============================================================   %    

% ------------------------------------------------------------   %
% Bat-inspired algorithm for continuous optimization (demo)      %
% Programmed by Xin-She Yang @Cambridge University 2010          %
% For details, please see the following papers:
% 1) Xin-She Yang, Bat algorithm for multi-objective optimization, 
% Int. J. Bio-Inspired Computation, Vol.3, No.5, 267-274 (2011).
% 2) Xin-She Yang, Xingshi He, Bat Algorithm: Literature Review
% and Applications, Int. J. Bio-Inspired Computation,
% Vol. 5, No. 4, pp. 141-149 (2013).
% ------------------------------------------------------------   %
% The main part of the Bat Algorithm                       % 
% Usage: bat_algorithm([20 0.25 0.5]);                     %
% Default parameters
% para=20;  
n=20;      % Population size, typically 10 to 25
A=1.6;      % Loudness  (constant or decreasing)
r0=0.0001;      % Pulse rate (constant or decreasing)
% This frequency range determines the scalings
Qmin=0;         % Frequency minimum
Qmax=2;         % Frequency maximum

REGRESSION=0;
CLASSIFIER=1;
Gain = 1;                                           %  Gain parameter for sigmoid

%%%%%%%%%%% Load training dataset
% load('Train.mat');
T=train_data(:,1)';
P=train_data(:,2:size(train_data,2))';
% clear train_data;                                   %   Release raw training data array

%%%%%%%%%%% Load testing dataset
% load('Test.mat');
TVT=test_data(:,1)';
TVP=test_data(:,2:size(test_data,2))';
% clear test_data;                                    %   Release raw testing data array

NumberofTrainingData=size(P,2);
NumberofTestingData=size(TVP,2);
NumberofInputNeurons=size(P,1);
NumberofValidationData = 50;%round(NumberofTestingData / 2);

if Elm_Type~=REGRESSION
    %%%%%%%%%%%% Preprocessing the data of classification
    sorted_target=sort(cat(2,T,TVT),2);
    label=zeros(1,1);                               %   Find and save in 'label' class label from training and testing data sets
    label(1,1)=sorted_target(1,1);
    j=1;
    for i = 2:(NumberofTrainingData+NumberofTestingData)
        if sorted_target(1,i) ~= label(1,j)
            j=j+1;
            label(1,j) = sorted_target(1,i);
        end
    end
    number_class=j;
    NumberofOutputNeurons=number_class;
    
    %%%%%%%%%% Processing the targets of training
    temp_T=zeros(NumberofOutputNeurons, NumberofTrainingData);
    for i = 1:NumberofTrainingData
        for j = 1:number_class
            if label(1,j) == T(1,i)
                break; 
            end
        end
        temp_T(j,i)=1;
    end
    T=temp_T*2-1;

    %%%%%%%%%% Processing the targets of testing
    temp_TV_T=zeros(NumberofOutputNeurons, NumberofTestingData);
    for i = 1:NumberofTestingData
        for j = 1:number_class
            if label(1,j) == TVT(1,i)
                break; 
            end
        end
        temp_TV_T(j,i)=1;
    end
    TVT=temp_TV_T*2-1;
end                                                 %   end if of Elm_Type
clear temp_T;
clear temp_T;
VV.P = P(:,1:NumberofValidationData);
VV.T = T(:,1:NumberofValidationData);
P(:,1:NumberofValidationData)=[];
T(:,1:NumberofValidationData)=[];
NumberofTrainingData = NumberofTrainingData - NumberofValidationData;

% Iteration parameters
r = r0;
%% In order to obtain better/more accurate results, N_iter
%% should be increased to N_iter=2000 or more if necessary.
N_iter= 20;       % Total number of function evaluations
% Dimension of the search variables
D=NumberofHiddenNeurons*(NumberofInputNeurons+1);
alpha = 0.9;
gamma = 0.99;
XVmin=-ones(1,D);
XVmax=ones(1,D);
% Initial arrays
Q=zeros(n,1);   % Frequency
v=zeros(n,D);   % Velocities
% Initialize the population/solutions
start_time_validation=cputime;
OutputWeight = cell(1,n);

for i=1:n,
  Sol(i,:)=XVmin + (XVmax - XVmin).*rand(1,D);

  [Fitness(i),OutputWeight{i}] = ELM_X(Elm_Type,Sol(i,:),P,T,VV,NumberofHiddenNeurons);
end
[fmin,I]=min(Fitness);
bestweight = OutputWeight{I};
best = Sol(I,:);
% ======================================================  %
% Note: As this is a demo, here we did not implement the  %
% reduction of loudness and increase of emission rates.   %
% Interested readers can do some parametric studies       %
% and also implementation various changes of A and r etc  %
% ======================================================  %

% Start the iterations -- Bat Algorithm
for i_ter=1:N_iter,
    % Loop over all bats/solutions
    for i=1:n,
      Q(i)=Qmin+(Qmin-Qmax)*rand;
      v(i,:)=v(i,:)+(Sol(i,:)-best)*Q(i);
%       S(i,:)=Sol(i,:)+v(i,:);
   S(i,:)=Sol(i,:) + 0.3*3.*Sol(i,:).*(1-Sol(i,:).*Sol(i,:));
%   S(i,:)= Sol(i,:) + 0.3*1./Sol(i,:) - round(1./Sol(i,:)) ;
%   Sol(i,:)= Sol(i,:) + 0.3*4.*Sol(i,:).* (1 - Sol(i,:)) ;
      % Apply simple bounds/limits
      Sol(i,:)=simplebounds(Sol(i,:),XVmin,XVmax);
      % Pulse rate
      if rand > r
          S(i,:)=best+0.01*randn(1,D);
      end
      % Evaluate new solutions
       [Fnew,OutputWeight] = ELM_X(Elm_Type,S(i,:),P,T,VV,NumberofHiddenNeurons);
       % If the solution improves or not too loudness
       if (Fnew<=Fitness(i)) & (rand < A)
            Sol(i,:)=S(i,:);
            Fitness(i)=Fnew;
            A = alpha * A;
            r = r0*(1-exp(-gamma*i_ter));
       end
      % Update the current best
      if Fnew<=fmin,
            best=S(i,:);
            fmin=Fnew;
            bestweight = OutputWeight;
      end
    end
end
% End of the main bat algorithm and output/display can be added here.
end_time_validation=cputime;
TrainingTime=end_time_validation-start_time_validation

start_time_validation=cputime;
Beta = mean(abs(OutputWeight));
NumberInputNeurons=size(P, 1);
NumberofTrainingData=size(P, 2);
NumberofTestingData=size(TVP, 2);
Gain=1;
temp_weight_bias=reshape(best, NumberofHiddenNeurons, NumberInputNeurons+1);
InputWeight=temp_weight_bias(:, 1:NumberInputNeurons);
BiasofHiddenNeurons=temp_weight_bias(:,NumberInputNeurons+1);
tempH=InputWeight*P;
ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);      %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH=tempH+BiasMatrix;
clear BiasMatrix
H = 1 ./ (1 + exp(-Gain*tempH));
clear tempH;
% OutputWeight=pinv(H') * T';
Y=(H' * bestweight)';
tempH_test=InputWeight*TVP;
ind=ones(1,NumberofTestingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);      %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH_test=tempH_test + BiasMatrix;
H_test = 1 ./ (1 + exp(-Gain*tempH_test));
TY=(H_test' * bestweight)';
end_time_validation=cputime;
TestingTime=end_time_validation-start_time_validation
if Elm_Type == 0
    TrainingAccuracy=sqrt(mse(T - Y))
    TestingAccuracy=sqrt(mse(TVT - TY))            %   Calculate testing accuracy (RMSE) for regression case
end

if Elm_Type == 1
%%%%%%%%%% Calculate training & testing classification accuracy
    MissClassificationRate_Training=0;
    MissClassificationRate_Testing=0;
    FP=0; FN=0;
    for i = 1 : size(T, 2)
        [x, label_index_expected]=max(T(:,i));
        [x, label_index_actual]=max(Y(:,i));
        if label_index_actual~=label_index_expected
            MissClassificationRate_Training=MissClassificationRate_Training+1;
        end
    end
    TrainingAccuracy=1-MissClassificationRate_Training/size(T,2)
    ww=zeros(size(TVT, 2),1);
    for i = 1 : size(TVT, 2)   % calculating testing accuracy, FP, FN
        [x, label_index_expected]=max(TVT(:,i));
        [x, label_index_actual]=max(TY(:,i));
        if label_index_actual~=label_index_expected
            MissClassificationRate_Testing=MissClassificationRate_Testing+1;
            if label_index_expected==2
                FP=FP+1;
                ww(i)=1;
            else
                FN=FN+1;
                ww(i)=2;
            end
        end
    end
    TestingAccuracy=1-MissClassificationRate_Testing/size(TVT,2)
    YO=TY;
end
% Application of simple limits/bounds
function s=simplebounds(s,Lb,Ub)
  % Apply the lower bound vector
  ns_tmp=s;
  I=ns_tmp<Lb;
  ns_tmp(I)=Lb(I);
  
  % Apply the upper bound vector 
  J=ns_tmp>Ub;
  ns_tmp(J)=Ub(J);
  % Update this new move 
  s=ns_tmp;

