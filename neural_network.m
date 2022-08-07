%% Simple Neural Network
% Jonas Müller Gonçalves - Engenharia Aeroespacial
% Algoritmos de Otimização
clear all; clc;

%% Dados de Entrada da Rede
a = 0.0000000006;   % Valor de ajuste
% a = 0.000000002;   % Valor de ajuste

% rng(5); % Define o Chute dos valores padrão
tol = 0.01;  % Tolerância de Erro
niter = 1000; % Tolerância de Iterações de Treino

%% Pesos dos Perceptrons

w1 = rand;
w2 = rand;
w3 = rand;
w4 = rand;
w5 = rand;
w6 = rand;


%% Input
f = @(x) 3.*x(1) + x(2)./2 - 4.*x(1) + 1.*x(2)
f1 = @(x,y) 3.*x +y./2 - 4.*x + 1.*y

ninputs = 1000; % Nº de Pontos de Treinamento

for ip = 1:ninputs
    inputs(ip,:) = [randi([0 800]) randi([0 800])];   % Gera inputs randômicos para a entrada
end

for ia = 1:ninputs
    actual(ia,:) = [f(inputs(ia,:))];   % Calcula a saída espera a partir da função de referência
end

scatter3(inputs(:,1),inputs(:,2),actual(:,1),'y*');hold on;
% Auxiliares
prediction = 0; iter=0; ierr = 1;

%% Treino da Rede
disp('--------- Início do Treinamento da Rede ---------');

while ierr == 1
    for j = 1:size(inputs,1)
        %% Forward Pass
        wa = [w1 w3;w2 w4]; % Armazena os pesos em matrizes
        wb = [w5;w6];       % Armazena os pesos em matrizes
        hidden1 = inputs(j,:)*wa; % Calcula a hidden layer 
        h(:,j) = [hidden1'];        % Armazena a hidden layer para cada conjunto de entradas
        outputs = hidden1*wb;               % Retorna os valores da hidden layer
        delta(j,1) = outputs - actual(j,1); % Obtém o valor de delta
        
        %% Erro
        Error_aux(j,1) = 0.5*(delta(j,1))^2;    % Calcula o erro do conjunto de entradas                   
    end
    
    %% Backpropagation
    WB = wb - a.*sum(delta'.*h,2);  % Calcula os novos pesos da hidden para a out 
    WA = wa - a.*wb'.*[sum(delta.*inputs,1)'];  % Calcula os novos pesos da in para hidden
    Error = sum(Error_aux); % Calcula o erro total obtido no processo
    
    w1 = WA(1,1);w2 = WA(2,1);w3 = WA(1,2);w4 = WA(2,2);
    w5 = WB(1,1);w6 = WB(2,1);
    
    %% Contador
    iter=iter+1;
    
    %% Condição de Parada do Treino da Rede
    disp(['Erro associado: ',num2str(Error)]);
    
    if Error < tol || iter > niter   
        if Error < tol 
          disp(['Condição de Parada Atingida: Erro < ',num2str(tol)]);
        elseif iter > niter
          disp(['Condição de Parada Atingida: Iterações > ',num2str(niter)]);            
        end
        disp(['Número de Iterações de Treino: ',num2str(iter)]);
        
        ierr =0;
    end
end

%% Teste de um ponto qualquer
x1 = 124.254;     % Ponto a ser encontrado x
x2 = 201.237;    % Ponto a ser encontrado y
prediction_sol = (x1*w1 + x2*w2)*w5 + (x1*w3+x2*w4)*w6; % Resultado da Rede a partir dos w's obtidos do treino
fsurf(f1); hold on; 
scatter3(x1,x2,f1(x1,x2),'ro','linewidth',1.4); hold on;    % Plot do resultado real
scatter3(x1,x2,prediction_sol,'b*','linewidth',1.8);        % Plot do resultado da rede
legend('Ponto de Treino','Solução de Referência','Ponto Exato', 'Ponto Estimado');

disp(' ');
disp('--------- Teste da Rede Neural ---------');
disp(['Resultado Exato da Função: ',num2str(f1(x1,x2))]);
disp(['Resultado da Rede Neural: ',num2str(prediction_sol)]);
disp(['Erro da Rede: ',num2str(Error)]);

