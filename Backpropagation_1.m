function [J grad] = nnCostFunction(input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON( hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%	X - input features matrix			матрица входных параметров
%	input_layer_size					количество входных параметров
%	y - output vector					выходной вектор
%	num_labels							количество значений выходного вектора
%				например 5 возможных вариантов:
%						[1 0 0 0 0] 	wait
%						[0 1 0 0 0]		buy open
%						[0 0 1 0 0]		sell open
%						[0 0 0 1 0]		sell close
%						[0 0 0 0 1]		buy close
%	m - number of examples				количество примеров
%	2 - number of hidden layers			количество скрытых слоев
%	hidden_layer_size 					количество элементов в
%										скрытом слое

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 								Theta initialization

% initialize each Theta to a random value in [-epsilon,epsilon]
% где еpsilon = 6^0.5/(Lin+Lout)^0.5
% Lin и Lout - are the number of units in the layers adjacent to Theta
init_epsilon1 = 6^0.5/(input_layer_size + hidden_layer_size)^0.5;
init_epsilon2 = 6^0.5/(hidden_layer_size + hidden_layer_size)^0.5;
init_epsilon3 = 6^0.5/(hidden_layer_size + num_labels)^0.5;

% Theta = rand(rows, columns)*(2*init_epsilon) - init_epsilon
Theta1 = rand(hidden_layer_size, (input_layer_size + 1)*(2*init_epsilon1) - init_epsilon1;
Theta2 = rand(hidden_layer_size, (hidden_layer_size + 1)*(2*init_epsilon2) - init_epsilon2;
Theta3 = rand(num_labels, (hidden_layer_size + 1)*(2*init_epsilon3) - init_epsilon3;

% Setup some useful variables
m = size(X, 1);
  
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));

%								FORWARD PROPAGATION



X = [ones(m, 1) X]; 				%	добавление биаса Х0 = 1
size(X)

a2 = zeros(m,hidden_layer_size);	%	обнуление матрицы первого скрытого слоя
a2 = sigmoid(X*Theta1');			%	вычисление первого скрытого слоя
a2 = [ones(m,1) a2];				%	добавление биаса первого скрытого слоя

a3 = zeros(m,hidden_layer_size);
a3 = sigmoid(a2*Theta2');
a3 = [ones(m,1) a3];

a4 = zeros(m,num_labels);
a4 = sigmoid(a3*Theta3');

									%	вычисление Cost function

for i = 1:num_labels
J = J + 1/m*(-y(:,i)'*log(a4(:,i)) - (1-y(:,i)')*log(1-a4(:,i)));
end;
J

									%	regularization

Theta1_cor = Theta1;				%	воод корректированных Theta с нулевым первым для упрощения вычислений
Theta2_cor = Theta2;				%	(нулевое Theta не регулируется)
Theta3_cor = Theta3;
Theta1_cor(:,1) = 0;
Theta2_cor(:,1) = 0;
Theta3_cor(:,1) = 0;
J = J + (lambda/(2*m))*(sum(Theta1_cor(2:end).^2) + sum(Theta2_cor(2:end).^2) + sum(Theta3_cor(2:end).^2))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%								BACKPROPAGATION

d4 = zeros(m,num_labels);			%	создание вектора ощибки выходного слоя
d4 = a4 - y;						%	вычисление вектора ошибки выходного слоя
size(d4)							%	проверка размерности вектора ошибки
size(Theta3)						%	проверка размерности вектора Theta3 для правильного перемножения в 
									%	формуле вычисления вектора ошибки предыдущего слоя

d3 = (d4*Theta3).*(a3.*(1-a3));		%	вычисление вектора ошибки 3-го слоя
size(d3)
d3 = d3(:,2:end);					%	удаление составляющей ошибки биаса - d3[m,hidden_layer_size]
size(d3)							%	проверка размерности вектора ошибки
size(Theta2)						%	проверка размерности вектора Theta2 для правильного перемножения в
									%	формуле вычисления вектора ошибки предыдущего слоя							

d2 = (d3*Theta2).*(a2.*(1-a2));		%	вычисление вектора ошибки 2-го слоя
size(d2)
d2 = d2(:,2:end);					%	удаление составляющей ошибки биаса - d3[m,hidden_layer_size]
size(d2)							%	проверка размерности вектора ошибки					

									% 	ВЫЧИСЛЕНИЕ

Delta3 = zeros(size(Theta3));		%	задание размерности Дельта3
Delta3 = Delta3 + d4'*a3;			%	вычисление Дельта3
size(Delta3)						%	проверка размерности Дельта3

Delta2 = zeros(size(Theta2));		%	задание размерности Дельта2
Delta2 = Delta2 + d3'*a2;			%	вычисление Дельта2
size(Delta2)						%	проверка размерности Дельта2

Delta1 = zeros(size(Theta1));
Delta1 = Delta1 + d2'*X;			%	a1 = X входная матрица
size(Delta1)

									% 	ВЫЧИСЛЕНИЕ ГРАДИЕНТА и его regularization
									%	коррекционные Тетта без биаса вычеслены ранее
									%	( Theta[0,:] не регулирутся )

Theta1_grad = Delta1./m + lambda/m*Theta1_cor;	%	вычисление градиента Theta1_grad
Theta2_grad = Delta2./m + lambda/m*Theta2_cor;	%	вычисление градиента Theta2_grad
Theta3_grad = Delta3./m + lambda/m*Theta3_cor;	%	вычисление градиента Theta3_grad

% Gradient cheking to compare dJ/dTheta computed using backpropagation vs. using numerical estimate of gradient of J(theta)
% then disable checking code

% Use gradient descent or advansed optimization methd with backpropagation to try to minimize J(Theta) as a function of parameters Theta














end
