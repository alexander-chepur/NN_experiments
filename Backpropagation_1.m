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

%	X - input features matrix			������� ������� ����������
%	input_layer_size					���������� ������� ����������
%	y - output vector					�������� ������
%	num_labels							���������� �������� ��������� �������
%				�������� 5 ��������� ���������:
%						[1 0 0 0 0] 	wait
%						[0 1 0 0 0]		buy open
%						[0 0 1 0 0]		sell open
%						[0 0 0 1 0]		sell close
%						[0 0 0 0 1]		buy close
%	m - number of examples				���������� ��������
%	2 - number of hidden layers			���������� ������� �����
%	hidden_layer_size 					���������� ��������� �
%										������� ����

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 								Theta initialization

% initialize each Theta to a random value in [-epsilon,epsilon]
% ��� �psilon = 6^0.5/(Lin+Lout)^0.5
% Lin � Lout - are the number of units in the layers adjacent to Theta
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



X = [ones(m, 1) X]; 				%	���������� ����� �0 = 1
size(X)

a2 = zeros(m,hidden_layer_size);	%	��������� ������� ������� �������� ����
a2 = sigmoid(X*Theta1');			%	���������� ������� �������� ����
a2 = [ones(m,1) a2];				%	���������� ����� ������� �������� ����

a3 = zeros(m,hidden_layer_size);
a3 = sigmoid(a2*Theta2');
a3 = [ones(m,1) a3];

a4 = zeros(m,num_labels);
a4 = sigmoid(a3*Theta3');

									%	���������� Cost function

for i = 1:num_labels
J = J + 1/m*(-y(:,i)'*log(a4(:,i)) - (1-y(:,i)')*log(1-a4(:,i)));
end;
J

									%	regularization

Theta1_cor = Theta1;				%	���� ���������������� Theta � ������� ������ ��� ��������� ����������
Theta2_cor = Theta2;				%	(������� Theta �� ������������)
Theta3_cor = Theta3;
Theta1_cor(:,1) = 0;
Theta2_cor(:,1) = 0;
Theta3_cor(:,1) = 0;
J = J + (lambda/(2*m))*(sum(Theta1_cor(2:end).^2) + sum(Theta2_cor(2:end).^2) + sum(Theta3_cor(2:end).^2))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%								BACKPROPAGATION

d4 = zeros(m,num_labels);			%	�������� ������� ������ ��������� ����
d4 = a4 - y;						%	���������� ������� ������ ��������� ����
size(d4)							%	�������� ����������� ������� ������
size(Theta3)						%	�������� ����������� ������� Theta3 ��� ����������� ������������ � 
									%	������� ���������� ������� ������ ����������� ����

d3 = (d4*Theta3).*(a3.*(1-a3));		%	���������� ������� ������ 3-�� ����
size(d3)
d3 = d3(:,2:end);					%	�������� ������������ ������ ����� - d3[m,hidden_layer_size]
size(d3)							%	�������� ����������� ������� ������
size(Theta2)						%	�������� ����������� ������� Theta2 ��� ����������� ������������ �
									%	������� ���������� ������� ������ ����������� ����							

d2 = (d3*Theta2).*(a2.*(1-a2));		%	���������� ������� ������ 2-�� ����
size(d2)
d2 = d2(:,2:end);					%	�������� ������������ ������ ����� - d3[m,hidden_layer_size]
size(d2)							%	�������� ����������� ������� ������					

									% 	����������

Delta3 = zeros(size(Theta3));		%	������� ����������� ������3
Delta3 = Delta3 + d4'*a3;			%	���������� ������3
size(Delta3)						%	�������� ����������� ������3

Delta2 = zeros(size(Theta2));		%	������� ����������� ������2
Delta2 = Delta2 + d3'*a2;			%	���������� ������2
size(Delta2)						%	�������� ����������� ������2

Delta1 = zeros(size(Theta1));
Delta1 = Delta1 + d2'*X;			%	a1 = X ������� �������
size(Delta1)

									% 	���������� ��������� � ��� regularization
									%	������������� ����� ��� ����� ��������� �����
									%	( Theta[0,:] �� ����������� )

Theta1_grad = Delta1./m + lambda/m*Theta1_cor;	%	���������� ��������� Theta1_grad
Theta2_grad = Delta2./m + lambda/m*Theta2_cor;	%	���������� ��������� Theta2_grad
Theta3_grad = Delta3./m + lambda/m*Theta3_cor;	%	���������� ��������� Theta3_grad

% Gradient cheking to compare dJ/dTheta computed using backpropagation vs. using numerical estimate of gradient of J(theta)
% then disable checking code

% Use gradient descent or advansed optimization methd with backpropagation to try to minimize J(Theta) as a function of parameters Theta














end
