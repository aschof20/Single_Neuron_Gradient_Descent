% Training a single neuron with Gradient Descent and weight decay
% regularization

% Load dataset
load A3.dat
% Define predictor variables
X = [ones(8,1) A3(:,1:2)];
% Define response variables
t = A3(:,3);

% Initialize weights and bias.
W = [0 0 0];
% Loop T times
T = 50000;
% Set the learning rate.
eta = 0.01;   
% Set weight decay rate.
alpha = 0.1; 
% Activity Rule
y = @(W) sigmf(W*X', [1 0]);

% Learning Rule
for i = 1:T
    % Calculate the objective function with weight decay regularization.
    grad = -(t' - y(W))*X + alpha*W;
    % Update weights 
    W = W - eta*grad;
end

% Plot average of learned function.
figure(1); clf
% Plot data points according to class.
plot(X(1:4,2),X(1:4,3),'bo', 'MarkerFaceColor', 'b'); hold on
plot(X(5:8,2),X(5:8,3),'ro', 'MarkerFaceColor', 'r')
xlim([0 10]); ylim([0 10]); axis square
xlabel('x1'); ylabel('x2')
hold on
learned_y = @(X) sigmf(W*X',[1 0]);
x1 = linspace(0,10);
x2 = x1;
[x1 x2] = meshgrid(x1, x2);
learned_y_cont = reshape(learned_y([ones(10000,1) x1(:), x2(:)]), 100, 100);
contour(x1, x2, learned_y_cont, [0.12 0.27 0.73 0.88],'--k'); hold on
contour(x1, x2, learned_y_cont, [0.5 0.5], "k" ); hold on
hold off