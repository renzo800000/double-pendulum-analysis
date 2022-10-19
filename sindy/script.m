clear all
close all
clc

%Add the SINDy library (see report for library source attribution)
addpath("./sindy_library/utils")

%% Define parameters and prepare data
n = 4; %2D system with velocities

%Load provided data: one big matrix with all trajectories sequentially
data = readmatrix('trajectories.csv');
t = data(:, 1);
x = data(:, 2:5);

% Calculate at which indices the trajectories in the data start and end
% We do this by looking at the time column: a negative time step indicates
% that a new trajectory has started.
t_shift = ones(size(t));
t_shift(2:end) = t(1:end-1);

all_indices = 1:size(t, 1);

traj_start_flags = (t - t_shift) < 0;
traj_starts = all_indices(traj_start_flags);

traj_ends = zeros(size(traj_starts));
traj_ends(1:end-1) = traj_starts(2:end)-1;
traj_ends(end) = all_indices(end);

% Determine a list of indices that will be used to test the performance 
% of the system of ODEs we find. Script will run faster if we use less
% test trajectories, but on the final run it should be all trajectories.
perf_test_traj_indices = 1:size(traj_starts, 2);

% A list of trajectory indices for which to generate a plot of the original
% vs the recreated ODE data.
plot_traj_indices = [1, 2, 3, 4, 5];


%Compute derivatives
velocities = [x(:, 2), x(:, 4)];

dx = zeros(size(x));
dx(:, [1,3]) = velocities;
dx(2:end, [2,4]) = diff(velocities)./diff(t);


% Set the derivative of the angular velocities to 0 at every 
% new trajectory start. Since these are determined numerically, the values
% that are in the matrix now at trajectory starts have no physical meaning.

for i = 1:size(traj_starts, 2)
    traj_start = traj_starts(i);
    dx(traj_start, [2,4]) = [0,0];
end

%% Build basis function library and Theta matrix

% Obtain a list of base functions as symbolic expresssions.
% For more details, see the base_functions_library() function.

disp("Building base functions library...")
[bases_sym] = base_functions_library();

% Now build the Theta matrix by filling in the symbolic base functions with
% the actual x values. 

disp("Building Theta matrix...")

Theta = zeros([size(x, 1), size(bases_sym,2)]);

syms t1 o1 t2 o2;
for i =1:size(Theta, 2)
    func = matlabFunction(bases_sym(i), 'Vars',{t1, o1, t2, o2});
    Theta(:, i) = func(x(:, 1), x(:, 2), x(:, 3), x(:, 4));
end

%% Identify the dynamics of the system

disp("Identifying dynamics...");

% Create a list of possible values for the sparsification knob: lambda
lambdas=[1e-4;5e-4;1e-3;2e-3;3e-3;4e-3;5e-3;6e-3;7e-3;8e-3;9e-3;1e-2;...
    2e-2;3e-2;4e-2;5e-2;6e-2;7e-2;8e-2;9e-2;1e-1;2e-1;3e-1;4e-1;5e-1;...
    6e-1;7e-1;8e-1;9e-1;1;1.5;2;2.5;3;3.5;4;4.5;5;...
    6;7;8;9;10;20;30;40;50;100;200];

% Integration options that will be used to asses how well a certain
% generated system of ODEs performs.
options = odeset('RelTol',1e-10,'AbsTol',1e-10*ones(1,n));

% List to keep track of generated systems of ODEs and their respective
% errors.
odes = sym.zeros(size(lambdas, 2), 4);
errors = zeros(size(lambdas));

%Loop through all lambda values
for i = 1:size(lambdas, 1)
    disp("Trying lambda value " + num2str(i) + "/" + num2str(size(lambdas, 1)) + " ...")
    lambda = lambdas(i);

    % Create a system of ODEs with the current lambda parameter value 
    Xi = sparsifyDynamics(Theta,dx,lambda,n);
    ODE = bases_sym*Xi;

    syms t1 o1 t2 o2;
    ODE_func = matlabFunction(ODE, 'Vars',{t1, o1, t2, o2});

    % Loop through all perf_test_traj_indices and integrate the system of
    % ODEs. Then, compare with the original data and calculate an average
    % MSE error for this system of ODEs. Store that for later usage
    mse_error = 0;

    for test_traj_index=perf_test_traj_indices
        test_start = traj_starts(test_traj_index);
        test_end = traj_ends(test_traj_index);

        test_t = t(test_start:test_end);
        test_x = x(test_start:test_end, :);
        test_x_0 = test_x(1, :);

        [t2, x2]=ode45(@(t,x)ODE_func(x(1), x(2), x(3), x(4))',test_t,test_x_0, options);

        % Note: if the integration fails somewhere (singularity, for
        % example), x2 might be smaller in size than text_x. Account for
        % this possibility in calculation of the error
        mse_error = mse_error + mean((x2-test_x(1:size(x2, 1), :)).^2, 'all');
    end

    mse_error = mse_error / size(perf_test_traj_indices, 1);

    odes(i, :) = ODE;
    errors(i, :) = mse_error;

    disp("Found an ODE with average MSE loss of:");
    disp(mse_error);
    disp("");
end

%% From all generated systems of ODEs, find the one with the lowest error
% Plot info about this 
[min_error_value, min_error_index] = min(errors);

optimal_lambda = lambdas(min_error_index);
optimal_ODE = odes(min_error_index, :);

syms t1 o1 t2 o2;
optimal_ODE_func = matlabFunction(optimal_ODE, 'File', 'optimal_ODE_func', 'Vars',{t1, o1, t2, o2});


disp(" ----- ----- ----- ----- ----- ")
disp("Optimal ODE found!")
disp("The optimal ODE was generated using lamda: " + num2str(optimal_lambda));
disp("The MSE error was: " + num2str(min_error_value));
disp("The optimal ODE function has been written to optimal_ODE_func.m");

disp("Optimal ODE (shortened):")
disp("t1' = " + string(vpa(optimal_ODE(1),3)));
disp("o1' = " + string(vpa(optimal_ODE(2),3)));
disp("t2' = " + string(vpa(optimal_ODE(3),3)));
disp("o2' = " + string(vpa(optimal_ODE(4),3)));


%% Plot generated trajectories of the optimal system vs the original data

for plot_traj_index=plot_traj_indices
    test_start = traj_starts(plot_traj_index);
    test_end = traj_ends(plot_traj_index);
    test_t = t(test_start:test_end);
    test_x = x(test_start:test_end, :);
    test_x_0 = test_x(1, :);

    [t2, x2]=ode45(@(t,x)optimal_ODE_func(x(1), x(2), x(3), x(4))',test_t,test_x_0, options);  % approximate
    
    figure();
    hold on
    plot3(test_x(:, 1), test_x(:, 3), test_t)
    plot3(x2(:, 1), x2(:, 3), t2)
    xlabel('theta_1')
    ylabel('theta_2')
    zlabel('time')
    legend("True", "Identified")
    hold off

end




