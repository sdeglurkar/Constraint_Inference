function toy()
% 1. Run Backward Reachable Set (BRS) with a goal
%     uMode = 'min' <-- goal
%     minWith = 'none' <-- Set (not tube)
%     compTraj = false <-- no trajectory
% 2. Run BRS with goal, then optimal trajectory
%     uMode = 'min' <-- goal
%     minWith = 'none' <-- Set (not tube)
%     compTraj = true <-- compute optimal trajectory
% 3. Run Backward Reachable Tube (BRT) with a goal, then optimal trajectory
%     uMode = 'min' <-- goal
%     minWith = 'zero' <-- Tube (not set)
%     compTraj = true <-- compute optimal trajectory
% 4. Add disturbance
%     dStep1: define a dMax (dMax = [.25, .25, 0];)
%     dStep2: define a dMode (opposite of uMode)
%     dStep3: input dMax when creating your DubinsCar
%     dStep4: add dMode to schemeData
% 5. Change to an avoid BRT rather than a goal BRT
%     uMode = 'max' <-- avoid
%     dMode = 'min' <-- opposite of uMode
%     minWith = 'zero' <-- Tube (not set)
%     compTraj = false <-- no trajectory
% 6. Change to a Forward Reachable Tube (FRT)
%     add schemeData.tMode = 'forward'
%     note: now having uMode = 'max' essentially says "see how far I can
%     reach"
% 7. Add obstacles
%     add the following code:
%     obstacles = shapeCylinder(g, 3, [-1.5; 1.5; 0], 0.75);
%     HJIextraArgs.obstacles = obstacles;

%% Grid
grid_size = 5;
grid_disc = 100;
grid_min = [-grid_size; -grid_size; -pi]; % Lower corner of computation domain
grid_max = [grid_size; grid_size; pi];    % Upper corner of computation domain
N = [grid_disc; grid_disc; grid_disc];         % Number of grid points per dimension
pdDims = 3;               % 3rd dimension is periodic
g = createGrid(grid_min, grid_max, N, pdDims);
% Use "g = createGrid(grid_min, grid_max, N);" if there are no periodic
% state space dimensions

%% target set
R = 0.5;
% data0 = shapeCylinder(grid,ignoreDims,center,radius)
data0 = shapeCylinder(g, 3, [0; 0; 0], R);
%data0 = shapeRectangleByCorners(g, [0; 0; -Inf], [1; 1; +Inf]);
% also try shapeRectangleByCorners, shapeSphere, etc.

figure(1)
[g2D, data02D] = proj(g, data0, [0, 0, 1], 0);
visSetIm(g2D, data02D);
hold on; 
%% time vector
t0 = 0;
tMax = 2;
dt = 0.05;
tau = t0:dt:tMax;

%% problem parameters

% input bounds
speed = 1;
wMax = 1;
% do dStep1 here

% control trying to min or max value function?
uMode = 'max';
% do dStep2 here


%% Pack problem parameters

% Define dynamic system
% obj = DubinsCar(x, wMax, speed, dMax)
dCar = DubinsCar([0, 0, 0], wMax, speed); %do dStep3 here

% Put grid and dynamic systems into schemeData
schemeData.grid = g;
schemeData.dynSys = dCar;
schemeData.accuracy = 'high'; %set accuracy
schemeData.uMode = uMode;
%do dStep4 here

%% Compute value function

HJIextraArgs.visualize = false; %show plot
HJIextraArgs.fig_num = 1; %set figure number
HJIextraArgs.deleteLastPlot = true; %delete previous plot as you update

%[data, tau, extraOuts] = ...
% HJIPDE_solve(data0, tau, schemeData, minWith, extraArgs)

if ~isfile('everything.mat')
    [data, tau2, ~] = HJIPDE_solve(data0, tau, schemeData, 'zero', HJIextraArgs);
    save('everything')
else
    load('everything')
end
%% My Stuff

%state = [0.1; -0.99; pi/2];
%state = [0.5; -0.2; pi/2];
state = [-2; 1; pi/2];
scatter(state(1), state(2), 8, 'g', 'filled');
[g2D, data2D] = proj(g, data, [0, 0, 1], state(3));
visSetIm(g2D, data2D);

% Find index of the theta in the state
theta = linspace(-pi, pi, grid_disc);
index = 1;
min_dist = 10000;
for i=1:length(theta)
    if abs(state(3) - theta(i)) < min_dist
        index = i;
        min_dist = abs(state(3) - theta(i));
    end
end

% Get indices of cells in grid corresponding to boundary of zero level set
[grid_x, grid_y] = find(data(:,:,index,end) > -0.019 & data(:,:,index,end) < 0.019);

% Convert to x and y
x = zeros(1, length(grid_x));
y = zeros(1, length(grid_y));
values = zeros(1, length(grid_x));
for i=1:length(grid_x)
    x(i) = -grid_size + (2 * grid_size/grid_disc) * grid_x(i);
    y(i) = -grid_size + (2 * grid_size/grid_disc) * grid_y(i);
    values(i) = data(grid_x(i), grid_y(i), index, end);
end

disp("Values of supposed zero boundary: ")
disp(values)
disp("x = ")
disp(x)
disp("y = ")
disp(y)
scatter(x, y, 8, 'k', 'filled')

correction = -1;

obstacles_x = [];
obstacles_y = [];

Deriv = computeGradients(g, data(:,:,:,end));
for i=1:length(x)
    boundary_state = [x(i); y(i); state(3)];
    deriv = eval_u(g, Deriv, boundary_state);
    u = dCar.optCtrl(tau(end), boundary_state, deriv, uMode);
    if u ~= correction
        continue
    else
        new_obstacle_center = [state(1) - boundary_state(1); state(2) - boundary_state(2); state(3)];
        obstacles_x(end + 1) = new_obstacle_center(1);
        obstacles_y(end + 1) = new_obstacle_center(2);
    end
end

viscircles([obstacles_x.' obstacles_y.'], R * ones(1, length(obstacles_x)), 'Color', 'k');
scatter(obstacles_x, obstacles_y, 8, 'b', 'filled');

%{
deriv = eval_u(g, Deriv, [0.48;0.14;state(3)]);
disp(dCar.optCtrl(tau(end), [0.48;0.14;state(3)], deriv, uMode))
next_state = dCar.updateState(-1, 0.25, [0.5;-0.2;state(3)]);
scatter(next_state(1), next_state(2), 8, 'g', 'filled')
next_state = dCar.updateState(1, 0.25, [0.5;-0.2;state(3)]);
scatter(next_state(1), next_state(2), 8, 'g', 'filled')
%}

%value = eval_u(g,data(:,:,:,end), state);

% Made some changezzz

end