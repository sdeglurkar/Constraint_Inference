function [data, g] = BRSPrecomputation(params)
%% Grid
grid_size = params.grid_size;
grid_disc = params.grid_disc;
grid_min = [-grid_size; -grid_size; -pi]; % Lower corner of computation domain
grid_max = [grid_size; grid_size; pi];    % Upper corner of computation domain
N = [grid_disc; grid_disc; grid_disc];         % Number of grid points per dimension
pdDims = 3;               % 3rd dimension is periodic
g = createGrid(grid_min, grid_max, N, pdDims);
% Use "g = createGrid(grid_min, grid_max, N);" if there are no periodic
% state space dimensions

%% target set
R = params.R;
% data0 = shapeCylinder(grid,ignoreDims,center,radius)
data0 = shapeCylinder(g, 3, [0; 0; 0], R);
%data0 = shapeRectangleByCorners(g, [0; 0; -Inf], [1; 1; +Inf]);
% also try shapeRectangleByCorners, shapeSphere, etc.

if params.displayBRS
    figure(1)
    [g2D, data02D] = proj(g, data0, [0, 0, 1], 0);
    visSetIm(g2D, data02D);
    hold on; 
end
%% time vector
t0 = params.t0;
tMax = params.tMax;
dt = params.dt;
tau = t0:dt:tMax;

%% problem parameters

% input bounds
% speed = 1;
% wMax = 1;
% do dStep1 here

% control trying to min or max value function?
uMode = params.uMode;

%% Pack problem parameters

% Define dynamic system
% obj = DubinsCar(x, wMax, speed, dMax)
dCar = params.dCar; 

% Put grid and dynamic systems into schemeData
schemeData.grid = g;
schemeData.dynSys = dCar;
schemeData.accuracy = 'high'; %set accuracy
schemeData.uMode = uMode;

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

if params.displayBRS
    [g2D, data2D] = proj(g, data, [0, 0, 1], params.state(3));
    visSetIm(g2D, data2D);
end

