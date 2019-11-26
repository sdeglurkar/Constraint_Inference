function data_newobs = computeNewL(correction, state)
%% Problem parameters
grid_size = 5;
grid_disc = 100;
R = 0.5;
displayBRS = false;
t0 = 0;
dt = 0.05;
tMax = 2;
uMode = 'max';
speed = 1;
wMax = 1;
dCar = DubinsCar([0, 0, 0], wMax, speed);
%state = [0.1; -0.99; pi/2];
%state = [0.5; -0.2; pi/2];
%state = [-2; 1; pi/2];

params.grid_size = grid_size;
params.grid_disc = grid_disc;
params.R = R;
params.displayBRS = displayBRS;
params.t0 = t0;
params.dt = dt;
params.tMax = tMax;
params.uMode = uMode;
params.dCar = dCar;
params.state = state;

%% Get the precomputed obstacle avoid set
[data, g] = BRSPrecomputation(params);

tau = t0:dt:tMax;

%% Find new obstacle center(s)

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
if displayBRS
    scatter(x, y, 8, 'k', 'filled')
    hold on;
end

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
        new_obstacle_center = [state(1) - boundary_state(1); ...
            state(2) - boundary_state(2); state(3)];
        obstacles_x(end + 1) = new_obstacle_center(1);
        obstacles_y(end + 1) = new_obstacle_center(2);
    end
end

scatter(state(1), state(2), 8, 'g', 'filled');
hold on;
viscircles([obstacles_x.' obstacles_y.'], R * ones(1, length(obstacles_x)), 'Color', 'k');
scatter(obstacles_x, obstacles_y, 8, 'b', 'filled');
hold on;

data_newobs = shapeCylinder(g, 3, [obstacles_x(1); obstacles_y(1); 0], R);
for i=2:length(obstacles_x)
    data_newobs = min(data_newobs, shapeCylinder(g, 3, [obstacles_x(i); obstacles_y(i); 0], R));
end
   
[g2D, data_newobs02D] = proj(g, data_newobs, [0, 0, 1], 0);
visSetIm(g2D, data_newobs02D);

end