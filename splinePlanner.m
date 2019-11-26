function [traj_x, traj_y, speed, speed_angle] = splinePlanner(initial_state, final_state, initial_speed, final_speed, T)
%% This code computes and plot a third-order spline for a Dubins car from a starting state, 
%  non-zero speed to a goal state, non-zero speed. 

% Times at which the Spline is plotted
time_samples = 100;
times = linspace(0., T, time_samples);

x0 = initial_state(1);
y0 = initial_state(2);
theta_0 = initial_state(3);
vx0 = initial_speed * cos(theta_0);
vy0 = initial_speed * sin(theta_0);
xf = final_state(1);
yf = final_state(2);
theta_f = final_state(3);
vxf = final_speed * cos(theta_f);
vyf = final_speed * sin(theta_f);

% Coefficients
d1 = x0;
d2 = y0;

c1 = vx0;
c2 = vy0;

a1 = 2 * (x0 - xf) + T * (vx0 + vxf);
a2 = 2 * (y0 - yf) + T * (vy0 + vyf);
b1 = ( xf - T^(3) * (2*(x0 - xf) + T*(vx0 + vxf)) - T*vx0 - x0 ) / T^(2);
b2 = ( yf - T^(3) * (2*(y0 - yf) + T*(vy0 + vyf)) - T*vy0 - y0 ) / T^(2);

% Compute the state trajectory at time steps using the above co-efficients
traj_x = zeros(1, time_samples);
traj_y = zeros(1, time_samples);
xs_dot = zeros(1, time_samples);
ys_dot = zeros(1, time_samples);

for i = 1:time_samples
  t = times(i);
  traj_x(i) = t*t*(a1*t + b1) + c1*t + d1;
  traj_y(i) = t*t*(a2*t + b2) + c2*t + d2;
  xs_dot(i) = 3*t*t*a1 + 2*t*b1 + c1;
  ys_dot(i) = 3*t*t*a2 + 2*t*b2 + c2;
 
speed = sqrt(xs_dot.^(2) + ys_dot.^(2));
speed_angle = atan2(ys_dot, xs_dot);


end