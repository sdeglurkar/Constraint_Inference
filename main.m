function main()

planner = AStarPlanner([4, -2, pi/2], [-5, -5, -pi], [5, 5, pi], 1, 1, 0.5);
[xs, ys, thetas, times] = planner.planTraj([-4, 3, pi]);
close all
scatter(xs, ys, 'filled')
axis([-5, 5, -5, 5])

for i=1:length(thetas)
    delta_x = 0.5 * cos(thetas(i));
    delta_y = 0.5 * sin(thetas(i));
    annotation('arrow', [(xs(i) - -5)/10 (xs(i) + delta_x - - 5)/10], [(ys(i) - - 5)/10 (ys(i) + delta_y - - 5)/10]);
end

%[~, cmdout] = system('python stuff.py');
%save(cmdout)
%lala = 3 * 4;

% initial_state = [4; -3; pi/2];
% final_state = [-4; 2; pi/2];
% initial_speed = 0.5;
% final_speed = 0;
% T = 1;
%     
% [traj_x, traj_y, speed, speed_angle] = splinePlanner(initial_state, final_state, initial_speed, final_speed, T);
% disp(speed)
% 
% close all
% visualize = true;
% if visualize
%     figure()
%     scatter(traj_x, traj_y, 'filled')
%     axis([-5, 5, -5, 5])
% end

end