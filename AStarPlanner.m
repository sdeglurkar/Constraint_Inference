classdef AStarPlanner
    properties
        start
        grid_bounds_min
        grid_bounds_max
        grid_reso
        curr_pos
        
        max_vel
        max_omega
        dt
        
    end
    
    methods
        
        function obj = AStarPlanner(start, grid_bounds_min, grid_bounds_max, max_vel, max_omega, dt)
            obj.start = start;
            obj.grid_bounds_min = grid_bounds_min;
            obj.grid_bounds_max = grid_bounds_max;
            obj.grid_reso = 0.5;
            obj.curr_pos = obj.start;
            
            obj.max_vel = max_vel;
            obj.max_omega = max_omega;
            obj.dt = dt;
        end
        
        function [xs, ys, thetas, times] = planTraj(obj, goal)
            xmin = obj.grid_bounds_min(1);
            ymin = obj.grid_bounds_min(2);
            thetamin = obj.grid_bounds_min(3);
            xmax = obj.grid_bounds_max(1);
            ymax = obj.grid_bounds_max(2);
            thetamax = obj.grid_bounds_max(3);
            
            xwidth = xmax - xmin;
            thetawidth = thetamax - thetamin;
            
%             nstart = Node(round( (obj.curr_pos(1) - xmin)/obj.grid_reso ), ...
%                 round( (obj.curr_pos(2) - ymin)/obj.grid_reso ), ...
%                 round( (obj.curr_pos(3) - thetamin)/obj.grid_reso ), 0.0, -1, 0.0);
            nstart = Node(obj.curr_pos(1), obj.curr_pos(2), obj.curr_pos(3), 0.0, -1, 0.0);
            
            open_set = containers.Map();
            closed_set = containers.Map();
            
            open_set(nstart.hash(xmin, ymin, thetamin, xwidth, thetawidth)) = nstart;
            
            gx = goal(1);
            gy = goal(2);
            gtheta = goal(3);
            
            count = 0;
            
            while 1
                count = count + 1;
                
                c_id = -1;
                min_cost = 10000000000;
                keys = open_set.keys();
                for i = 1:length(keys)
                    node = open_set(keys{i});
                    cost = node.cost + obj.calculateHeuristic(gx, gy, gtheta, node.x, node.y, node.theta);
                    if cost < min_cost
                        min_cost = cost;
                        c_id = keys{i};
                    end
                end
                
                current = open_set(c_id);
                
                dist_from_goal = sqrt((current.x - gx)^(2) + (current.y - gy)^(2) + ...
                    (current.theta - gtheta)^(2));

                if dist_from_goal < 0.1 
%                     ngoal = Node(round(gx/self.grid_reso) - xmin, ...
%                         round(gy/self.grid_reso) - ymin, ...
%                         round(gtheta/self.grid_reso) - thetamin, ...
%                         current.cost, current.pind, current.tstamp);
                    ngoal = Node(gx, gy, gtheta, current.cost, current.pind, current.tstamp);
                    break
                end
                
                open_set.remove(c_id);
                closed_set(c_id) = current;
                
                % Expand search grid based on dynamics
                for i = 1:3
                    if i == 1
                        next_theta = current.theta;
                    elseif i == 2
                        next_theta = wrapToPi(current.theta + obj.dt * obj.max_omega);
                    elseif i == 3
                        next_theta = wrapToPi(current.theta - obj.dt * obj.max_omega);
                    end
                    
                    next_x = current.x + obj.dt * obj.max_vel * cos(next_theta);
                    next_y = current.y + obj.dt * obj.max_vel * sin(next_theta);
                    
                    dist = sqrt((next_x - current.x)^(2) + (next_y - current.y)^(2));
                    
                    node = Node(next_x, next_y, next_theta, current.cost + dist, ...
                        c_id, current.tstamp + obj.dt);
                    n_id = node.hash(xmin, ymin, thetamin, xwidth, thetawidth);
                    
                    if ~obj.verifyNode(node)
                        continue
                    end

                    if closed_set.isKey(n_id)
                        continue
                    end
                    
                    % Otherwise if it is already in the open set
                    if open_set.isKey(n_id)
                        if open_set(n_id).cost > node.cost
                            old_node_obj = open_set(n_id);
                            new_node_obj = Node(old_node_obj.x, old_node_obj.y, old_node_obj.theta, ...
                                node.cost, c_id, old_node_obj.tstamp);
                            open_set(n_id) = new_node_obj;
                            %open_set(n_id).cost = node.cost;
                            %open_set(n_id).pind = c_id;
                        end
                    else
                        open_set(n_id) = node;
                    end

           
                
            end   
            
            end
            
            [xs, ys, thetas, times] = obj.calcFinalTraj(ngoal, closed_set);
            xs = flip(xs);
            ys = flip(ys);
            thetas = flip(thetas);
            times = flip(times);
            
            disp("Cost of path: " + ngoal.cost)
        
        end
        
        
        function value = calculateHeuristic(obj, gx, gy, gtheta, x, y, theta)
            weight = 10.0;
            value = weight * sqrt((gx - x)^(2) + (gy - y)^(2));
        end
        
        
        function verified = verifyNode(obj, node)
            verified = true;
            if node.x < obj.grid_bounds_min(1) || node.y < obj.grid_bounds_min(2) ...
                    || node.theta < obj.grid_bounds_min(3)
                verified = false;
                return 
            end
            
            if node.x > obj.grid_bounds_max(1) || node.y > obj.grid_bounds_max(2) ...
                    || node.theta > obj.grid_bounds_max(3)
                verified = false;
                return
            end
            
        end
        
        
        function [xs, ys, thetas, times] = calcFinalTraj(obj, ngoal, closed_set)
            xs = [ngoal.x];
            ys = [ngoal.y];
            thetas = [ngoal.theta];
            times = [ngoal.tstamp];
            
            pind = ngoal.pind;
            
            while pind ~= -1
                n = closed_set(pind);
                xs(end + 1) = n.x;
                ys(end + 1) = n.y;
                thetas(end + 1) = n.theta;
                times(end + 1) = n.tstamp;
                
                pind = n.pind;
            end
            
        end
        
    end
end