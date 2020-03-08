function [value, q_value, optimal_policies] = valueIteration(final_value, discount)
    %% Value iteration for an MDP with deterministic dynamics

    policies = ["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest", "exit"];
    grid_size = size(final_value);
    q_value = zeros([grid_size, length(policies)]);
    optimal_policies = zeros(grid_size);
    old_value = final_value;

    cont = true;
    while cont
        new_value = old_value(:, :);
        for i = 1:grid_size(1)
            for j = 1:grid_size(2)
                for k = 1:length(policies)
                    % Checks for goal/obstacle states
                    if k == 9
                        if final_value(i, j) ~= 0
                            r = old_value(i, j);
                            q_value(i, j, k) = r;
                            continue
                        end
                    else
                        if final_value(i, j) ~= 0
                            q_value(i, j, k) = NaN;
                            continue
                        end
                    end


                    x = [i, j];
                    u = policies(k);
                    x_prime = dynamics(x, u, grid_size);
                    if x_prime == 0
                        q_value(i, j, k) = NaN;
                        continue
                    end
                    r = reward(x, u, x_prime);
                    q_value(i, j, k) = r + discount * old_value(x_prime(1), x_prime(2));
                end
            end
        end

        % Compute value from q-value
        for i = 1:grid_size(1)
            for j = 1:grid_size(2)
                max_q_value = -Inf;
                for k = 1:length(policies)
                    if ~isnan(q_value(i, j, k)) && q_value(i, j, k) > max_q_value
                        max_q_value = q_value(i, j, k);
                        optimal_policies(i, j) = k;
                    end
                end
                new_value(i, j) = max_q_value;
            end
        end

        % Check convergence
        nrm = norm(new_value - old_value);
        if nrm > 0.0001
            cont = true;
            old_value = new_value;
        else
            cont = false;
        end

    end %end while

    value = new_value;


    end



    function result = reward(x, u, x_prime)

    result = 0;

    end


    function x_prime = dynamics(x, u, grid_size)

    x_prime = [0, 0];
    if strcmp(u, 'north')
        x_prime = [x(1) - 1, x(2)];
    elseif strcmp(u, 'south')
        x_prime = [x(1) + 1, x(2)];
    elseif strcmp(u, 'east')
        x_prime = [x(1), x(2) + 1];
    elseif strcmp(u, 'west')
        x_prime = [x(1), x(2) - 1];
    elseif strcmp(u, 'northeast')
        x_prime = [x(1) - 1, x(2) + 1];
    elseif strcmp(u, 'northwest')
        x_prime = [x(1) - 1, x(2) - 1];
    elseif strcmp(u, 'southeast')
        x_prime = [x(1) + 1, x(2) + 1];
    elseif strcmp(u, 'southwest')
        x_prime = [x(1) + 1, x(2) - 1];
    elseif strcmp(u, 'exit')
        x_prime = 0;
    end

    if x_prime(1) <= 0 || x_prime(2) <= 0 || x_prime(1) > grid_size(1) || x_prime(2) > grid_size(2)
        x_prime = 0;
    end

end