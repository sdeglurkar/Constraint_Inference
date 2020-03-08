classdef Inference
    
    properties
        grid_size
        discount
        robot_state
        beta
        thetas
        theta_prior
        %state_prior
        policies
    end
    
    
    methods
        
        function obj = Inference(grid_size, discount, robot_state)
            obj.grid_size = grid_size;
            obj.discount = discount;
            obj.robot_state = robot_state;
            
            obj.beta = 10;
            obj.thetas = obj.beta * obj.generateThetas();
            obj.theta_prior = ones(1, length(obj.thetas))/length(obj.thetas);  % Uniform prior
%             obj.state_prior = ones(1, obj.grid_size * obj.grid_size)/(obj.grid_size^2);
%             obj.state_prior = reshape(obj.state_prior, [obj.grid_size, obj.grid_size]);
            
            obj.policies = ["north", "south", "east", "west", "northeast", ...
                "northwest", "southeast", "southwest", "exit"];
        end
        
        
        function list_binary_arrays = generateThetas(obj)
            n = obj.grid_size^2;
            max_number_represented = 2^n - 1;
            numbers = 0:max_number_represented;
            list_binary_strings = dec2bin(numbers);
            list_binary_arrays = zeros(obj.grid_size, obj.grid_size, length(numbers));

            for i = 1:length(numbers)
                character_array = reshape(list_binary_strings(i, :), [obj.grid_size, obj.grid_size]);

                for j = 1:obj.grid_size
                    for k = 1:obj.grid_size
                        character_array(j, k) = str2double(character_array(j, k));
                    end
                end

                list_binary_arrays(:, :, i) = character_array;
            end

        end
        
        
        function probability = humanModel(obj, policy_index, theta)
            [value, q_value, optimal_policies] = valueIteration(theta, obj.discount);
            exp_q_vals = zeros(1, length(obj.policies));
            for i = 1:length(obj.policies)
                exp_q_vals(i) = exp(-q_value(obj.robot_state(1), obj.robot_state(2), i));
            end

            sum = 0;
            for i = 1:length(exp_q_vals)
                if ~isnan(exp_q_vals(i))
                    sum = sum + exp_q_vals(i);
                end
            end

            exp_q_vals = exp_q_vals/sum;
            probability = exp_q_vals(policy_index);

        end
        
        
        function probability = stateGivenTheta(obj, theta, state_row, state_col)
            if theta(state_row, state_col) == 0
                probability = 0;
            else
                nonzero_ind = find(theta);
                cardinality = length(nonzero_ind);
                probability = 1/cardinality;
            end
        end
        
        
        function dstb = stateInference(obj, policy_index)
            tic
            dstb = zeros(obj.grid_size, obj.grid_size);
            for i = 1:obj.grid_size
                for j = 1:obj.grid_size
                    for k = 1:length(obj.thetas)
                        theta = obj.thetas(:, :, k);
                        state_row = i;
                        state_col = j;
                        numerator = obj.stateGivenTheta(theta, state_row, state_col) * ...
                            obj.humanModel(policy_index, theta) * obj.theta_prior(k);
                        if ~isnan(numerator)
                            dstb(state_row, state_col) = dstb(state_row, state_col) + numerator;
                        end
                    end
                end
            end
            
            sum_dstb = sum(sum(dstb));
            dstb = dstb/sum_dstb;
            
            toc
            
        end
        
        
        function dstb = sampling(obj, policy_index, num_samples)
            % Likelihood Weighting
            tic
            dstb = zeros(obj.grid_size, obj.grid_size);
            samples_list = zeros(num_samples, 3);
            weights = zeros(1, num_samples);
            for i = 1:num_samples
                theta_index = randi([1, length(obj.thetas)]); %sample from prior(theta) (uniform)
                theta = obj.thetas(:, :, theta_index);
                nonzero_ind = find(theta);
                [row, col] = ind2sub([obj.grid_size, obj.grid_size], nonzero_ind);
                if ~isempty(row)
                    samples_list(i, 1) = theta_index;
                    rand_index = randi([1, length(row)]);
                    samples_list(i, 2) = row(rand_index); %sample from stateGivenTheta (uniform)
                    samples_list(i, 3) = col(rand_index); %sample from stateGivenTheta (uniform)
                    weights(i) = obj.humanModel(policy_index, theta);
                end
                 
            end
            
            for i = 1:obj.grid_size
                for j = 1:obj.grid_size
                    for k = 1:num_samples
                        if samples_list(k, 2) == i && samples_list(k, 3) == j
                            if ~isnan(weights(k))
                                dstb(i, j) = dstb(i, j) + weights(k);
                            end
                        end
                    end
                end
            end
            
            sum_dstb = sum(sum(dstb));
            dstb = dstb/sum_dstb;
            
            toc
            
        end
        
        
%         function probability = state_inference1(obj, policy_index, state_row, state_col)
%             probability = 0;
%             for i = 1:length(obj.thetas)
%                 theta = obj.thetas(:, :, i);
%                 prob = humanModel(obj, policy_index, theta);
%                 if ~isnan(prob)
%                     prob = prob * theta(state_row, state_col)/obj.beta * ...
%                         obj.state_prior(state_row, state_col);
%                     probability = probability + prob;
%                 end
%             end
%         end
%         
%         
%         function grid = visualize_state_dstb(obj, policy_index)
%             grid = zeros(obj.grid_size);
%             for i = 1:obj.grid_size
%                 for j = 1:obj.grid_size
%                     grid(i, j) = obj.state_inference1(policy_index, i, j);
%                 end
%             end
%         end
 

        function probability = theta_inference(obj, policy_index, theta_index)
            prob_all_thetas = zeros(1, length(obj.thetas));
            for i = 1:length(prob_all_thetas)
                prob_all_thetas(i) = obj.humanModel(policy_index, obj.thetas(:, :, i)) * obj.theta_prior(i);
            end
            
            sum = 0;
            for i = 1:length(prob_all_thetas)
                if ~isnan(prob_all_thetas(i))
                    sum = sum + prob_all_thetas(i);
                end
            end
         
            prob_all_thetas = prob_all_thetas/sum;
            probability = prob_all_thetas(theta_index);
        end
        
        
        function dstb_modified = thetaInferenceDistribution(obj, policy_index)
            dstb = zeros(1, length(obj.thetas));
            for i = 1:length(obj.thetas)
                dstb(i) = obj.theta_inference(policy_index, i);
            end
            
            dstb_modified = zeros(1, length(dstb));
            for i = 1:length(dstb)
                if ~isnan(dstb(i))
                    dstb_modified(i) = dstb(i);
                end
            end
            
            plot(1:length(dstb), dstb_modified);
            xlabel('Thetas');
            ylabel('Probability');
            title('P(theta | u_H, x_R)');
            
        end
        
        
%         function grid = visualize(obj, dstb)
%             grid = zeros(obj.grid_size);
%             for i = 1:length(dstb)
%                 theta = obj.thetas(:, :, i);
%                 nonzero_ind = find(theta);
%                 [row, col] = ind2sub([obj.grid_size, obj.grid_size], nonzero_ind);
%                 for j = 1:length(row)
%                     grid(row(j), col(j)) = grid(row(j), col(j)) + dstb(i);
%                 end
%             end
%         end
        
        
        
    end
end