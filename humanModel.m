function probability = humanModel(policy_index, theta, robot_state, policies, discount)
    [value, q_value, optimal_policies] = valueIteration(theta, discount);
    exp_q_vals = zeros(1, length(policies));
    for i = 1:length(policies)
        exp_q_vals(i) = exp(q_value(robot_state(1), robot_state(2), i));
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
