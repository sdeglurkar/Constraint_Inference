grid_size = 10;
final_value = zeros(grid_size);
final_value(1, 1) = -1;
final_value(6, 7) = -1;
discount = 0.9;
[value, q_value, optimal_policies] = valueIteration(final_value, discount);

%surf(value)
value
optimal_policies