function list_binary_arrays = generateThetas(grid_size)
    n = grid_size^2;
    max_number_represented = 2^n - 1;
    numbers = 0:max_number_represented;
    list_binary_strings = dec2bin(numbers);
    list_binary_arrays = zeros(grid_size, grid_size, length(numbers));

    for i = 1:length(numbers)
        character_array = reshape(list_binary_strings(i, :), [grid_size, grid_size]);

        for j = 1:grid_size
            for k = 1:grid_size
                character_array(j, k) = str2double(character_array(j, k));
            end
        end

        list_binary_arrays(:, :, i) = character_array;
    end

end