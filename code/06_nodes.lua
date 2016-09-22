#!/home/tias/torch/install/bin/th

--[[
  Helper for finding the maximum and minimum number of weights for different
  distributions of hidden units in a fully connected neural network

  network structure:

  10 inputs -> 36 hidden units in any number of layers -> 1 output
--]]

find_max_weights = function (num_inputs, num_hidden_units)
  if num_hidden_units == 0 then
    return num_inputs
  end

  local max_weights = 0

  for size_next_layer = 2, num_hidden_units do
    local weights = num_inputs * (size_next_layer - 1) + find_max_weights(size_next_layer, num_hidden_units - size_next_layer)

    if weights > max_weights then
      max_weights = weights
    end
  end

  return max_weights
end

print('max # weights for 10 inputs and 36 hidden units: ' ..
  find_max_weights(10, 36) .. '.')


find_min_weights = function (num_inputs, num_hidden_units)
  if num_hidden_units == 0 then
    return num_inputs
  end

  local min_weights = 2^32

  for size_next_layer = 2, num_hidden_units do
    local weights = num_inputs * (size_next_layer - 1) + find_min_weights(size_next_layer, num_hidden_units - size_next_layer)

    if weights < min_weights then
      min_weights = weights
    end
  end

  return min_weights
end

print('min # weights for 10 inputs and 36 hidden units: ' ..
  find_min_weights(10, 36) .. '.')

