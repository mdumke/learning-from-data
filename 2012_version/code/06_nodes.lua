#!/home/tias/torch/install/bin/th

--[[
  Helper for finding the maximum number of weights for different
  distributions of hidden units in a fully connected neural network

  network structure:

  10 inputs -> 36 hidden units -> 1 output
--]]

weights_for_two_layers = function ()
  max_weights = 0

  for i = 2, 34 do
    l1 = i
    l2 = 36 - i

    weights = 10 * (l1 - 1) + l1 * (l2 - 1) + l2

    if weights > max_weights then
      max_weights = weights
    end
  end

  return max_weights
end


weights_for_three_layers = function ()
  max_weights = 0

  for i = 2, 32 do
    l1 = i
    remaining = 36 - i

    for j = 2, remaining do
      l2 = j
      l3 = remaining - j

      weights = 10 * (l1 - 1) + l1 * (l2 - 1) + l2 * (l3 - 1) + l3

      if weights > max_weights then
        max_weights = weights
      end
    end

    return max_weights
  end
end

weights_for_four_layers = function ()
  max_weights = 0

  for i = 2, 30 do
    l1 = i

    for j = 2, (36 - i) do
      l2 = j

      remaining = 36 - i - j
      for k = 2, remaining do
        l3 = k
        l4 = remaining - k

        weights = 10 * (l1 - 1) + l1 * (l2 - 1) + l2 * (l3 - 1) + l3 * (l4 - 1) + l4

        if weights > max_weights then
          max_weights = weights
        end
      end

    end

    return max_weights
  end
end

print(weights_for_two_layers())
print(weights_for_three_layers())
print(weights_for_four_layers())


