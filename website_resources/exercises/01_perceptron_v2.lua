#!/home/tias/torch/install/bin/th

--[[
  Perceptron with Perceptron learning algorithm on
  artificial data
--]]

math.randomseed(os.time())

-- returns an nx3 tensor, dim1 is filled with 1s, dim2,3 with random values
initialize_training_input = function (n)
  xs = torch.Tensor(n, 3):apply(function () return rand_val() end)
  xs[{{}, 1}] = 1
  return xs
end

-- returns the sum of all values in the given table
sum_array = function (arr)
  sum = 0

  for _, val in pairs(arr) do
    sum = sum + val
  end

  return sum
end

-- returns a radom value from [-1, 1]
rand_val = function ()
  return math.random() * 2 - 1
end

-- returns a lua-table with n elements in random order
random_order = function (n)
  result = {}

  for i = 1, n do
    result[i] = i
  end

  for i = 1, n do
    local j = math.random(i, n)
    result[i], result[j] = result[j], result[i]
  end

  return result
end

-- returns a linear function defined on [-1, 1]x[-1, 1]
generate_target_function = function ()
  p1 = {rand_val(), rand_val()}
  p2 = {rand_val(), rand_val()}

  m = (p2[2] - p1[2]) / (p2[1] - p1[1])

  return function (x)
    return m * (x - p1[1]) + p1[2]
  end
end

-- returns +1 if the function returns a positive value, else -1
evaluate_point = function (point, f)
  if f(point[2]) > point[3] then
    return 1
  else
    return -1
  end
end

-- returns an nx1 tensor with the function applied to the given data
evaluate_target_function = function (data, f)
  local y = torch.Tensor(data:size()[1])

  for i, slice in ipairs(data:split(1, 1)) do
    y[i] = evaluate_point(slice[1], f)
  end

  return y
end

-- returns 1 if the product of x and w is positive, else -1
predict = function (x, w)
  return torch.sign(w:t() * x)[1]
end

-- returns a linear function extracted from the given weights vector
compute_hypothesis = function(w)
  local a = -(w[2][1] / w[3][1])
  local b = w[1][1] / w[3][1]

  return function (x)
    return a * x + b
  end
end

-- returns a linear function that explains the point-classes
learn_approximation = function (xs, y)
  w = torch.Tensor(3, 1):fill(0)
  local n = xs:size()[1]
  local success = false
  local steps = 0

  while not success do
    success = true

    -- pick a random misclassified point
    for _, i in pairs(random_order(n)) do
      if predict(xs[i], w) ~= y[i] then
        success = false
        steps = steps + 1
        w = w + y[i] * xs[i]
        break
      end
    end
  end

  return compute_hypothesis(w), steps
end

-- returns a hypothesis and #steps for a new target function
-- n: integer, number of generated training examples
run_perceptron = function (n)
  xs = initialize_training_input(n)
  f = generate_target_function()
  y = evaluate_target_function(xs, f)
  g, steps = learn_approximation(xs, y)

  return g, steps
end




training_examples = 10
num_trials = 1000
all_steps = {}

for i = 1, num_trials do
  g, steps = run_perceptron(training_examples)
  all_steps[#all_steps + 1] = steps
end

print(sum_array(all_steps) / num_trials)






