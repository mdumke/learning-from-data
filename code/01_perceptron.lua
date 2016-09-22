#!/home/tias/torch/install/bin/th

--[[
  Perceptron with Perceptron learning algorithm on
  artificial data
--]]

require('gnuplot')
math.randomseed(os.time())

-- returns an nx3 tensor, dim1 is filled with 1s, dim2,3 with random values
initialize_training_input = function (n)
  xs = torch.Tensor(n, 3):apply(function () return rand_val() end)
  xs[{{}, 1}] = 1
  return xs
end

-- plots the given functions
plot = function (f, g)
  f_x = torch.linspace(-1, 1)
  f_y = torch.linspace(-1, 1):apply(f)

  g_x = torch.linspace(-1, 1)
  g_y = torch.linspace(-1, 1):apply(g)

  gnuplot.figure(1)
  gnuplot.axis({-1, 1, -1, 1})
  gnuplot.xlabel('x')
  gnuplot.ylabel('y')
  gnuplot.title('target vs hypothesis')

  gnuplot.plot(
    {'f(x)', f_x, f_y, '-'},
    {'g(x)', g_x, g_y, '-'}
  )
end

-- prints a starting message with basic information
report_general_information = function (num_trials, training_examples)
  print('--------')
  print('- Perceptron Learning Algorithm')
  print('-')
  print('- running with ' .. num_trials .. ' trials, each with ' ..
    training_examples .. ' training examples...')
end

-- prints the given numbers to the command line
report_results = function (avg_steps, avg_p_miss)
  print('-  * average convergence after ' .. avg_steps .. ' steps')
  print('-  * average missclassification rate is ' .. avg_p_miss)
  print('--------')
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
evaluate_point = function (x1, x2, f)
  if f(x1) > x2 then
    return 1
  else
    return -1
  end
end

-- returns an nx1 tensor with the function applied to the given data
evaluate_target_function = function (data, f)
  local y = torch.Tensor(data:size()[1])

  for i, slice in ipairs(data:split(1, 1)) do
    y[i] = evaluate_point(slice[1][2], slice[1][3], f)
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
  local b = -(w[1][1]) / w[3][1]

  return function (x)
    return a * x + b
  end
end

-- returns the probability that f and g disagree on a point, uses n examples
validate = function(f, g, n)
  local total_miss = 0

  for i = 1, n do
    local x = {rand_val(), rand_val()}

    if evaluate_point(x[1], x[2], f) ~= evaluate_point(x[1], x[2], g) then
      total_miss = total_miss + 1
    end
  end

  return total_miss / n
end

-- returns a linear function that explains the point-classes
learn_approximation = function (xs, y, f)
  local w = torch.Tensor(3, 1):fill(0)
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

  local g = compute_hypothesis(w)
  local validation_set_size = 1000
  local p_miss = validate(f, g, validation_set_size)

  return g, p_miss, steps
end

-- returns #steps to comp. g and P(g is wrong), for random target fn
-- n: integer, number of generated training examples
run_perceptron = function (n)
  local xs = initialize_training_input(n)
  local f = generate_target_function()
  local y = evaluate_target_function(xs, f)

  g, p_miss, steps = learn_approximation(xs, y, f)

  return f, g, p_miss, steps
end

-- returns average number of missclass. and steps after simulation
run_simulation = function (num_trials, num_training_examples)
  local total_steps = 0
  local total_p_miss = 0

  for i = 1, num_trials do
    f, g, p_miss, steps = run_perceptron(num_training_examples)
    total_steps = total_steps + steps
    total_p_miss = total_p_miss + p_miss
  end

  -- plot(f, g)

  local avg_steps = total_steps / num_trials
  local avg_p_miss = total_p_miss / num_trials

  return avg_steps, avg_p_miss
end



---------------- MAIN CONTROL ------------------

if arg[1] then
  training_examples = tonumber(arg[1])
else
  training_examples = 1000
end

num_trials = 1000

report_general_information(num_trials, training_examples)
report_results(run_simulation(num_trials, training_examples))




