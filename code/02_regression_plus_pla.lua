#!/home/tias/torch/install/bin/th

--[[
  Linear Regression with PLA for Classification

  Using linear regression to get a head start before applying the
  Perceptron Learning Algorithm

  usage: 02_regression_with_pla [num_examples] [--plot]
  When run with --plot, there will be only one trial
--]]

require('gnuplot')
math.randomseed(os.time())
show_plot = false

-- returns the first command line argument or 100
get_num_examples = function ()
  if arg[1] then
    return tonumber(arg[1])
  else
    return 10
  end
end

-- returs the number of times to run the simulation
get_num_trials = function ()
  if arg[2] and arg[2] == '--plot' then
    show_plot = true
    return 1
  else
    return 1000
  end
end

-- returns a random value between -1 and 1
rand_val = function ()
  return math.random() * 2 - 1
end

-- prints the results to stdout
report_results = function (num_trials, num_examples, num_steps)
  print('\n-- Linear Regression with PLA for Classification --')
  print('Simulation ran ' .. num_trials .. ' times with ' .. num_examples ..
    ' examples.\n')
  print('  average convergence of PLA: ' .. num_steps .. ' steps.\n')
end

-- plots the given functions
plot = function (f, g, x, y)
  f_x = torch.linspace(-1.5, 1.5)
  f_y = torch.linspace(-1.5, 1.5):apply(f)

  g_x = torch.linspace(-1.5, 1.5)
  g_y = torch.linspace(-1.5, 1.5):apply(g)

  above_x = x:index(1, y:eq(1):nonzero():t()[1]):t()[2]
  above_y = x:index(1, y:eq(1):nonzero():t()[1]):t()[3]

  below_x = x:index(1, y:eq(-1):nonzero():t()[1]):t()[2]
  below_y = x:index(1, y:eq(-1):nonzero():t()[1]):t()[3]

  gnuplot.figure(1)
  gnuplot.axis({-1.5, 1.5, -1.5, 1.5})
  gnuplot.xlabel('x')
  gnuplot.ylabel('y')
  gnuplot.title('target vs hypothesis')

  gnuplot.plot(
    {'f(x)', f_x, f_y, '-'},
    {'g(x)', g_x, g_y, '-'},
    {'1', above_x, above_y, '+'},
    {'-1', below_x, below_y, '+'}
  )
end

-- returns a linear function that classifies points to +/- 1
generate_target_function = function ()
  -- make sure the function is not too steep
  repeat
    p1 = {rand_val(), rand_val()}
    p2 = {rand_val(), rand_val()}

    m = (p2[2] - p1[2]) / (p2[1] - p1[1])
  until math.abs(m) < 5


  return function (x)
    return m * (x - p1[1]) + p1[2]
  end
end

-- returns an nx3 tensor with dim1 = 1, dim2,3 in [-1,1]
get_data_points = function (n)
  local data = torch.rand(n, 3) * 2 - 1
  data:t()[1] = 1
  return data
end

-- returns an nx1 tensor of +/-1 by evaluating the given function on x
evaluate_function = function (f, x)
  local length = x:size()[1]
  local result = torch.Tensor(length)

  for i = 1, length do
    if f(x[i][2]) > x[i][3] then
      result[i] = 1
    else
      result[i] = -1
    end
  end

  return result
end

-- returns data points x with target values y and the corresponding function f
generate_training_data = function (num_examples)
  local x = get_data_points(num_examples)
  local f = generate_target_function()
  local y = evaluate_function(f, x)

  return x, y, f
end

-- returns a linear function extracted from the given weights vector
compute_hypothesis = function(w)
  local a = -(w[2][1] / w[3][1])
  local b = -(w[1][1]) / w[3][1]

  return function (x)
    return a * x + b
  end
end

-- returns an estimate for the out-of-sample performance of g compared to f
compute_e_out = function (f, g)
  data = get_data_points(1000)
  y_f = evaluate_function(f, data)
  y_g = evaluate_function(g, data)

  return y_f:ne(y_g):sum() / 1000
end

-- returns the learned hypothesis and the number of steps pla needed
run_pla = function (x, y, w_linreg)
  w = w_linreg:resizeAs(torch.Tensor(1, 3))
  steps = 0
  success = false

  while not success do
    success = true

    -- evaluate the current hypothesis
    y_h = torch.sign(w * x:t())
    miss_idx = y_h[1]:ne(y):nonzero()
    num_miss = torch.numel(miss_idx)

    if num_miss > 0 then
      -- pick a missclassified point at random
      rand_idx = miss_idx[math.random(num_miss)][1]

      -- apply weights-update using this point
      w = w + y[rand_idx] * x[rand_idx]

      success = false
      steps = steps + 1
    end
  end

  g = compute_hypothesis(w:t())
  return g, steps
end

-- returns the hypothesis and in-sample-error for one trial
run_once = function (num_examples)
  x, y, f = generate_training_data(num_examples)

  -- compute linear regression
  pseudo_inverse = torch.inverse(x:t() * x) * x:t()
  w_linreg = pseudo_inverse * y
  g, num_pla_steps = run_pla(x, y, w_linreg)

  -- evaluate hypothesis
  y_g = evaluate_function(g, x)
  total_miss = y:ne(y_g):sum()
  in_sample_error = total_miss / num_examples
  out_of_sample_error = compute_e_out(f, g)

  if show_plot then
    plot(f, g, x, y)
  end

  return num_pla_steps
end


-- returns the mean number of steps the pla needed to converge
run_simulation = function (num_trials, num_examples)
  pla_steps = torch.Tensor(num_trials)

  for i = 1, num_trials do
    pla_steps[i] = run_once(num_examples)
  end

  return torch.mean(pla_steps)
end


------------ MAIN CONTROL ---------------

-- configure simulation
num_examples = get_num_examples()
num_trials = get_num_trials()

pla_steps = run_simulation(num_trials, num_examples)
report_results(num_trials, num_examples, pla_steps)


