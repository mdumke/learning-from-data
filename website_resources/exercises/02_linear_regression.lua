#!/home/tias/torch/install/bin/th

--[[
  Linear Regression for Classification

  Using a self-generated set of linearly separable data and
  performing linear regression through matrix inversion

  usage: linear_regression.lua [num_examples] [--plot]

  When --plot is specified, there will be only 1 trial for which the
  corresponding scatterplot will be printed.
--]]

require('gnuplot')
math.randomseed(os.time())
show_plot = false

-- returns the first command line argument or 100
get_num_examples = function ()
  if arg[1] then
    return tonumber(arg[1])
  else
    return 100
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
report_results = function (e_in, e_out, num_trials, num_examples)
  print('\n-- Linear Regression for Classification --')
  print('\nTraining linear regression on a set of ' .. num_examples ..
    ' data points.')
  print('Simulation ran ' .. num_trials .. ' times.\n')
  print('  average in-sample-error: ' .. e_in)
  print('  average out-of-sample-error: ' .. e_out .. '\n')
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
  local a = -(w[2] / w[3])
  local b = -(w[1] / w[3])

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

-- returns the hypothesis and in-sample-error for one trial
run_once = function (num_examples)
  x, y, f = generate_training_data(num_examples)

  -- compute linear regression
  pseudo_inverse = torch.inverse(x:t() * x) * x:t()
  w = pseudo_inverse * y
  g = compute_hypothesis(w)

  -- evaluate hypothesis
  y_g = evaluate_function(g, x)
  in_sample_error = y:ne(y_g):sum() / num_examples
  out_of_sample_error = compute_e_out(f, g)

  if show_plot then
    plot(f, g, x, y)
  end

  return in_sample_error, out_of_sample_error
end


-- returns the mean in-sample and out-of-sample errors
run_simulation = function (num_trials, num_examples)
  in_sample_errs = torch.Tensor(num_trials)
  out_of_sample_errs = torch.Tensor(num_trials)

  for i = 1, num_trials do
    in_sample_errs[i], out_of_sample_errs[i] = run_once(num_examples)
  end

  return in_sample_errs, out_of_sample_errs
end


------------ MAIN CONTROL ---------------

-- configure simulation
num_examples = get_num_examples()
num_trials = get_num_trials()

e_in, e_out = run_simulation(num_trials, num_examples)
report_results(torch.mean(e_in), torch.mean(e_out), num_trials, num_examples)

