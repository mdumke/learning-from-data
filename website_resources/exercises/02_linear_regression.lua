#!/home/tias/torch/install/bin/th

--[[
  Linear Regression for Classification

  Using a self-generated set of linearly separable data and
  performing linear regression through matrix inversion
--]]

require('gnuplot')
math.randomseed(os.time())

-- returns the first command line argument or 100
get_num_examples = function ()
  if arg[1] then
    return tonumber(arg[1])
  else
    return 100
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

-- returns a linear function that classifies points to +/- 1
generate_target_function = function ()
  p1 = {rand_val(), rand_val()}
  p2 = {rand_val(), rand_val()}

  m = (p2[2] - p1[2]) / (p2[1] - p1[1])

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
  local b = -(w[1]) / w[3]

  return function (x)
    return a * x + b
  end
end

-- returns an estimate for the out-of-sample performance of g compared to f
compute_e_out = function (f, g)
  local data = get_data_points(1000)
  local y_f = evaluate_function(f, data)
  local y_g = evaluate_function(g, data)

  return torch.sum(torch.abs(y_f - y_g)) / 2 / 1000
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
  in_sample_error = torch.sum(torch.abs(y - y_g)) / 2 / num_examples
  out_of_sample_error = compute_e_out(f, g)

  -- plot(f, g)

  return in_sample_error, out_of_sample_error
end


-- returns the mean in-sample and out-of-sample errors
run_simulation = function (num_trials, num_examples)
  in_sample_errs = torch.Tensor(num_trials)
  out_of_sample_errs = torch.Tensor(num_trials)

  for i = 1, num_trials do
    in_sample_errs[i], out_of_sample_errs[i] = run_once(num_examples)
  end

  return torch.mean(in_sample_errs), torch.mean(out_of_sample_errs)
end


------------ MAIN CONTROL ---------------

-- configure simulation
num_examples = get_num_examples()
num_trials = 1000

e_in, e_out = run_simulation(num_trials, num_examples)
report_results(e_in, e_out, num_trials, num_examples)



