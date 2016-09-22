#!/home/tias/torch/install/bin/th

--[[
  Nonlinear transformation

  Using linear regression for classification of a target-distribution.
  First with linear raw data, next with nonlinear transformation.
--]]

math.randomseed(os.time())

-- returns the target function to approximate
get_target_function = function ()
  return function (x1, x2)
    if x1^2 + x2^2 - 0.6 > 0 then
      return 1
    else
      return -1
    end
  end
end

-- returns an nx3 x-tensor with a corresponding nx1 y-tensor
generate_data = function (num_examples, f)
  -- all points are drawn unifomly from within [-1, 1]
  local x = torch.rand(num_examples, 3) * 2 - 1
  x:t()[1] = 1

  -- evaluate the target function to get y
  local y = torch.Tensor(num_examples, 1)

  for i = 1, num_examples do
    y[i] = f(x[i][2], x[i][3])
  end

  -- add 10% noise
  idx = torch.randperm(num_examples):narrow(1, 1, num_examples / 10)

  for i = 1, idx:size(1) do
    y[idx[i]] = y[idx[i]] * -1
  end

  return x, y
end

-- returns an nx2 x-tensor with a corresponding nx1 y-tensor
generate_training_data = function (num_examples)
  local f = get_target_function()
  local x, y = generate_data(num_examples, f)
  return x, y, f
end

-- returns the in-sample-error when predicting with w
in_sample_error = function (x, y, w)
  predictions = torch.sign(w:t() * x:t())
  misses = torch.sum(predictions:ne(y)[1])
  return misses / x:size(1)
end

-- returns a 3x1 weights-tensor learned through linear regression
linear_regression = function (x, y)
  local pseudo_inverse = torch.inverse(x:t() * x) * x:t()
  return pseudo_inverse * y
end

-- returns an nx6 tensor with added features
compute_non_linear_features = function (x)
  local num_examples = x:size(1)
  local x_tf = torch.Tensor(num_examples, 6)
  x_tf:narrow(2, 1, 3):copy(x)

  for i = 1, num_examples do
    x_tf[i][4] = x_tf[i][2] * x_tf[i][3]
    x_tf[i][5] = x_tf[i][2]^2
    x_tf[i][6] = x_tf[i][3]^2
  end

  return x_tf
end

-- returns the estimated out-of-sample-error for the given weights
out_of_sample_error = function (w, f)
  -- generate 1000 test points
  x_test, y_test = generate_data(1000, f)
  x_test = compute_non_linear_features(x_test)

  -- in-sample-error in test data as an estimate of the out-of-sample-error!
  return in_sample_error(x_test, y_test, w)
end

-- prints the results of the simulation
report_results = function (e_in, weights, e_in_tf, e_out, num_trials, num_examples)
  print('\n--- Linear Regression with non-linear transformation ---\n')
  print('  Running ' .. num_trials .. ' trials, each with ' ..
    num_examples .. ' examples.\n')
  print(  '  * avg. in-sample-error with simple linear regression: ' .. e_in)
  print(  '  * non-linear transormation: avg. weights: ')
  for i = 1, weights:size(2) do
    print('      ' ..  weights[1][i])
  end
  print(  '  * avg. in-sample-error after transformation: ' .. e_in_tf .. '.')
  print(  '  * avg. out-of-sample-error after transformation: ' .. e_out .. '.\n')
end

-- main loop for running the simulation
run_simulation = function (num_examples, num_trials)
  e_ins = torch.Tensor(num_trials)
  w_tfs = torch.Tensor(num_trials, 6)
  e_in_tfs = torch.Tensor(num_trials)
  e_outs = torch.Tensor(num_trials)

  for i = 1, num_trials do
    x, y, f = generate_training_data(num_examples)

    -- simple linear regression
    w = linear_regression(x, y)
    e_ins[i] = in_sample_error(x, y, w)

    -- linear regression with nonlinear transformation
    x_tf = compute_non_linear_features(x)
    w_tf = linear_regression(x_tf, y)
    w_tfs[i] = w_tf
    e_in_tfs[i] = in_sample_error(x_tf, y, w_tf)
    e_outs[i] = out_of_sample_error(w_tf, f)
  end

  return torch.mean(e_ins), torch.mean(w_tfs, 1), torch.mean(e_in_tfs), torch.mean(e_outs)
end



---------------- MAIN CONTROL -------------------

num_examples = 1000
num_trials = 1000

print('running simulation, please wait...')

e_in, w, e_in_tf, e_out = run_simulation(num_examples, num_trials)
report_results(e_in, w, e_in_tf, e_out, num_trials, num_examples)






