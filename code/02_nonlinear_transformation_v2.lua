#!/home/tias/torch/install/bin/th

--[[
  nonlinear transformation

  runs linear regression on self generated data. in a second step,
  use a nonlinear transformation and re-run regression. compare results
--]]

require('helpers')
math.randomseed(os.time())

-- the target function as defined by the pset
target_function = function (x1, x2)
  return sign(x1^2 + x2^2 - 0.6)
end

-- returns n datapoints with labels for the given function
generate_data = function (n, target)
  local x = torch.rand(n, 3) * 2 - 1
  x:t()[1] = 1
  local y = torch.Tensor(n, 1)

  for i = 1, n do
    y[i][1] = target(x[i][2], x[i][3])
  end

  return x:t(), y:t()
end

-- switches signs on p% of the given lables
add_noise = function (y, p)
  local n = y:size()[2]
  local idx = torch.randperm(n):narrow(1, 1, n / p)

  for i = 1, idx:size(1) do
    y[1][idx[i]] = y[1][idx[i]] * -1
  end
end

-- runs linear regression and returns a weights vector
linear_regression = function (x, y)
  local pseudo_inverse = torch.inverse(x * x:t()) * x
  return pseudo_inverse * y:t()
end

-- applies w to x and returns the fraction of missclassified points
compute_error = function (x, y, w)
  local predictions = torch.sign(w:t() * x)
  return y:ne(predictions):sum() / y:size()[2]
end

-- performs a non-linear transformation according to the pset
transform = function (x1, x2)
  return torch.Tensor({1, x1, x2, x1 * x2, x1^2, x2^2})
end

-- performs a non-linear transform of the data as required by the pset
nonlinear_transform = function (x)
  local n = x:size()[2]
  local x_tf = torch.Tensor(n, 6)

  for i = 1, n do
    x_tf[i] = transform(x:t()[i][2], x:t()[i][3])
  end

  return x_tf:t()
end

-- generates data, runs regressions, reports errors
run_once = function (n)
  local x, y = generate_data(n, target_function)
  add_noise(y, 10)

  -- 1. simple linear regression
  local w_linreg = linear_regression(x, y)
  local e_in_linreg = compute_error(x, y, w_linreg)

  -- 2. with nonlinear transformation
  local x_transform = nonlinear_transform(x)
  local w_transform = linear_regression(x_transform, y)
  local e_in_transform = compute_error(x_transform, y, w_transform)

  -- 3. estimate out of sample error with nonlinear transformation
  local x_test, y_test = generate_data(1000, target_function)
  add_noise(y_test, 10)
  x_test_transform = nonlinear_transform(x_test)
  local e_out_transform =  compute_error(x_test_transform, y_test, w_transform)

  return e_in_linreg, e_in_transform, e_out_transform, w_transform
end

num_trials = 1000
n = 100
e_in_linreg = torch.Tensor(num_trials)
e_in_transform = torch.Tensor(num_trials)
e_out_transform = torch.Tensor(num_trials)
w_transform = torch.Tensor(num_trials, 6)

for i = 1, num_trials do
  e_in_linreg[i], e_in_transform[i], e_out_transform[i], w_transform[i] = run_once(n)
end

print("avg. e_in: " .. torch.mean(e_in_linreg))
print("avg. e_in with transform: " .. torch.mean(e_in_transform))
print("avg. e_out with transform: " .. torch.mean(e_out_transform))
print("avg. w_transform: ")
print(torch.mean(w_transform, 1))




