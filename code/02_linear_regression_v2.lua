#!/home/tias/torch/install/bin/th

--[[
  linear regression, v2

  using linear regression for classification on a self-generated
  dataset.
--]]

math.randomseed(os.time())

-- returns the weights-vector to use for classification
generate_target_function = function ()
  -- pick two random points from [-1, 1]x[-1, 1]
  local p1 = torch.rand(2) * 2 - 1
  local p2 = torch.rand(2) * 2 - 1

  -- calculate the slope of the line and components of w
  local m = (p2[2] - p1[2]) / (p1[1] - p2[1])
  local w = torch.Tensor(3, 1)

  w[1][1] = -p1[1] - p1[2] / m
  w[2][1] = 1
  w[3][1] = 1 / m

  return w
end

-- returns a 3xn-tensor with data and a 1xn tensor of labels
generate_data = function (n, w)
  local x = torch.rand(3, n) * 2 - 1
  x[1] = 1
  local y = torch.sign(w:t() * x)
  return x, y
end

-- trains a linear regression model and return weights w
linear_regression = function (x, y)
  local pseudo_inverse = torch.inverse(x * x:t()) * x
  return pseudo_inverse * y:t()
end

-- returns the error of w on the given labeled data points
compute_error = function (x, y, w)
  local predictions = torch.sign(w:t() * x)
  return y:ne(predictions):sum() / x:size()[2]
end

-- generates 1000 data points and estimates the model's performance
estimate_out_of_sample_error = function (w, w_model)
  local x_test, y_test = generate_data(1000, w)
  return compute_error(x_test, y_test, w_model)
end

-- trians a model on n self-generated datapoints and returns e_in
simulate_training = function (n)
  local w = generate_target_function()
  local x, y = generate_data(n, w)
  local w_linreg = linear_regression(x, y)
  local e_in = compute_error(x, y, w_linreg)
  local e_out = estimate_out_of_sample_error(w, w_linreg)
  return e_in, e_out
end


----------------- MAIN -------------------

num_simulations = 1000
num_training_data = 100
e_in = torch.Tensor(num_simulations)
e_out = torch.Tensor(num_simulations)

for i = 1, num_simulations do
  e_in[i], e_out[i] = simulate_training(num_training_data)
end

print("average in-sample error: " .. e_in:mean())
print("average out-of-sample error: " .. e_out:mean())

