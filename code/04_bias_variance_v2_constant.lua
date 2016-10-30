#!/home/tias/torch/install/bin/th

--[[
  approximates a sine-target function with known only through 2 data points
  with a hypthesis set of the form h(x) = b
--]]

require('helpers')
math.randomseed(os.time())

-- returns  a single b optimizing h(x)=b as hypothesis function
compute_hypothesis = function (x, y)
  x_train = torch.ones(2, 1)
  local pseudo_inverse = torch.inverse(x_train:t() * x_train) * x_train:t()
  return (pseudo_inverse * y)[1][1]
end

-- returns x and y, 2x1-Tensors with training data
generate_training_data = function ()
  local x = torch.rand(2, 1) * 2 - 1
  local y = torch.sin(x * math.pi)

  return x, y
end

-- returns an estimate for the bias from a simulation with n datapoints
compute_bias_estimate = function (b, n)
  local x_test = torch.rand(n, 1) * 2 - 1
  local y_test = torch.sin(x_test * math.pi)
  local y_predict = b

  return torch.mean(torch.pow(y_predict - y_test, 2))
end

-- return an estimate for the variance based on n points and d datasets
compute_variance_estimate = function (g_bar, n, d)
  local errors = torch.zeros(d)

  for i = 1, d do
    local g = compute_hypothesis(generate_training_data())
    errors[i] = (g - g_bar)^2
  end

  return torch.mean(errors)
end

-- returns an estimate of best hypothesis from n iterations of training
compute_best_hypothesis = function (n)
  local g = torch.Tensor(n)

  for i = 1, n do
    g[i] = compute_hypothesis(generate_training_data())
  end

  return g:mean()
end

g_bar = compute_best_hypothesis(10000)
bias = compute_bias_estimate(g_bar, 1000)
var = compute_variance_estimate(g_bar, 1000, 1000)

print("g_bar:",             round(g_bar, 2))
print("bias estimate:",     round(bias, 2))
print("variance estimate:", round(var, 2))
print("E_out estimate:",    round(bias + var, 2))

