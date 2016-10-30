#!/home/tias/torch/install/bin/th

--[[
  approximates a sine-target function with known only through 2 data points
  with a hypthesis set of the form h(x) = a * x + b
--]]

require('helpers')
math.randomseed(os.time())

-- returns w0, w1 optimizing h(x)=w0+w1x as hypothesis function
compute_hypothesis = function (x, y)
  local x_train = torch.ones(2, 2)
  x_train:t()[2] = x

  local pseudo_inverse = torch.inverse(x_train:t() * x_train) * x_train:t()
  local w = pseudo_inverse * y

  return w[1][1], w[2][1]
end

-- returns x and y, 2x1-Tensors with training data
generate_training_data = function ()
  local x = torch.rand(2, 1) * 2 - 1
  local y = torch.sin(x * math.pi)

  return x, y
end

-- returns an estimate for the bias from a simulation with n datapoints
compute_bias_estimate = function (w0, w1, n)
  local x_test = torch.rand(n, 1) * 2 - 1
  local y_test = torch.sin(x_test * math.pi)
  local y_predict = w0 + x_test * w1

  return torch.mean(torch.pow(y_predict - y_test, 2))
end

-- return an estimate for the variance based on n points and d datasets
compute_variance_estimate = function (w0, w1, n, d)
  errors = torch.zeros(d)

  for i = 1, d do
    wd0, wd1 = compute_hypothesis(generate_training_data())

    x_test = torch.rand(n, 1) * 2 - 1
    y_g = wd0 + wd1 * x_test
    y_g_bar = w0 + w1 * x_test

    errors[i] = torch.mean(torch.pow(y_g - y_g_bar, 2))
  end

  return torch.mean(errors)
end

-- returns w0, w1, an estimate of best hypothesis from n iterations of training
compute_best_hypothesis = function (n)
  local g = torch.Tensor(2, n)

  for i = 1, n do
    g[1][i], g[2][i] = compute_hypothesis(generate_training_data())
  end

  return g[1]:mean(), g[2]:mean()
end

w0, w1 = compute_best_hypothesis(10000)
bias = compute_bias_estimate(w0, w1, 10000)
var = compute_variance_estimate(w0, w1, 1000, 1000)

print("g_bar:",             round(g_bar, 2))
print("bias estimate:",     round(bias, 2))
print("variance estimate:", round(var, 2))
print("E_out estimate:",    round(bias + var, 2))

