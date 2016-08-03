#!/home/tias/torch/install/bin/th

--[[
  performing bias-variance analysis for
  the function sin(pi * x) and n=2 training points
--]]

require('gnuplot')
math.randomseed(os.time())

-- rounds the result to 2 decimal places
round = function (x)
  return math.floor(x * 100 + 0.5) / 100
end

-- displays the simulation/approximation results
plot_estimate = function (g_bar)
  sin_x = torch.linspace(-1, 1)
  sin_y = torch.linspace(-1, 1):apply(get_target_function())

  hyp_x = torch.linspace(-1, 1)
  hyp_y = torch.linspace(-1, 1):apply(g_bar)

  gnuplot.figure(1)
  gnuplot.axis({-1.2, 1.2, -1.2, 1.2})
  gnuplot.xlabel('x')
  gnuplot.ylabel('y')
  gnuplot.title('best hypothesis through averaging')

  gnuplot.plot(
    {'target', sin_x, sin_y, '-'},
    {'g bar', hyp_x, hyp_y, '-'})
end

-- returns the target function sin(pi * x)
get_target_function = function ()
  return function (x)
    return math.sin(math.pi * x)
  end
end

-- returns 2 training points
get_training_data = function (n)
  local x = torch.rand(1, n) * 2 - 1
  local y = x:clone():apply(get_target_function())
  return x, y
end

-- returns the hypothesis as a function of x
compute_hypothesis = function (w)
  return function (x)
    return w[1][2] * x + w[1][1]
  end
end

-- returns the weight from a single trial
run_once = function ()
  x, y = get_training_data(2)

  -- adapt x to be suitable for the hypothesis computation
  x_train = torch.ones(2, 2)
  x_train:t()[2] = x

--   local pseudo_inverse = x_train:t() * torch.inverse(x_train * x_train:t())
--   return y * pseudo_inverse
  pseudo_inverse = torch.inverse(x_train:t() * x_train) * x_train:t()
  return pseudo_inverse * y:t()
end

-- computes the average hypothesis
run_simulation = function (num_trials)
  weights = torch.Tensor(num_trials, 2)

  for i = 1, num_trials do
    weights[i] = run_once()
  end

  w_avg = torch.Tensor(1, 2)
  w_avg[1][1] = round(torch.mean(weights[{{}, 1}]))
  w_avg[1][2] = round(torch.mean(weights[{{}, 2}]))

  return compute_hypothesis(w_avg)
end

-- generates multiple random points and return the estimated bias
compute_bias_estimate = function (g_bar)
  x_test, y_test = get_training_data(1000)
  predictions = g_bar(x_test)
  return torch.mean(torch.pow(predictions - y_test, 2))
end

-- generates random samples to estimate the variance
compute_variance_estimate = function (g_bar)
  local n = 1000
  errors = torch.Tensor(n)

  for i = 1, n do
    w = run_once()
    g = compute_hypothesis(w:t())
    x = torch.rand(1, 1000) * 2 - 1

    y_g = x:clone():apply(g)
    y_g_bar = x:clone():apply(g_bar)

    errors[i] = torch.mean(torch.pow(y_g - y_g_bar, 2))
  end

  return torch.mean(errors)
end

g_bar = run_simulation(1000)

plot_estimate(g_bar)
print("bias estimate: " .. round(compute_bias_estimate(g_bar)))
print("variance estimate: " .. round(compute_variance_estimate(g_bar)))

