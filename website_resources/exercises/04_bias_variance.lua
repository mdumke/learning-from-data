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
  hyp_y = torch.linspace(-1, 1):apply(compute_hypothesis(g_bar))

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
    return w * x
  end
end

-- returns the weight from a single trial
single_hypothesis = function ()
  local x, y = get_training_data(2)
  local pseudo_inverse = x:t() * torch.inverse(x * x:t())
  return y * pseudo_inverse
end

-- computes the average hypothesis
run_simulation = function (num_trials)
  local hypotheses = torch.Tensor(num_trials)

  for i = 1, num_trials do
    hypotheses[i] = single_hypothesis()
  end

  return round(torch.mean(hypotheses))
end

-- generates multiple random points and return the estimated bias
compute_bias_estimate = function (g_bar)
  test_x, test_y = get_training_data(1000)
  predictions = test_x:clone() * g_bar
  return torch.mean(torch.pow(predictions - test_y, 2))
end

-- generates random samples to estimate the variance
compute_variance_estimate = function (g_bar)
  local n = 1000
  errors = torch.Tensor(n)

  for i = 1, n do
    local g = single_hypothesis()[1][1]
    local x = torch.rand(1, 1000) * 2 - 1

    local y_g = x:clone() * g
    local y_g_bar = x:clone() * g_bar

    errors[i] = torch.mean(torch.pow(y_g - y_g_bar, 2))
  end

  return torch.mean(errors)
end

g_bar = run_simulation(1000)
plot_estimate(g_bar)
print("Best hypothesis estimate: " .. g_bar)
print("bias estimate: " .. round(compute_bias_estimate(g_bar)))
print("variance estimate: " .. round(compute_variance_estimate(g_bar)))

