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
    {'g_bar', hyp_x, hyp_y, '-'})
end

-- returns the target function sin(pi * x)
get_target_function = function ()
  return function (x)
    return math.sin(math.pi * x)
  end
end

-- returns 2 training points
get_training_data = function ()
  local x = torch.rand(1, 2) * 2 - 1
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
run_once = function ()
  local x, y = get_training_data()
  local pseudo_inverse = x:t() * torch.inverse(x * x:t())
  return y * pseudo_inverse
end

-- computes the average hypothesis
run_simulation = function (num_trials)
  local hypotheses = torch.Tensor(num_trials)

  for i = 1, num_trials do
    hypotheses[i] = run_once()
  end

  return round(torch.mean(hypotheses))
end

g_bar = run_simulation(2000)
plot_estimate(g_bar)
print("Best hypothesis estimate: " .. g_bar)

