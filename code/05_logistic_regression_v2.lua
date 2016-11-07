#!/home/tias/torch/install/bin/th

--[[
  Runs logistic regression on a generated set of data points
--]]

require('helpers')

ETA = 0.01

-- returns a target-function in the form of a 3x1-tensor of weights
generate_target_function = function ()
  -- pick two points from [-1, 1]x[-1, 1]
  local x = torch.rand(2, 2) *2 - 1

  -- set w2 arbitrarily, this will determine w0 and w1
  local w2 = 1
  local w1 = - w2 * (x[1][2] - x[2][2]) / (x[1][1]-x[2][1])
  local w0 = - w2 * x[1][2] - w1 * x[1][1]

  local target = torch.Tensor(1, 3)
  target[1] = torch.Tensor({w0, w1, w2})

  return target:t()
end

-- returns data points x (nx3-tensor) and y (nx1-tensor) generated from w
generate_data = function (w, n)
  local x = torch.rand(n, 3) * 2 - 1
  x:t()[1] = 1
  local y = torch.sign(w:t() * x:t()):t()
  return x, y
end

-- returns the logistic-regression-gradient at the given point
compute_gradient = function (x, y, w)
  return - y[1] * x / (1 + torch.exp(w:t() * x * y[1]))[1]
end

-- returns the cross-entropy-error
compute_error = function (x, y, w)
  local length = x:size()[1]
  local error_terms = torch.Tensor(length)

  for i = 1, length do
    error_terms[i] = torch.log(1 + torch.exp(-y[i][1] * w:t() * x[i]))
  end

  return torch.mean(error_terms)
end

-- computes a cross-entropy-error estimate for the out-of-sample error
estimate_e_out = function (w_target, w)
  x_test, y_test = generate_data(w_target, 1000)
  return compute_error(x_test, y_test, w)
end



-- runs the logistic-regression simulation and returns e_out after convergence
run_logistic_regression =  function (n)
  w_target = generate_target_function()
  x, y = generate_data(w_target, n)

  w = torch.zeros(3, 1)
  epoch = 1

  repeat
    w_prev = w
    x, y = random_permutation(x, y)

    for i = 1, n do
      grad = compute_gradient(x[i], y[i], w)
      w = w - ETA * grad
    end

    epoch = epoch + 1
  until (torch.norm(w_prev - w) < 0.01)

  return estimate_e_out(w_target, w), epoch
end


n = 100
num_trials = 20

error_estimates = torch.Tensor(num_trials)
epochs = torch.Tensor(num_trials)

for i = 1, num_trials do
  if (i % 10 == 0) then print(i) end
  error_estimates[i], epochs[i] = run_logistic_regression(n)
end

print("estiated e_out:", torch.mean(error_estimates))
print("mean epochs:", torch.mean(epochs))



