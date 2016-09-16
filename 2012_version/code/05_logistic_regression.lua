#!/home/tias/torch/install/bin/th

--[[
  Logistic Regression
--]]

require('gnuplot')
math.randomseed(os.time())

-- returns a random value between -1 and 1
rand_val = function ()
  return math.random() * 2 - 1
end

-- returns a function that classifies points to +/- 1
generate_target_function = function ()
  -- make sure the function is not too steep
  repeat
    p1 = {rand_val(), rand_val()}
    p2 = {rand_val(), rand_val()}

    m = (p2[2] - p1[2]) / (p2[1] - p1[1])
  until math.abs(m) < 20

  return function (x)
    return m * (x - p1[1]) + p1[2]
  end
end

-- returns +/- 1 depending on where the point lies
evaluate_point = function (target_fn, x1, x2)
  return target_fn(x1) < x2 and 1 or -1
end

-- returns a 3xn-tensor with training data and a 1xn-tensor with results
generate_training_data = function (n, f)
  xs = torch.rand(n, 3) * 2 - 1
  xs:t()[1] = 1
  y = torch.Tensor(n, 1)

  for i = 1, n do
    y[i] = evaluate_point(f, xs[i][2], xs[i][3])
  end

  return xs, y
end

-- plot the target function
plot = function (f, x1, x2)
  f_x = torch.linspace(-1.5, 1.5)
  f_y = torch.linspace(-1.5, 1.5):apply(f)

  gnuplot.figure(1)
  gnuplot.axis({-1.5, 1.5, -1.5, 1.5})
  gnuplot.xlabel('x')
  gnuplot.ylabel('y')
  gnuplot.title('target function')

  gnuplot.plot(
    {'f(x)', f_x, f_y, '-'},
    {'data', x1, x2, '.'}
  )
end

-- returns the gradient evaluated at given w using one point xs / y
compute_gradient = function (w, xs, y)
  g1 = (- y * xs[1] / (1 + math.exp(y * (w:t() * xs))))[1]
  g2 = (- y * xs[2] / (1 + math.exp(y * (w:t() * xs))))[1]
  g3 = (- y * xs[3] / (1 + math.exp(y * (w:t() * xs))))[1]
  return torch.Tensor({g1, g2, g3})
end

-- returns the given tensors in randomized order
permutation = function (xs, y)
  n = xs:size()[1]
  idx = torch.randperm(n)

  new_xs = torch.Tensor(xs:size())
  new_y = torch.Tensor(y:size())

  for i = 1, n do
    new_xs[i] = xs[idx[i]]
    new_y[i] = y[idx[i]]
  end

  return new_xs, new_y
end

-- returns the cross-entropy error for the given weights and data
compute_error = function (w, xs, y)
  n = xs:size()[2]
  total_error = 0

  for i = 1, n do
    total_error =
      total_error + math.log(1 + math.exp(-y[i][1] * (w:t() * xs[i])[1]))
  end

  return total_error / n
end

-- computes the sigmoid function for logistic regression
sigmoid = function (s)
  return math.exp(s) / (1 + math.exp(s))
end

-- returns the o.o. sample error estimated on 1000 datapoints
out_of_sample_estimate = function (w, target_fn)
  xs, y = generate_training_data(1000, target_fn)
  return compute_error(w, xs, y)
end

-- returns the epoch and final error after SGD
single_trial = function ()
  n = 100
  eta = 0.01
  f = generate_target_function()
  xs, y = generate_training_data(n, f)
  epoch_counter = 0
  old_w = torch.ones(3, 1)
  w = torch.zeros(3, 1)

  while torch.norm(old_w - w) >= 0.01 do
    old_w = w
    xs, y = permutation(xs, y)

    -- SGD updates over one epoch
    for i = 1, n do
      grad = compute_gradient(w, xs[i], y[i])
      w = w - eta * grad
    end

    epoch_counter = epoch_counter + 1
  end

  return epoch_counter, out_of_sample_estimate(w, f)
end


------------------
-- main control --
------------------

num_trials = 100

all_epochs = torch.Tensor(num_trials)
all_e_out = torch.Tensor(num_trials)

for i = 1, num_trials do
  print("running trial " .. i)
  all_epochs[i], all_e_out[i] = single_trial()
end

print("average number of epochs: " .. torch.sum(all_epochs) / num_trials)
print("average out of sample error: " .. torch.sum(all_e_out) / num_trials)

