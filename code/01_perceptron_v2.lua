#!/home/tias/torch/install/bin/th

--[[
  perceptron v2

  generates linearly separable training data and trains a perceptron
  on it. Reports the results of 1000 runs of this experiment.
--]]

require('gnuplot')
require('helpers')
math.randomseed(os.time())

-- plots the given data
plot = function (x, y)
  positive = x:index(1, y:eq(1):nonzero():t()[1])
  negative = x:index(1, y:eq(-1):nonzero():t()[1])

  gnuplot.plot(
    {'+1', positive:t()[1], positive:t()[2], '+'},
    {'-1', negative:t()[1], negative:t()[2], '+'})
end

-- returns the misclassifications on 1000 random points
estimate_error = function (f, h)
  local n = 1000
  x_test, y = generate_training_data(n, f)
  predictions = torch.Tensor(1, n)

  for i = 1, n do
    predictions[1][i] = h(x_test[i][1], x_test[i][2])
  end

  return y:ne(predictions):sum() / n
end

-- returns a target function that maps (x1, x2) -> +/- 1
generate_target_function = function ()
  local x1 = torch.rand(2) * 2 - 1
  local x2 = torch.rand(2) * 2 - 1

  local slope = (x2[2] - x1[2]) / (x2[1] - x1[1])
  local intercept = x1[2] - slope * x1[1]

  return function (x1, x2)
    if x1 * slope + intercept < x2 then
      return 1
    else
      return -1
    end
  end
end

-- returns training data by evaluating the given target function
generate_training_data = function (n, target)
  local x = torch.rand(n, 2) * 2 - 1
  local y = torch.rand(n, 1)

  for i = 1, n do
    y[i][1] = target(x[i][1], x[i][2])
  end

  return x, y
end

-- runs the perceptron learning algorithm and returns final hypothesis
train_perceptron = function (x, y)
  -- add 1-column to xs
  local x_train = torch.ones(x:size()[1], 3)
  x_train:t()[2] = x:t()[1]
  x_train:t()[3] = x:t()[2]

  -- initialize weights
  local w = torch.zeros(3, 1)
  local steps = 0

  repeat
    -- determine misclassified points
    local predictions = torch.sign(x_train * w)
    local missed_indices = y:ne(predictions):nonzero()

    -- test for convergence
    if missed_indices:dim() == 0 then
      break
    end

    -- pick a missclassified point and update weights
    missed_indices = missed_indices:t()[1]
    local missed_idx = missed_indices[math.random(missed_indices:size()[1])]
    w = w + y[missed_idx][1] * x_train[missed_idx]
    steps = steps + 1
  until false

  -- return w? or g using w?
  return steps, function (x, y)
    if w[1][1] + w[2][1] * x + w[3][1] * y <= 0 then
      return -1
    else
      return 1
    end
  end
end

-- runs PLA on generated data and returns number of training steps and e_out
generate_and_evaluate_perceptron = function (num_training_examples)
  local f = generate_target_function()
  local x, y = generate_training_data(num_training_examples, f)
  local steps, h = train_perceptron(x, y)
  local e_out = estimate_error(f, h)

  return steps, e_out
end



---------------- RUN SIMULATION ----------------------

num_training_examples = 100
num_runs = 1000

steps = torch.Tensor(num_runs)
e_out = torch.Tensor(num_runs)

for i = 1, num_runs do
  steps[i], e_out[i] = generate_and_evaluate_perceptron(num_training_examples)
end

print("avg number of steps: " .. round(torch.mean(steps), 2))
print("avg out of sample error: " .. round(torch.mean(e_out), 2))

