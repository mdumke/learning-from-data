#!/home/tias/torch/install/bin/th

--[[
  Linear Regression for Classification

  Using a self-generated set of linearly separable data and
  performing linear regression through matrix inversion
--]]

math.randomseed(os.time())

-- configure simulation
num_examples = 100
num_trials = 1000

-- returns a random value between -1 and 1
rand_val = function ()
  return math.random() * 2 - 1
end

-- returns a linear function that classifies points to +/- 1
generate_target_function = function ()
  p1 = {rand_val(), rand_val()}
  p2 = {rand_val(), rand_val()}

  m = (p2[2] - p1[2]) / (p2[1] - p1[1])

  return function (x)
    return m * (x - p1[1]) + p1[2]
  end
end

-- returns an nx1 tensor of +/-1 by evaluating the given function on x
evaluate_function = function (f, x)
  result = torch.Tensor(num_examples)

  for i = 1, num_examples do
    if f(x[i][2]) > x[i][3] then
      result[i] = 1
    else
      result[i] = -1
    end
  end

  return result
end

-- returns data points x with target values y and the corresponding function f
generate_training_data = function (num_examples)
  -- nx3 tensor with random values between -1, 1 and added helper constant 1
  local x = torch.rand(num_examples, 3) * 2 - 1
  x:t()[1] = 1

  local f = generate_target_function()
  local y = evaluate_function(f, x)

  return x, y, f
end

-- returns a linear function extracted from the given weights vector
compute_hypothesis = function(w)
  local a = -(w[2] / w[3])
  local b = -(w[1]) / w[3]

  return function (x)
    return a * x + b
  end
end


hypotheses = {}
miss_fractions = torch.Tensor(num_trials)


run_once = function (num_examples)
  x, y, f = generate_training_data(num_examples)

  -- compute linear regression
  pseudo_inverse = torch.inverse(x:t() * x) * x:t()
  w = pseudo_inverse * y
  g = compute_hypothesis(w)

  -- evaluate hypothesis
  y_g = evaluate_function(g, x)
  in_sample_error = math.abs(torch.sum(y - y_g)) / 2 / num_examples

  return g, in_sample_error
end

for i = 1, num_trials do
  hypotheses[i], miss_fractions[i] = run_once(num_examples)
end

print(torch.mean(miss_fractions))


