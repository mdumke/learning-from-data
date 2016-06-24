#!/home/tias/torch/install/bin/th

--[[

Learning From Data, Exercise 1
Perceptron Learning Algorithm

June 2016

This is a simulation of the perceptron learning algorithm running
on a set of generated datapoints.

--]]

-- some preparations
require('gnuplot')
math.randomseed(os.time())

-- plots the given points and function in the range [-1, 1]
function plot(xs, y, f)
  -- help with gnuplot: github.com/torch/gnuplot
  -- display the target function
  f_x = torch.linspace(-1.5, 1.5, 10)
  f_y = torch.linspace(-1.5, 1.5, 10)
  f_y:apply(f)

  -- find indices of positive and negative points
  pos_indices = {}
  neg_indices = {}

  for i = 1, y:size()[1] do
    if y[i] == 1 then
      pos_indices[#pos_indices + 1] = i
    else
      neg_indices[#neg_indices + 1] = i
    end
  end

  -- split the training data according to label
  pos_xs = torch.Tensor(#pos_indices, 2)
  neg_xs = torch.Tensor(#neg_indices, 2)

  for i = 1, #pos_indices do
    pos_xs[i] = xs:t()[pos_indices[i]]
  end

  for i = 1, #neg_indices do
    neg_xs[i] = xs:t()[neg_indices[i]]
  end

  gnuplot.figure(1)
  gnuplot.axis({-1.5, 1.5, -1.5, 1.5})
  gnuplot.xlabel('x')
  gnuplot.ylabel('y')
  gnuplot.title('title')

  gnuplot.plot(
   {'pos', pos_xs:t()[1], pos_xs:t()[2], '+'},
   {'neg', neg_xs:t()[1], neg_xs:t()[2], '+'},
   {'f(x)', f_x, f_y, '-'}
  )
end

-- returns a random value within [-1, 1]
function random_coordinate()
  return math.random() * 2 - 1
end

-- returns a random point within [-1, 1]x[-1, 1]
function random_point()
  return {random_coordinate(), random_coordinate()}
end

-- returns the (random) linear target function
function generate_target_function()
  -- select two points that do not generate a vertical line
  epsilon = 0.001

  repeat
    p1 = random_point()
    p2 = random_point()
  until math.abs(p1[1] - p2[1]) > epsilon

  slope = (p2[2] - p1[2]) / (p2[1] - p1[1])

  return function (x)
    return slope * x - slope * p1[1] + p1[2]
  end
end

-- returns +/-1 if the value is more/less than the given target function
function evaluate_target_function(f, point)
  if point[2] >= f(point[1]) then
    return 1
  else
    return -1
  end
end

-- generates n training points and returns tensor xs and y
function generate_training_data(f, n)
  xs = torch.Tensor(n, 2):apply(function () return random_coordinate() end)
  y = torch.Tensor(n)

  for i = 1, n do
    y[i] = evaluate_target_function(f, xs[i])
  end

  return xs:t(), y
end



-- generate target function f
f = generate_target_function()

-- pick n points at random and evaluate f on them
n = 4
xs, y = generate_training_data(f, n)

plot(xs, y, f)

-- intialize weights vector
w = torch.Tensor(2, 1):fill(0)

has_converged = false
steps = 0

while (not has_converged) do
  -- compute predictions
  predictions = torch.sign(w:t() * xs)

  -- collect indices of misclassified points
  misclassified = {}

  for i = 1, n do
    if predictions[1][i] ~= y[i] then
      misclassified[#misclassified + 1] = i
    end
  end

  if #misclassified == 0 then
    has_converged = true
  else
    -- pick a misclassified point at random
    rand_idx = misclassified[math.random(1, #misclassified)]
    rand_x = xs:t()[rand_idx]
    rand_y = y[rand_idx]

    -- apply the transformation to update w
    w = w + rand_y * rand_x

    steps = steps + 1
  end
end

-- determine the final hypothesis
g = function (x)
  return w[2][1] * x + w[1][1]
end

print(steps)
--plot(g)










--plot(f)

-- 3. display the situation...

-- 4. learn g from the training data

-- 5. display final situation...

-- 6. make this a simulation






