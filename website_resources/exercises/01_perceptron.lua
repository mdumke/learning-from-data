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

-- plots the given functions in the range [-1, 1]
function plot(f)
  -- help with gnuplot: github.com/torch/gnuplot
  local x = torch.linspace(-1, 1)
  local y = torch.linspace(-1, 1)
  y:apply(f)

  gnuplot.figure(1)
  gnuplot.axis({-1, 1, -1, 1})
  gnuplot.xlabel('x')
  gnuplot.ylabel('y')
  gnuplot.title('title')

  gnuplot.plot(
   {'f(x)'  , x, y, '+'}
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
  p1 = random_point()
  p2 = random_point()

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

  return xs, y
end

f = generate_target_function()
n = 4
xs, y = generate_training_data(f, n)

print(y)


plot(f)





-- 1. generate target function f
-- 2. pick n points at random and evaluate f on them
-- 3. display the situation...

print(generate_target_function()(1));

