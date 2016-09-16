#!/home/tias/torch/install/bin/th

--[[
  Helper functions for computing some values
  of potential growth functions
--]]

require('gnuplot')

-- displays 1 to n datapoints of the given function
plot = function (f, n)
  x = torch.linspace(1, n, n)
  y = torch.linspace(1, n, n):apply(f)
  gnuplot.figure(1)
  gnuplot.plot({'f', x, y, '-'})
end

-- returns the factorial of the given integer
factorial = function (n)
  if n <= 1 then
    return 1
  else
    return n * factorial(n - 1)
  end
end

-- returns the combinatorial quantity n choose k
choose = function (n, k)
  return factorial(n) / (factorial(n - k) * factorial(k))
end



----------- CANDIDATE GROWTH FUNCTIONS ----------------

-- returns the first function's result for n data points
first = function (n)
  return n + 1
end

-- returns the second function's result for n data points
second = function (n)
  return 1 + n + choose(n, 2)
end

-- returns the third function's result for n data points
third = function (n)
  local result = 0

  for i = 1, math.floor(math.sqrt(n)) do
    result = result + choose(n, i)
  end

  return result
end

-- returns the fourth function' result for n data points
fourth = function (n)
  return 2^(math.floor(n / 2))
end

n = tonumber(arg[1])
plot(fourth, n)
