#!/home/tias/torch/install/bin/th

--[[
  plots of different functions for a generalization error
--]]

require('gnuplot')

-- returns an estimate for the number of hypotheses from H on n data points
growth_function = function (n, d_vc)
  if n <= 50 then
    return 2^n
  else
    return n^d_vc
  end
end

-- returns the vc-bound for a dataset of n points
original_vc_bound = function (n, delta, d_vc)
  return math.sqrt(8 / n * math.log(4 * growth_function(2 * n, d_vc) / delta))
end

-- returns the rademacher-penalty-bound for a dataset of n points
rademacher_bound = function (n, delta, d_vc)
  return math.sqrt((2 * math.log(2 * n * growth_function(n, d_vc))) / n) +
         math.sqrt(2 / n * math.log(1 / delta)) +
         1 / n
end

-- returns the parrondo and van den broek bound
parrondo = function (n, delta, epsilon, d_vc)
  return math.sqrt(1 / n * (2 * epsilon + math.log((6 * growth_function(2 * n, d_vc)) / n)))
end

-- returns the devroye bound for a dataset of n points
devroye = function (n, delta, epsilon, d_vc)
  return math.sqrt(1 / (2 * n) * (4 * epsilon * (1 + epsilon) + math.log((4 * growth_function(n^2, d_vc)) / delta)))
end

-- alternative computation for the devroye bound
devroye2 = function (n, delta, epsilon, d_vc)
  return math.sqrt(1 / (2 * n) * (4 * epsilon * (1 + epsilon) + 2 * math.log(2 / math.sqrt(delta) * n^d_vc)))
end


d_vc = 50
delta = 0.05
epsilon = 1
sample_size = 20

-- collect boundary-data into a tensor
points = torch.Tensor(sample_size, 5)
points:t()[1] = torch.linspace(1, sample_size, sample_size)

for i = 1, sample_size do
  points[i][2] = original_vc_bound(points[i][1], delta, d_vc)
  points[i][3] = rademacher_bound(points[i][1], delta, d_vc)
  points[i][4] = parrondo(points[i][1], delta, epsilon, d_vc)
  points[i][5] = devroye(points[i][1], delta, epsilon, d_vc)
end

gnuplot.figure(1)
gnuplot.axis({0, sample_size, 0, 8})
gnuplot.plot(
  {'original bound', points:t()[1], points:t()[2], '-'},
  {'rademacher bound', points:t()[1], points:t()[3], '-'},
  {'parrondo bound', points:t()[1], points:t()[4], '-'},
  {'devroye bound', points:t()[1], points:t()[5], '-'})
