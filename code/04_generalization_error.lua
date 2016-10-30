#!/home/tias/torch/install/bin/th

--[[
  Some helper functions to compute generalization bounds
--]]

require('gnuplot')
math.randomseed(os.time())


------ PROBLEM 1 ------

-- returns m_h(n), the growth function for n, dvc is the VC-dimension
mh = function (n)
  if (n > dvc) then
    return n^dvc
  else
    return 2^n
  end
end

-- returns the generalization error via the VC-inequality
error_probability = function (n, epsilon, dvc)
  return 4 * mh(2 * n, dvc) * math.exp(-1 / 8 * epsilon^2 * n)
end

-- prints probabilities to answer the first problem
problem_1 = function ()
  epsilon = 0.05
  dvc = 10

  candidates = {
    400000,
    420000,
    440000,
    450000,
    460000,
    480000
  }

  for i = 1, #candidates do
    print(candidates[i] .. ": " .. error_probability(candidates[i], epsilon, dvc))
  end
end


------ PROBLEM 2 ------

-- returns the original vc-bound for n examples
vc_bound = function (n)
  return math.sqrt(8 / n * math.log(4 * mh(2 * n) / delta))
end

-- returns the rademacher-penalty-bound for a given n
rp_bound = function (n)
  return math.sqrt(2 * math.log(2 * n * mh(n)) / n) + math.sqrt(2 / n * math.log(1 / delta)) + 1 / n
end

-- returns the parrondo and van den broek bound
pv_bound = function (n)
  return math.sqrt(1 / n * (2 * epsilon + math.log(6 * mh(2 * n) / delta)))
end

-- returns the devroye bound
de_bound = function (n)
  return math.sqrt(1 / (2 * n) * (4 * epsilon * (1 + epsilon) + math.log(4 * mh(n^2) / delta)))
end

problem_2 = function ()
  delta = 0.05
  dvc = 50
  epsilon = 0.002

  points = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 16, 20}
  --points = {900, 1000, 1200}
  x = torch.Tensor(points)

  y_vc = torch.Tensor(points):apply(vc_bound)
  y_rp = torch.Tensor(points):apply(rp_bound)
  y_pv = torch.Tensor(points):apply(pv_bound)
  y_de = torch.Tensor(points):apply(de_bound)

  gnuplot.figure(1)
  gnuplot.plot(
    {'vc', x, y_vc, '-'},
    {'rademacher', x, y_rp, '-'},
    {'parrondo', x, y_pv, '-'},
    {'devroye', x, y_de, '-'})
end

problem_2()
