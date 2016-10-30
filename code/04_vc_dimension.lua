--[[
  computes versions of a given growth function depending on a factor q
--]]

-- computes the factorial of a given integer
factorial = function (n)
  if n <= 1 then
    return 1
  else
    return n * factorial(n - 1)
  end
end

-- returns the binomial coefficient
choose = function (n, k)
  if (k > n) then
    return 0
  else
    return factorial(n) / (factorial(n - k) * factorial(k))
  end
end

-- computes the growth function depending on q
m_h = function (n, q)
  if n <= 1 then
    return 2
  else
    return 2 * m_h(n - 1, q) - choose(n - 1, q)
  end
end

-- collect data on how many points can be shattered for each q
n = 20
data = torch.Tensor(n, 8)

for i = 1, n do
  data[i][1] = i
  data[i][2] = m_h(i, 1)
  data[i][3] = m_h(i, 2)
  data[i][4] = m_h(i, 3)
  data[i][5] = m_h(i, 4)
  data[i][6] = m_h(i, 5)
  data[i][7] = m_h(i, 6)
  data[i][8] = 2^i
end

print(data)
