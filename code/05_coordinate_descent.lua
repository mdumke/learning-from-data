#!/home/tias/torch/install/bin/th

--[[
  performs coordinate descent in the (u, v)-space
--]]

ETA = 0.1

-- rounds the given number to the specified decimal place
round = function (n, precision)
  return math.floor((n * 10^precision) + 0.5) / 10^precision
end

-- returns the value of the error function at the given point
compute_error = function (u, v)
  return (u * math.exp(v) - 2 * v * math.exp(-u))^2
end

-- returns the function's partial derivative along u-coordinate
partial_u = function (u, v)
  return 2 * (u * math.exp(v) - 2 * v * math.exp(-u)) * (math.exp(v) + 2 * v * math.exp(-u))
end

-- returns the function's partial derivative along v-coordinate
partial_v = function (u, v)
  return 2 * (u * math.exp(v) - 2 * v * math.exp(-u)) * (u * math.exp(v) - 2 * math.exp(-u))
end


w1, w2 = 1, 1

for i = 1, 15 do
  w1 = w1 - ETA * partial_u(w1, w2)
  w2 = w2 - ETA * partial_v(w1, w2)
end

print("\ncoordinate descent, 15 iterations")
print("final error: " .. compute_error(w1, w2))

