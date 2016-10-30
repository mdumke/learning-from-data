#!/home/tias/torch/install/bin/th

--[[
  perform gradient descent on a given error surface
--]]

require('helpers')

ETA = 0.1

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

-- returns the partial derivatives at the given point
compute_gradient = function (u, v)
  return partial_u(u, v), partial_v(u, v)
end

-- returns the new weights after one update step along the (negative) gradient
update_weights = function (w1, w2)
  du, dv = compute_gradient(w1, w2)
  return w1 - ETA * du, w2 - ETA * dv
end

w1, w2 = 1, 1
iteration = 0

repeat
  w1, w2 = update_weights(w1, w2)
  iteration = iteration + 1
until (compute_error(w1, w2) < 10e-14)

print("\niterations: " .. iteration)
print("remaining error: " .. compute_error(w1, w2))
print("(w1, w2) = (" .. round(w1, 3) .. ", " .. round(w2, 3) .. ")")

