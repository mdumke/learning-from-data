#!/home/tias/torch/install/bin/th

--[[
  Regularization with Weight Decay

  Start by performing a simple linear regression on the data provided
  using a non-linear transformation
--]]

require('helpers')

-- returns the pseudo-inverse of a given matrix
pseudo_inverse = function (matrix)
  return torch.inverse(matrix:t() * matrix) * matrix:t()
end

-- returns the given x-data after the required transformation
non_linear_transform = function (x)
  local n = x:size()[1]
  local z = torch.Tensor(n, 8)

  for i = 1, n do
    z[i][1] = 1
    z[i][2] = x[i][1]
    z[i][3] = x[i][2]
    z[i][4] = x[i][1]^2
    z[i][5] = x[i][2]^2
    z[i][6] = x[i][1] * x[i][2]
    z[i][7] = math.abs(x[i][1] - x[i][2])
    z[i][8] = math.abs(x[i][1] + x[i][2])
  end

  return z
end

-- trains a model using a non-linear transformation
train_model = function (x, y)
  z = non_linear_transform(x)
  return pseudo_inverse(z) * y
end

-- returns the error of the given model w on the data
compute_error = function (w, x, y)
  local n = x:size()[1]
  return y:ne(torch.sign(non_linear_transform(x) * w)):sum() / n
end


------------ MAIN CONTROL ------------------

x, y = read_data_from_file('in.dta')
w = train_model(x, y)

e_in = compute_error(w, x, y)

x_test, y_test = read_data_from_file('out.dta')
e_out = compute_error(w, x_test, y_test)

print('\nlinear regression with non-linear-transformation')
print('  e_in: ' .. round(e_in, 3))
print('  e_out: ' .. round(e_out, 3) .. '\n')

