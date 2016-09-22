#!/home/tias/torch/install/bin/th

--[[
  Regularization with Weight Decay

  Performs a linear regression on the data provided
  using a non-linear transformation and weight decay with
  lambda = 10^k (k provided via command line)

  usage: ./06_regularization.lua [optional int to influence weight decay]
--]]

require('helpers')

-- returns the amount or regularization as determined by cmd-line args, or 0
get_regularization_amount = function ()
  if arg[1] then
    return 10^tonumber(arg[1])
  else
    return 0
  end
end

-- returns the pseudo-inverse of a given matrix, using weight-decay
pseudo_inverse = function (matrix, lambda)
  local regularizer = lambda * torch.eye(matrix:size()[2])
  return torch.inverse(matrix:t() * matrix + regularizer) * matrix:t()
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
train_model = function (x, y, lambda)
  z = non_linear_transform(x)
  return pseudo_inverse(z, lambda) * y
end

-- returns the error of the given model w on the data
compute_error = function (w, x, y)
  local n = x:size()[1]
  return y:ne(torch.sign(non_linear_transform(x) * w)):sum() / n
end


------------ MAIN CONTROL ------------------

lambda = get_regularization_amount()

x, y = read_data_from_file('in.dta')
w = train_model(x, y, lambda)

e_in = compute_error(w, x, y)

x_test, y_test = read_data_from_file('out.dta')
e_out = compute_error(w, x_test, y_test)

print('\nlinear regression with weight decay (lambda = ' .. round(lambda, 3) .. '):')
print('  e_in: ' .. round(e_in, 3))
print('  e_out: ' .. round(e_out, 3) .. '\n')


