#!/home/tias/torch/install/bin/th

--[[
  run linear regression under a non-linear transformation and
  apply regularization via weight decay
--]]

require('helpers')

-- returns the nx8-tensor after the required transformation of x
non_linear_transform = function (x)
  local length = x:size()[1]
  local x_transform = torch.Tensor(length, 8)

  for i = 1, length do
    x_transform[i][1] = 1
    x_transform[i][2] = x[i][1]
    x_transform[i][3] = x[i][2]
    x_transform[i][4] = x[i][1]^2
    x_transform[i][5] = x[i][2]^2
    x_transform[i][6] = x[i][1] * x[i][2]
    x_transform[i][7] = math.abs(x[i][1] - x[i][2])
    x_transform[i][8] = math.abs(x[i][1] + x[i][2])
  end

  return x_transform
end

-- returns the 8-dim weights vector after running linear regression
linear_regression = function (x, y)
  local pseudo_inverse = torch.inverse(x:t() * x) * x:t()
  return pseudo_inverse * y
end

-- returns the 8-dim weigts vector for lin-reg with regularization
regularized_lin_reg = function(x, y, lambda)
  local pseudo_inverse = torch.inverse(x:t() * x + lambda * torch.eye(8)) * x:t()
  return pseudo_inverse * y
end

-- returns the misclassification-error of the model w on the given dataset
compute_error = function (w, x, y)
  local predictions = torch.sign(w:t() * x:t())
  local num_misclassified = y:ne(predictions):sum()
  return num_misclassified / x:size()[1]
end

-- prepare data
x_train, y_train = read_data_from_file('in.dta')
x_train = non_linear_transform(x_train)
x_test, y_test = read_data_from_file('out.dta')
x_test = non_linear_transform(x_test)

-- run linear regression
w_lin = linear_regression(x_train, y_train)
print("lin e_in: ", round(compute_error(w_lin, x_train, y_train), 2))
print("lin e_out: ", round(compute_error(w_lin, x_test, y_test), 2))

-- run linear regression with regularization
for k = -3, 3 do
  print("\nk = " .. k)
  lambda = 10^k
  w_reg = regularized_lin_reg(x_train, y_train, lambda)
  print("  reg e_in: ", round(compute_error(w_reg, x_train, y_train), 2))
  print("  reg e_out: ", round(compute_error(w_reg, x_test, y_test), 2))
end
