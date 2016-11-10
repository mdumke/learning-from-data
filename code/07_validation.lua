#!/home/tias/torch/install/bin/th

--[[
  Performs linear regression without regularization and evaluates
  the performance using different sizes of validation sets
--]]

require('helpers')

-- returns x under k-dimensional non-linear transform
nonlinear_transform = function (x, k)
  local num_rows = x:size()[1]
  local x_transform = torch.Tensor(8, num_rows)

  x_transform[1] = 1
  x_transform[2] = x:t()[1]
  x_transform[3] = x:t()[2]
  x_transform[4] = torch.pow(x:t()[1], 2)
  x_transform[5] = torch.pow(x:t()[2], 2)
  x_transform[6] = x:t()[1] * x:t()[2]
  x_transform[7] = torch.abs(x:t()[1] - x:t()[2])
  x_transform[8] = torch.abs(x:t()[1] + x:t()[2])

  return x_transform:t()[{{}, {1, k + 1}}]
end

-- returns the weights-vector after running linear regression
linear_regression = function (x, y)
  local pseudo_inverse = torch.inverse(x:t() * x) * x:t()
  return pseudo_inverse * y
end

-- returns the model's estimated squared-error on the given test/validation-set
compute_error = function (w, x_test, y_test)
  local predictions = torch.sign(x_test * w)
  return y_test:ne(predictions):sum() / x_test:size()[1]
end


-- read in data and split into training and validation set
x, y = read_data_from_file('in.dta')

x_train = x:sub(1, 25)
y_train = y:sub(1, 25)

x_validation = x:sub(26, 35)
y_validation = y:sub(26, 35)

x_test, y_test = read_data_from_file('out.dta')

-- compute validation-error for different sizes of the validation set
for k = 3, 7 do
  x_train_transform = nonlinear_transform(x_train, k)
  x_validation_transform = nonlinear_transform(x_validation, k)
  x_test_transform = nonlinear_transform(x_test, k)

  w = linear_regression(x_train_transform, y_train)

  e_val = compute_error(w, x_validation_transform, y_validation)
  e_out = compute_error(w, x_test_transform, y_test)

  print("k: " .. k .. ", e_val: ", e_val, ", e_out: ", e_out)
end

