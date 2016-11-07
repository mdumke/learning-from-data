-- rounds the given number to the specified decimal place
round = function (n, precision)
  return math.floor((n * 10^precision) + 0.5) / 10^precision
end

-- returns the sign of the number, but not 0
sign = function (n)
  if n > 0 then return 1 else return -1 end
end

-- returns the number of lines in the file
num_lines = function (filename)
  local handle = io.popen("wc -l < " .. filename)
  local result = handle:read("*a")
  handle:close()
  return tonumber(result)
end

-- reads in data from the given file
read_data_from_file = function (filename)
  local length = num_lines(filename)
  local x = torch.Tensor(length, 2)
  local y = torch.Tensor(length, 1)

  local line_number = 1

  for line in io.lines(filename) do
    local entry = 1

    for number in line:gmatch("%S+") do
      if entry  == 1 then
        x[line_number][1] = tonumber(number)
      elseif entry == 2 then
        x[line_number][2] = tonumber(number)
      else
        y[line_number][1] = tonumber(number)
      end

      entry = entry + 1;
    end

    line_number = line_number + 1
  end

  return x, y
end

-- randomizes the rows of the given 2-dim tensors
random_permutation = function (x, y)
  local num_rows = x:size()[1]
  local idx = torch.randperm(num_rows)

  local new_x = torch.Tensor(num_rows, 3)
  local new_y = torch.Tensor(num_rows, 1)

  for i = 1, num_rows do
    new_x[i] = x[idx[i]]
    new_y[i] = y[idx[i]]
  end

  return new_x, new_y
end
