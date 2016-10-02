-- rounds the given number to the specified decimal place
round = function (n, precision)
  return math.floor((n * 10^precision) + 0.5) / 10^precision
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

