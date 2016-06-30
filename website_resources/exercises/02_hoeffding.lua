#!/home/tias/torch/install/bin/th

--[[
  Simulation of 10 tosses of 1000 coins, collecting information
  about the probabilities of Heads under certain cirumstances
--]]

math.randomseed(os.time())

-- configure simulation
num_coins = 1000
num_tosses = 10
num_trials = 10000

-- runs the ith trial
single_trial = function (num_coins, num_tosses, current_trial)
  -- initialize mxn matrix, randomly with 0s and 1s to simulate m coin tosses
  local tosses = torch.Tensor(num_coins, num_tosses):apply(
    function () return math.random(0, 1) end)

  -- compute the fractions of heads for each coin and extract target values
  local fractions = torch.sum(tosses, 2) / num_tosses

  c_1[current_trial] = fractions[1][1]
  c_rand[current_trial] = fractions[math.random(1, num_coins)][1]
  c_min[current_trial] = torch.min(fractions)
end

-- runs the simulation
run_simulation = function (num_coins, num_tosses, num_trials)
  for i = 1, num_trials do
    single_trial(num_coins, num_tosses, i)
  end
end

-- prints some statistics about the simulation
report_results = function ()
  print(' ')
  print('computing probability distributions for throwing ' .. num_coins ..
    ' coins, each ' .. num_tosses .. ' times.')
  print(' ')
  print('first coin:          E[# Heads] = ' .. torch.mean(c_1))
  print('random coin:         E[# Heads] = ' .. torch.mean(c_rand))
  print('minimun #Heads coin: E[# Heads] = ' .. torch.mean(c_min))
end

-- initialize results-containers
c_1 = torch.Tensor(num_trials)
c_rand = torch.Tensor(num_trials)
c_min = torch.Tensor(num_trials)

run_simulation(num_coins, num_tosses, num_trials)
report_results()

