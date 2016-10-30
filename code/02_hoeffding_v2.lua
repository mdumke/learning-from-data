#!/home/tias/torch/install/bin/th

--[[
  hoeffding v2

  re-implementation of an experiment to estimate probabilities
  of different observations on coin-tosses
--]]

require('helpers')
math.randomseed(os.time())

num_trials = 100000
num_coins = 1000
num_tosses = 10

first_coin = torch.Tensor(num_trials)
random_coin = torch.Tensor(num_trials)
min_coin = torch.Tensor(num_trials)

for i = 1, num_trials do
  -- throw n fair coins each 10 times and count the fraction of heads
  --coin_flips = torch.floor(torch.rand(num_coins, num_tosses) * 2)
  --faster implementation:
  coin_flips = torch.Tensor(num_coins, num_tosses):apply(
    function () return math.random(0, 1) end)

  head_counts = torch.sum(coin_flips, 2) / num_tosses

  -- count the number of heads for different coins
  first_coin[i] = head_counts[1][1]
  random_coin[i] = head_counts[math.random(num_coins)][1]
  min_coin[i] = torch.min(head_counts)
end

print("nu_1: " .. round(torch.mean(first_coin), 3) ..
  "\nnu_rand: " .. round(torch.mean(random_coin), 3) ..
  "\nnu_min: " .. round(torch.mean(min_coin), 3))

