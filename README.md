# Deep RL Bootscamp (Berkeley CA)

Link to course: [https://sites.google.com/view/deep-rl-bootcamp/home](https://sites.google.com/view/deep-rl-bootcamp/home)

## Lab 04

This Repo contains the lab 04: [https://sites.google.com/view/deep-rl-bootcamp/labs](https://sites.google.com/view/deep-rl-bootcamp/labs)

### Results

#### Task 3.3 & 3.4: Accumulating Policy Gradient

Run Point-V0 with accumulating policy gradient:
```
./docker_run.sh simplepg/main.py Point-v0 --use-baseline False
```

Result:
```
Gradient check passed!
[2020-10-14 04:57:51,729] Making new env: Point-v0
Test for __main__.compute_update passed!
Iteration: 0 AverageReturn: -41.86 |theta|_2: 0.20
Iteration: 1 AverageReturn: -41.96 |theta|_2: 0.17
Iteration: 2 AverageReturn: -40.24 |theta|_2: 0.23
...
Iteration: 97 AverageReturn: -25.78 |theta|_2: 2.48
Iteration: 98 AverageReturn: -25.43 |theta|_2: 2.50
Iteration: 99 AverageReturn: -24.65 |theta|_2: 2.52
```

#### Task 3.5: Time-Dependent Baseline

Run Point-V0 with time-dependet baseline:
```
./docker_run.sh simplepg/main.py Point-v0 --render True
```

Result:
```
Gradient check passed!
[2020-10-14 05:41:34,173] Making new env: Point-v0
Test for __main__.compute_update passed!
Test for __main__.compute_baselines passed!
Iteration: 0 AverageReturn: -41.86 |theta|_2: 0.20
Iteration: 1 AverageReturn: -41.96 |theta|_2: 0.25
Iteration: 2 AverageReturn: -40.19 |theta|_2: 0.27
...
Iteration: 97 AverageReturn: -20.18 |theta|_2: 5.49
Iteration: 98 AverageReturn: -19.75 |theta|_2: 5.54
Iteration: 99 AverageReturn: -19.77 |theta|_2: 5.57
```

#### Task 3.6

Run Cart Pole with time-dependent baseline:
```
./docker_run.sh simplepg/main.py CartPole-v0 --render True
```

Result:
```
Gradient check passed!
[2020-10-14 06:35:59,233] Making new env: CartPole-v0
Test for __main__.compute_update passed!
Test for __main__.compute_baselines passed!
Iteration: 0 AverageReturn: 18.52 Entropy: 0.69 Perplexity: 1.99 |theta|_2: 0.20
Iteration: 1 AverageReturn: 20.19 Entropy: 0.69 Perplexity: 2.00 |theta|_2: 0.18
Iteration: 2 AverageReturn: 23.86 Entropy: 0.69 Perplexity: 1.99 |theta|_2: 0.22
...
Iteration: 97 AverageReturn: 196.55 Entropy: 0.56 Perplexity: 1.75 |theta|_2: 3.81
Iteration: 98 AverageReturn: 170.42 Entropy: 0.55 Perplexity: 1.73 |theta|_2: 3.84
Iteration: 99 AverageReturn: 198.64 Entropy: 0.56 Perplexity: 1.75 |theta|_2: 3.88
```

#### Task 3.7: Hyperparameter Tuning
In this chapter we will not do any actual tuning but try to understand the different parameters.

Example run of Cart Pole with different hyperparameters:
```
./docker_run.sh simplepg/main.py CartPole-v0 --batch_size b --discount d --learning_rate l
```

Result Point-v0:

| Settings              | Average Return | Theta 2|
|-----------------------|----------------|--------|
|b=1000, d=0.95, a=0.01 | -34.11          | 0.74   |
|b=2048, d=0.95, a=0.01 | -33.64          | 0.84   |
|b=1000, d=0.97, a=0.01 | -34.21          | 0.73   |
|b=1000, d=0.93, a=0.01 | -34.05          | 0.75   |
|b=1000, d=0.95, a=0.02 | -28.66          | 1.46   |
|b=1000, d=0.95, a=0.03 | -25.99         | 2.13   |
|b=1000, d=0.95, a=0.04 | -23.95         | 2.80   |
|b=1000, d=0.95, a=0.05 | -22.35         | 3.36   |
|b=1000, d=0.95, a=0.5 | -18.38         | 9.87   |
|b=1000, d=0.95, a=0.005 | -37.84         | 0.38   |


Result Cartpole-v0:

| Settings              | Average Return | Entropy | Perplexity | Theta 2|
|-----------------------|----------------|---------|------------|--------|
|b=1000, d=0.95, a=0.01 | 37.96          | 0.65    | 1.92       | 0.79   |
|b=2048, d=0.95, a=0.01 | 43.89          | 0.65    | 1.91       | 0.81   |
|b=1000, d=0.97, a=0.01 | 36.64         | 0.65     | 1.92       | 0.79   |
|b=1000, d=0.93, a=0.01 | 53.11         | 0.65     | 1.92       | 0.77   |
|b=1000, d=0.95, a=0.02 | 77.93          | 0.63    | 1.88       | 1.46   |
|b=1000, d=0.95, a=0.03 | 100.90          | 0.61    | 1.83       | 1.99   |
|b=1000, d=0.95, a=0.04 | 126.38          | 0.60    | 1.82       | 2.39   |
|b=1000, d=0.95, a=0.05 | 159.00         | 0.59    | 1.80       | 2.67   |
|b=1000, d=0.95, a=0.5 | 200 (max)         | 0.49    | 1.63       | 6.40   |
|b=1000, d=0.95, a=0.005 | 33.16          | 0.68    | 1.98       | 0.36   |

As the two tables illustrate, both algorithms can be optimized especially by increasing the learning rate. Also the mini-batch size and the discount factor have an influence on the average return, but less than the learning rate. For better performance, a random search should be performed with different parameters and whereby the learning rate is higher.


#### Task 3.8: Natural Gradient

Run Cart Pole natural gradient:
```
./docker_run.sh simplepg/main.py CartPole-v0 --natural True
```

Result:
```
Gradient check passed!
[2020-10-14 13:11:27,635] Making new env: CartPole-v0
Test for __main__.compute_update passed!
Test for __main__.compute_baselines passed!
Test for __main__.compute_fisher_matrix passed!
Test for __main__.compute_natural_gradient passed!
Test for __main__.compute_step_size passed!
Iteration: 0 AverageReturn: 18.52 Entropy: 0.69 Perplexity: 2.00 |theta|_2: 0.83
Iteration: 1 AverageReturn: 23.05 Entropy: 0.68 Perplexity: 1.98 |theta|_2: 1.49
Iteration: 2 AverageReturn: 32.08 Entropy: 0.66 Perplexity: 1.93 |theta|_2: 2.66
...
Iteration: 97 AverageReturn: 200.00 Entropy: 0.56 Perplexity: 1.76 |theta|_2: 20.77
Iteration: 98 AverageReturn: 200.00 Entropy: 0.53 Perplexity: 1.69 |theta|_2: 25.92
Iteration: 99 AverageReturn: 200.00 Entropy: 0.51 Perplexity: 1.67 |theta|_2: 20.52

```

#### Task 4: Advanced Policy Gradient

Run Cartpole policy gradient:
```
./docker_run.sh experiments/run_pg_cartpole.py
```

Results:

| Average Return | Surrogate Loss |
|----------------|----------------|
| <img alt="Average Return Baseline" src="/plots/APG_avg_return_cartpole.png" width="750"> | <img alt="Average Return Baseline" src="/plots/APG_Surr_Loss_cartpole.png" width="750"> |

The graphs show that the agent has learned a wrong move after about 70 steps. As a result, the average return has collapsed and the surrogate loss has increased. However, the agent was able to correct this mistake quickly.


#### Task 5: Trust Region Policy Optimization (TRPO)


Run discrete policy on CartPole-V0:
```
./docker_run.sh experiments/run_trpo_cartpole.py
```

Result:
<img alt="Average Return Baseline" src="/plots/TRPO_avg_return_cartpole.png" width="750">


Run discrete policy on Pendulum-V0:
```
./docker_run.sh experiments/run_trpo_pendulum.py
```

Result:
<img alt="Average Return Baseline" src="/plots/TRPO_avg_return_pendulum.png" width="750">


Run discrete policy on RoboschoolHalfCheetah-v1:
```
./docker_run.sh experiments/run_trpo_half_cheetah.py
```

Result:
<img alt="Average Return Baseline" src="/plots/TRPO_avg_return_half_cheetah.png" width="750">


#### Task 6: Advanced Actor Critic (A2C)

Run A2C Pong with warm start:
```
./docker_run.sh experiments/run_a2c_pong_warm_start.py
```

Result:
<img alt="Average Return Baseline" src="/plots/A2C_avg_return_pong.png" width="750">

Run A2C Breakout:
```
./docker_run.sh experiments/run_a2c_breakout.py
```

Result:
<img alt="Average Return Baseline" src="/plots/A2C_avg_return_breakout.png" width="750">




## Other labs
- [https://github.com/sagerpascal/deep-rl-bootcamp-lab01](https://github.com/sagerpascal/deep-rl-bootcamp-lab01)
- [https://github.com/sagerpascal/deep-rl-bootcamp-lab2](https://github.com/sagerpascal/deep-rl-bootcamp-lab2)
- [https://github.com/sagerpascal/deep-rl-bootcamp-lab3](https://github.com/sagerpascal/deep-rl-bootcamp-lab3)


