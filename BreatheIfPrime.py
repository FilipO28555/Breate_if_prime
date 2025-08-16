import torch
import cv2
from sympy import isprime
from sympy import sieve
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu" # cpu is just faster probably

# create NxM tensor:
# the first row contains the numbers from 1q to M,
# the second row contains the numbers from 1 to M,
def create_tensor(N, M):
    return torch.tensor([[j for j in range(1, M + 1)] for i in range(N)], device=device)

def create_add_tensor(N, M):
    return torch.tensor([[N-1-i for j in range(1, M + 1)] for i in range(N)], device=device)   

def plot_tensor(tensor, name='Tensor Visualization', size=3):
    tensor_np = torch.log10(tensor+1).cpu().numpy()
    tensor_np = (tensor_np - tensor_np.min()) / (tensor_np.max() - tensor_np.min()) * 255
    tensor_np = tensor_np.astype('uint8')
    # enlarge the tensor for better visualization
    tensor_np = cv2.resize(tensor_np, (size * tensor_np.shape[1], size * tensor_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    cv2.imshow(name, tensor_np)

# plot_tensor(create_tensor(3, 5))

N = 100 # breath girth
M = 100 # lung capacity
start_values = create_tensor(N, M)
add_tensor = create_add_tensor(N, M)
Air = create_tensor(N, M)
alive_mask = Air > 0

steps_in = torch.zeros_like(Air, device=device)

def step(breath):
    global Air, alive_mask, start_values, add_tensor, steps_in
    # if breath we add to the Air: the add tensor and cap the values to the start values
    steps_in[alive_mask] += 1
    if breath:
        Air[alive_mask] += add_tensor[alive_mask]
        Air = torch.clamp(Air, max=start_values)
    else:
        Air[alive_mask] -= 1
        alive_mask = Air > 0

# different method - not by steps but by gaps
maxPrime = 20971520
maxGap = 0
while maxGap < M:
    maxPrime *= 2
    primes = np.array(list(sieve.primerange(2,maxPrime)))
    gaps = primes[1:]-primes[:-1]
    maxGap = gaps.max()
    print(f"maxGap: {maxGap} up to {maxPrime}")      
def gap_step(gap_len):
    global Air, alive_mask, start_values, add_tensor, steps_in
    
    Air[alive_mask] -= gap_len
    # print(Air)
    steps_in[alive_mask] += gap_len
    # print(steps_in)
    
    mask = Air<0
    steps_in += Air*mask
    Air -= Air*mask
    # Air = torch.clamp(Air,min=0)
    # print(steps_in)
    
    alive_mask = Air > 0
    
    Air[alive_mask] += add_tensor[alive_mask]
    Air = torch.clamp(Air, max=start_values)

# gap step loop
i=0
previously_alive = alive_mask.sum()
y = [previously_alive]
while alive_mask.any():
    while previously_alive == alive_mask.sum():
        gap_step(gaps[i])
        i += 1
    print(f"Step {i + 1}:")
    previously_alive = alive_mask.sum()
    y.append(previously_alive)
    
    plot_tensor(Air, "Air left in lungs")  # visualize the Air tensor after each step
    plot_tensor(steps_in, 'seconds surviving')  # visualize the steps_in tensor after each step
    cv2.waitKey(1)
    # uncomment to go step by step with q
    # while True:
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    
exit()


# Old loop
i=0
previously_alive = alive_mask.sum()
y = [previously_alive]
while alive_mask.any():
    while previously_alive == alive_mask.sum():
        i += 1
        step(isprime(i))  # breathe in on prime steps, breathe out on non-prime steps
    print(f"Step {i + 1}:")
    previously_alive = alive_mask.sum()
    y.append(previously_alive)
    
    plot_tensor(Air)  # visualize the Air tensor after each step
    plot_tensor(steps_in, f'seconds surviving')  # visualize the steps_in tensor after each step
    cv2.waitKey(1)  # wait for a key press to update the window


# plot and wait for closing the windows
print(f"lung capacity: {M}, breath girth: {N}, seconds alive: {steps_in[0,M-1]}")

import matplotlib.pyplot as plt

plt.plot(y)
plt.show()

while True:
    plot_tensor(Air)  # visualize the final Air tensor
    plot_tensor(steps_in, f'finished after {i} steps')  # visualize the final steps_in tensor
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


