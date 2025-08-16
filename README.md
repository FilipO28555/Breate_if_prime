# Prime Breathing Simulation  
A PyTorch-based simulation that visualizes breathing patterns based on prime numbers and prime gaps.  
Please have fun creating new ways to visualize this and finding new patterns in primes.
  
## Description  
  
This project simulates a breathing mechanism where:
  - each entity has two numbers associated to it:
     * N - how much units of air it breaths in when the timer is prime. Represented on Y axes
     * M - maximum units of air it can hold. Represented on X axes.
  - there are two algorithms to update the state of entities:
    1) per number: every step is updating all entities by 1 timer number. (Old)
    2) per gap: every step is updatin each entities by the gap size between consecutive primes. (New)
  
  
## Performance  
  
- 100x100 runs in approximately 30 seconds on my CPU. GPU actually doesn't help for such small tensors but you can check it more thoroughly.
  
## Requirements  
  
- PyTorch  
- OpenCV (cv2)  
- SymPy  
- NumPy  
- Matplotlib  
  
## Installation  
  
Install the required dependencies:  
  
```bash  
pip install -r requirements.txt  
``` 
