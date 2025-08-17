import torch
import cv2
from sympy import isprime
from sympy import sieve
import numpy as np
import matplotlib.pyplot as plt
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu" # cpu is just faster probably

# Choose method: 1 for prime step method, 2 for gap method
method_choice = 2
save_video = True

# Create output directory for frames (only if saving video)
output_dir = "output"
if save_video:
    air_frames_dir = os.path.join(output_dir, "air_frames")
    steps_frames_dir = os.path.join(output_dir, "steps_frames")
    os.makedirs(air_frames_dir, exist_ok=True)
    os.makedirs(steps_frames_dir, exist_ok=True)
    print(f"\nFrames will be saved to: {output_dir}")
else:
    print("\nVideo saving disabled - running visualization only")

# create NxM tensor:
# the first row contains the numbers from 1q to M,
# the second row contains the numbers from 1 to M,
def create_tensor(N, M):
    return torch.tensor([[j for j in range(1, M + 1)] for i in range(N)], device=device)

def create_add_tensor(N, M):
    return torch.tensor([[N-1-i for j in range(1, M + 1)] for i in range(N)], device=device)   

def plot_tensor(tensor, name='Tensor Visualization', size=3, save_path=None):
    tensor_np = torch.log10(tensor+1).cpu().numpy()
    tensor_np = (tensor_np - tensor_np.min()) / (tensor_np.max() - tensor_np.min()) * 255
    tensor_np = tensor_np.astype('uint8')
    # enlarge the tensor for better visualization
    tensor_np = cv2.resize(tensor_np, (size * tensor_np.shape[1], size * tensor_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    cv2.imshow(name, tensor_np)
    
    # Save frame if path is provided and saving is enabled
    if save_path and save_video:
        cv2.imwrite(save_path, tensor_np)
    
    return tensor_np


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
if method_choice == 2:  # Only calculate gaps if gap method is chosen
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
    steps_in[alive_mask] += gap_len
    
    mask = Air<0
    steps_in += Air*mask
    Air -= Air*mask
    
    alive_mask = Air > 0
    
    Air[alive_mask] += add_tensor[alive_mask]
    Air = torch.clamp(Air, max=start_values)

def run_simulation():
    """Run the chosen simulation method"""
    global Air, alive_mask, start_values, add_tensor, steps_in
    
    i = 0
    frame_count = 0
    previously_alive = alive_mask.sum()
    y = [previously_alive]
    
    print(f"\nStarting simulation with method {method_choice}")
    print("Press 'q' in any window to exit early")
    
    if method_choice == 1:
        # Prime step method
        print("Using Prime Step Method...")
        while alive_mask.any():
            while previously_alive == alive_mask.sum():
                i += 1
                step(isprime(i))  # breathe in on prime steps, breathe out on non-prime steps
            print(f"Step {i}:")
            previously_alive = alive_mask.sum()
            y.append(previously_alive)
            
            # Save frames if enabled
            if save_video:
                air_frame_path = os.path.join(air_frames_dir, f"frame_{frame_count:04d}.png")
                steps_frame_path = os.path.join(steps_frames_dir, f"frame_{frame_count:04d}.png")
                plot_tensor(Air, "Air left in lungs", save_path=air_frame_path)
                plot_tensor(steps_in, 'seconds surviving', save_path=steps_frame_path)
            else:
                plot_tensor(Air, "Air left in lungs")
                plot_tensor(steps_in, 'seconds surviving')
            
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    else:
        # Gap method
        print("Using Gap Method...")
        while alive_mask.any():
            while previously_alive == alive_mask.sum():
                gap_step(gaps[i])
                i += 1
            print(f"Step {i}:")
            previously_alive = alive_mask.sum()
            y.append(previously_alive)
            
            # Save frames if enabled
            if save_video:
                air_frame_path = os.path.join(air_frames_dir, f"frame_{frame_count:04d}.png")
                steps_frame_path = os.path.join(steps_frames_dir, f"frame_{frame_count:04d}.png")
                plot_tensor(Air, "Air left in lungs", save_path=air_frame_path)
                plot_tensor(steps_in, 'seconds surviving', save_path=steps_frame_path)
            else:
                plot_tensor(Air, "Air left in lungs")
                plot_tensor(steps_in, 'seconds surviving')
            
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    return y, i

# Run the simulation
y, final_step = run_simulation()
    
def create_video_from_frames(frames_dir, output_path, fps=10):
    """Create MP4 video from saved frames"""
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    if not frame_files:
        print(f"No frames found in {frames_dir}")
        return
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    height, width, layers = first_frame.shape
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        video.write(frame)
    
    video.release()
    print(f"Video saved to {output_path}")

def create_combined_frames(air_frames_dir, steps_frames_dir, combined_frames_dir):
    """Create side-by-side combined frames from air and steps frames"""
    if not os.path.exists(combined_frames_dir):
        os.makedirs(combined_frames_dir, exist_ok=True)
    
    air_files = sorted([f for f in os.listdir(air_frames_dir) if f.endswith('.png')])
    steps_files = sorted([f for f in os.listdir(steps_frames_dir) if f.endswith('.png')])
    
    if len(air_files) != len(steps_files):
        print(f"Warning: Mismatched frame counts - Air: {len(air_files)}, Steps: {len(steps_files)}")
        return
    
    print(f"Creating {len(air_files)} combined frames...")
    
    for i, (air_file, steps_file) in enumerate(zip(air_files, steps_files)):
        # Read both frames
        air_frame = cv2.imread(os.path.join(air_frames_dir, air_file))
        steps_frame = cv2.imread(os.path.join(steps_frames_dir, steps_file))
        
        if air_frame is None or steps_frame is None:
            print(f"Error reading frame {i}")
            continue
        
        # Ensure both frames have the same height
        height = max(air_frame.shape[0], steps_frame.shape[0])
        air_frame = cv2.resize(air_frame, (air_frame.shape[1], height))
        steps_frame = cv2.resize(steps_frame, (steps_frame.shape[1], height))
        
        # Combine frames side by side
        combined_frame = np.hstack([air_frame, steps_frame])
        
        # Save combined frame
        combined_path = os.path.join(combined_frames_dir, f"combined_{i:04d}.png")
        cv2.imwrite(combined_path, combined_frame)
    
    print(f"Combined frames saved to {combined_frames_dir}")

def create_gif_from_frames(frames_dir, output_path, fps=10):
    """Create GIF from frames using OpenCV and save as GIF"""
    try:
        from PIL import Image
        
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
        if not frame_files:
            print(f"No frames found in {frames_dir}")
            return
        
        # Load all frames
        frames = []
        for frame_file in frame_files:
            frame_path = os.path.join(frames_dir, frame_file)
            # Convert BGR to RGB for PIL
            frame_bgr = cv2.imread(frame_path)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            frames.append(pil_frame)
        
        # Calculate duration per frame in milliseconds
        duration_ms = int((1000 / fps))
        
        # Save as GIF
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0  # Loop forever
        )
        print(f"GIF saved to {output_path}")
        
    except ImportError:
        print("PIL (Pillow) not available. Install with: pip install Pillow")
        print("Falling back to MP4 creation...")
        # Fallback to MP4 if PIL not available
        mp4_path = output_path.replace('.gif', '.mp4')
        create_video_from_frames(frames_dir, mp4_path, fps)

# Create videos from saved frames (only if saving was enabled)
if save_video:
    print("\nCreating videos from saved frames...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Create individual MP4 videos
    create_video_from_frames(air_frames_dir, os.path.join(output_dir, "air_simulation.mp4"))
    create_video_from_frames(steps_frames_dir, os.path.join(output_dir, "steps_simulation.mp4"))
    
    # Create combined side-by-side frames
    combined_frames_dir = os.path.join(output_dir, "combined_frames")
    create_combined_frames(air_frames_dir, steps_frames_dir, combined_frames_dir)
    
    # Create combined MP4 and GIF
    create_video_from_frames(combined_frames_dir, os.path.join(output_dir, "combined_simulation.mp4"))
    create_gif_from_frames(combined_frames_dir, os.path.join(output_dir, "combined_simulation.gif"), fps=24)

    # Also create individual GIFs
    create_gif_from_frames(air_frames_dir, os.path.join(output_dir, "air_simulation.gif"), fps=24)
    create_gif_from_frames(steps_frames_dir, os.path.join(output_dir, "steps_simulation.gif"), fps=24)


# plot and save the y data
print(f"\nSimulation completed!")
print(f"lung capacity: {M}, breath girth: {N}, seconds alive: {steps_in[0,M-1]}")

plt.figure(figsize=(10, 6))
plt.plot(N*M-np.array(y))
method_name = "Prime Step Method" if method_choice == 1 else "Gap Method"
plt.title(f'Number of dead Entities Over Time')
plt.xlabel('Simulation Steps')
plt.ylabel('Number of dead Entities')
plt.grid(True)

if save_video:
    plt.savefig(os.path.join(output_dir, "alive_entities_plot.png"), dpi=300, bbox_inches='tight')
    
    # Save y data as text file
    with open(os.path.join(output_dir, "alive_entities_data.txt"), 'w') as f:
        f.write(f"# Number of alive entities at each step - {method_name}\n")
        f.write("# Step\tAlive_Count\n")
        for step, count in enumerate(y):
            f.write(f"{step}\t{count}\n")

plt.show()

if save_video:
    print(f"\nAll outputs saved to '{output_dir}' directory:")
    print(f"- air_simulation.mp4: Video of air levels")
    print(f"- air_simulation.gif: GIF of air levels")
    print(f"- steps_simulation.mp4: Video of survival times")
    print(f"- steps_simulation.gif: GIF of survival times")
    print(f"- combined_simulation.mp4: Side-by-side video")
    print(f"- combined_simulation.gif: Side-by-side GIF")
    print(f"- alive_entities_plot.png: Plot of alive entities over time")
    print(f"- alive_entities_data.txt: Raw data of alive entities")
else:
    print("\nSimulation completed (no files saved)")

print("\nPress 'q' in any visualization window to exit")

while True:
    plot_tensor(Air, "Final Air State")  # visualize the final Air tensor
    plot_tensor(steps_in, f'Finished after {final_step} steps')  # visualize the final steps_in tensor
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


