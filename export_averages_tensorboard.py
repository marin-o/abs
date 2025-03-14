import os
import numpy as np
import csv
from tensorboard.backend.event_processing import event_accumulator

logs_path = 'logs'

averages = {}

suffixes = {
    'highway-fast-v0': ['_merge', '_merge_optimized', '_optimized', ''],
    'merge-v0': ['_optimized', '']
}
environments = ['highway-fast-v0', 'merge-v0']
algorithms = ['PPO', 'QRDQN', 'A2C']

combinations = []
for algo in algorithms:
    for env in environments:
        for suffix in suffixes[env]:
            combinations.append(f"{algo}_{env}{suffix}")

for combination in combinations:
    averages[combination] = []

for root, dirs, files in os.walk(logs_path):
    if not any(file.endswith('.zip') for file in files):
        event_files = [f for f in files if "tfevents" in f]
        
        if event_files:
            event_file_paths = [os.path.join(root, f) for f in event_files]
            event_accumulators = [event_accumulator.EventAccumulator(file) for file in event_file_paths]
            for event_acc in event_accumulators:
                event_acc.Reload()

            for event_acc in event_accumulators:
                if 'Evaluation/Episode Reward' in event_acc.Tags()['scalars']:
                    scalar_data = event_acc.Scalars('Evaluation/Episode Reward')
                    key = os.path.relpath(root, logs_path)
                    
                    for combination in combinations:
                        if combination in key:
                            for scalar in scalar_data:
                                averages[combination].append(scalar.value)

sorted_averages = sorted(averages.items(), key=lambda x: x[0])

csv_filename = 'average_values_sorted.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Configuration', 'Average Value'])
    
    for key, values in sorted_averages:
        if values:
            average_value = np.mean(values)
            writer.writerow([key, average_value])
        else:
            writer.writerow([key, 'No Data'])

print(f"Results exported to {csv_filename}")