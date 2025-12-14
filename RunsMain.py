import subprocess

runs = 1

for _ in range(runs):
    subprocess.run(['python', 'train.py', '--dataset', 'snippets'])

# Snippets