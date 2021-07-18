from pathlib import Path
home_dir = Path(__file__).parents[1]
data_dir = home_dir / 'tests' / 'data'
data_dir_real = str(data_dir / 'real_datasets')
data_dir_animals = str(data_dir / 'animals')
# script_dir = str(home_dir / 'camtrappy')
notebook_dir = str(home_dir / 'notebooks')

# Set working directory
import os
os.chdir(data_dir)

# Make project scripts available
import sys
sys.path.insert(0, str(home_dir))

print(
    f'home:      {home_dir}',
    f'data:      {data_dir}',
    # f'scripts:   {script_dir}',
    f'notebooks: {notebook_dir}',
    '--Setup complete--',
    sep='\n'
)
