import os
import sys
from pathlib import Path

projectfolder = str(Path.cwd().parent)

sys.path.insert(0, projectfolder)
os.chdir(projectfolder)
print("Current working directory:", projectfolder)