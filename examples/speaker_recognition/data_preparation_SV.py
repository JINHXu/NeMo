# download and convert `an4`, 

import os
NEMO_ROOT = os.getcwd()
print(NEMO_ROOT)
import glob
import subprocess
import tarfile
import wget

data_dir = os.path.join(NEMO_ROOT,'data')
os.makedirs(data_dir, exist_ok=True)

# Download the dataset. This will take a few moments...
print("******")
if not os.path.exists(data_dir + '/an4_sphere.tar.gz'):
    an4_url = 'http://www.speech.cs.cmu.edu/databases/an4/an4_sphere.tar.gz'
    an4_path = wget.download(an4_url, data_dir)
    print(f"Dataset downloaded at: {an4_path}")
else:
    print("Tarfile already exists.")
    an4_path = data_dir + '/an4_sphere.tar.gz'

# Untar and convert .sph to .wav (using sox)
tar = tarfile.open(an4_path)
tar.extractall(path=data_dir)

print("Converting .sph to .wav...")
sph_list = glob.glob(data_dir + '/an4/**/*.sph', recursive=True)
for sph_path in sph_list:
    wav_path = sph_path[:-4] + '.wav'
    cmd = ["sox", sph_path, wav_path]
    subprocess.run(cmd)
print("Finished conversion.\n******")