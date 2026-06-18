

### asap recon running on tyger cloud
To deploy a python scripts on tyger cloud where you can access advanced GPU processors, the input file needs to be in [MRD format](https://ismrmrd.github.io/mrd/reference/model.html). 
Tyger reads a file in a streaming manner to allow real time processing, meaning it reads the binary data from the beginning of the file. Mapvbvd reads Siemens raw data `.dat` file when the entire file is loaded on disk.
First step is to convert your siemens .dat file into mrd format, which can be done in `convert_siemens_to_mrd.py`. This takes additional parameters e.g. binning, matrix size, as cmd line arguments to be used in reconstruction later.
Then, use tyger to run the reconstruction, `tyger_recon.py`, steps described below.

### setup tyger
1. Install tyger cli following the [tyger tutorial](https://microsoft.github.io/tyger/introduction/installation/cloud-installation.html#install-the-tyger-cli)
2. Login to tyger. Contact Steve or Kento for spinhance tyger login credentials. Once you obtain PEM and `LOGIN.yml` file, run `tyger login -f LOGIN.yml` on terminal.

### run reconstruction on tyger
Tyger reads mrd file with pipings. Use `cat` command to read the converted raw data mrd file, `stdout` to `tyger run exec -f tyger_deploy/recon_codespec.yml`, which specifies the instance to run reconstruction. 
`--logs` option lets you read the print statements from your python code. At the end, `stdin` to output your file into another mrd file with images at your preferred filepath. 

Example: Suppose the raw mrd file exists under directory `data` at the repo root. 
`cat data/xe_dyn_raw.mrd | tyger run exec -f tyger_deploy/recon_codespec.yml --logs > ./xe_dyn_recon.mrd`
