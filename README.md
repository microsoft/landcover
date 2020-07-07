# Land cover mapping project

This repository holds both the "frontend" web-application and "backend" web API server that make up our "Land Cover Mapping" tool.
An instance of this tool may be live [here](http://aka.ms/landcoverdemo).

## Project setup instructions

- Open a terminal on the machine
- Install conda (note: if you are using a DSVM on Azure then you can skip this step as conda is preinstalled!)

```bash
# Install Anaconda
cd ~
wget https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh
bash Anaconda3-2019.07-Linux-x86_64.sh # select "yes" for setting up conda init
rm Anaconda3-2019.07-Linux-x86_64.sh

# logout and log back in
exit
```

- Install NVIDIA drivers if you intend on using GPUs; note this might require a reboot (note: again, if you are using a DSVM on a Azure GPU VM then this is also handled)
- Setup the repository and install the demo data

```bash
# Get the project and demo project data
git clone https://github.com/microsoft/landcover.git

wget -O landcover.zip "https://mslandcoverstorageeast.blob.core.windows.net/web-tool-data/landcover.zip"
unzip -q landcover.zip
rm landcover.zip

# unzip the tileset that comes with the demo data
cd landcover/data/basemaps/
unzip -q hcmc_sentinel_tiles.zip
unzip -q m_3807537_ne_18_1_20170611_tiles.zip
rm *.zip
cd ../../../

# install the conda environment
# Note: if using a DSVM on Azure, as of 7/6/2020 you need to first run `sudo chown -R $USER /anaconda/`

cd landcover
conda env create --file environment.yml
cd ..
```

### Configure the *web-tool*

A few more steps are needed to configure the interactive *web-tool*.

- Create and edit `web_tool/endpoints.mine.js`. Replace "localhost" with the address of your machine (or leave it alone it you are running locally), and choose the port you will use (defaults to 8080). Note: make sure this port is open to your machine if you are using a remote sever (e.g. with a DSVM on Azure, use the Networking tab to open port 8080).

```bash
cp landcover/web_tool/endpoints.js landcover/web_tool/endpoints.mine.js
nano landcover/web_tool/endpoints.mine.js
```

- Edit `self._WORKERS` of the SessionHandler class in SessionHandler.py to include the GPU resources you want to use on your machine. By default this is set to use GPU IDs 0 through 4.

``` bash
nano landcover/web_tool/SessionHandler.py
```

## Running an instance of the *web-tool*

Whether you setup the server in an Azure VM or locally, the following steps should apply to start an instance of the server:

- Open a terminal on the machine and `cd` to the root directory (`wherever/you/cloned/landcover/`)
- `python server.py local`
  - This will start an HTTP server on :8080 that both serves the "frontend" web application and responds to API calls from the "frontend", allowing the web-app to interface with our CNN models (i.e. the "backend").
  - The tool comes preloaded with a dataset (defined in `web_tool/datasets.json`) and two models (defined in `web_tool/models.json`).
- You should now be able to visit `http://<your machine's address>:8080/` and see the "frontend" interface.
