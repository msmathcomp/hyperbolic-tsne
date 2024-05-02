FROM continuumio/miniconda3:24.1.2-0

# Install necessary system tools
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

COPY . /src
WORKDIR /src

# Create a Conda environment, install the requirements, build the acceleration structures and install them
RUN conda create -y -n htsne python=3.9.16 
RUN conda run -n htsne pip install -r requirements.txt
RUN conda run -n htsne python setup.py build_ext --inplace \
 && conda run -n htsne pip install .

# Run Teaser plot script
RUN conda run -n htsne --cwd experiments_and_plots python plot_tree_teaser.py
