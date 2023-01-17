# automatic-torax-labeling-scripts

## requeriments

This code requieres python 3.8. It won't work in newer versions

To install dependencies, run:

python3.8 -m venv venv

. venv/bin/activate

pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install pandas==1.4.2 matplotlib==3.2.2 Jinja2==3.1.1 weasyprint opencv-python==4.1.2.30 scikit-image==0.19.3 requests PyYAML==5.3.1 tqdm==4.41.0 seaborn

## Usage
 
To execute the script, first activate the environment if not already with

. venv/bin/activate

and, then, run the command

python process_img_covid.py path_to_image

The report will be generated in the same folder with the name *report.pdf*

