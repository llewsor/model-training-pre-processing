# model-training-pre-processing

python -m venv venv
venv\Scripts\activate

pip install numpy opencv-python
pip install opencv-contrib-python
pip install scipy
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install albumentations
pip install tqdm

pip uninstall opencv-python opencv-python-headless -y
pip cache purge # optional, but clears corrupted/incomplete wheels
pip install opencv-python
