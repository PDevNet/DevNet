from setuptools import setup, find_packages

setup(
    name="Path_Development_Net",
    version="0.0.1",
    author="PDevNet",
    description="Path Development Network with Finite dimensional Lie Group",
    url="https://github.com/PDevNet/DevNet",
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=['matplotlib == 3.3.2',
                      'ml_collections == 0.1.0',
                      'numpy == 1.19.3',
                      'PyYAML == 6.0',
                      'scikit_learn == 1.0.2',
                      'scipy == 1.5.2',
                      'torch == 1.9.1',
                      'signatory == 1.2.6.1.9.0',
                      'sktime == 0.9.0',
                      'torchaudio == 0.9.1',
                      'torchvision == 0.10.1',
                      'seaborn == 0.11.2',
                      'tqdm'],
)
