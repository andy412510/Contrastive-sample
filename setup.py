from setuptools import setup, find_packages


setup(name='ICASSP2023',
      version='1.0.0',
      description='ICASSP paper',
      author='Andy',
      author_email='',
      url='https://github.com/andy412510/Contrastive-sample',
      install_requires=[
          'numpy', 'six', 'h5py', 'Pillow', 'scipy', 'tensorboard', 'opencv-python',
          'scikit-learn', 'metric-learn', 'faiss-gpu==1.6.3', 'PyYAML', 'tqdm'],
      packages=find_packages(),
      keywords=[
          'Novel View Synthesis'
          'Contrastive Learning',
          'Unsupervised Person Re-identification'
      ])