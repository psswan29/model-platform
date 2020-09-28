from setuptools import setup, find_packages

setup(name='modeling_platform',
      version='1.0',
      author='Jason',
      author_email='657590764@qq.com',
      description='the code for model building',
      install_requires=[
          'scikit-learn>=0.2.0',
          'matplotlib',
          'seaborn',
          'scipy',
          'numpy',
          'pandas>=1.0.0',
          'statsmodels',
          'sklearn2pmml',
          'factor_analyzer'
      ],

      packages=find_packages())
