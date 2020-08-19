from setuptools import setup


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name='noisy_outlier',
    version='0.1.0',
    author='Jan Diers',
    author_email='jan.diers@uni-jena.de',
    description='Self-Supervised Learning for Outlier Detection.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/JanDiers/self-supervised-outlier',
    packages=['noisy_outlier', 'noisy_outlier.model', 'noisy_outlier.hyperopt'],
    classifiers=[
                  "Programming Language :: Python :: 3",
                  "License :: OSI Approved :: MIT License",
                  "Operating System :: OS Independent",
              ],
    python_requires='>=3.6',
    install_requires=[
        'scikit-learn',
    ],
)
