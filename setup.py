from setuptools import setup, find_packages

setup(
    name='potoos',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'luminol == 0.4',
        'redis == 5.2.1',
        'numpy == 1.22.4',
    ],
    tests_require=[
        'unittest'
    ],
    test_suite='tests',
    author='Alexis Franche',
    author_email='alexis199807@live.ca',
    description='A package for anomaly detection on Redis time series data using Luminol.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/afranche7/Potoos',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires=">=3.10",
)
