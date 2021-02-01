import os
from setuptools import setup, find_packages

# get long_description from README.md
with open("README.md", "r") as fh:
    long_description = fh.read()


# list of all scripts to be included with package
scripts = [os.path.join('scripts',f) for f in os.listdir('scripts')]

setup(
    name='ATL11',
    version='2.0.0.0',
    description='ASAS L3B Land Ice PGE',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/suzanne64/ATL11',
    author='Ben Smith, Suzanne Murphy, Ben Jelley',
    author_email='besmith@uw.edu',
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='altimetry, ICESat-2, remote sensing',
    packages=find_packages(),
    scripts=scripts,
    include_package_data=True,
    package_data={'ATL11':['package_data/*']}
)
