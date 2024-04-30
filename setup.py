from setuptools import setup, find_packages  # Always prefer setuptools over distutils
from codecs import open  # To use a consistent encoding
from os import path

here = path.abspath(path.dirname(__file__))

#for packaging files must be in a package (with init) and listed in package_data
# package-externals can be included with data_files,
# and there is a bug in patter nmatching http://bugs.python.org/issue19286
# install unclear for data_files

# Read the contents of the README.md file for use as long_description
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='dextramixer',

    # Version:
    version='0.0.1',

    description='Bayesian model for pMHC dextramer assignment of single-cell data',
    long_description=long_description,
    long_description_content_type='text/markdown',

    # The project's main homepage.
    url='https://github.com/schubertLab/dextramixer',

    # Author details
    author='Benjamin Schubert',
    author_email='benjamin.schubert@helmholtz-muenchen.de',

    # maintainer details
    maintainer='Benjamin Schubert',
    maintainer_email='benjamin.schubert@helmholtz-muenchen.de',

    # Choose your license
    license='BSD',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',

        # The license as you wish (should match "license" above)
        'License :: OSI Approved :: BSD License',

        'Programming Language :: Python :: 3 :: Only',
    ],

    # What dextramixer relates to:
    keywords='single-cell pMHC binder assignment',

    # Specify  packages via find_packages() and exclude the tests and
    # documentation:
    packages=find_packages(),

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    #include_package_data=True,
    package_data={

    },

    data_files=[
            ('docs', ['CHANGELOG.md']),
            ],

    #package_data is a lie: http://stackoverflow.com/questions/7522250/how-to-include-package-data-with-setuptools-distribute

    # Run-time dependencies.
    install_requires=[
            'numpy>=1.25.2',
            'scipy>=1.11.4'
            'pandas>=2.1.4',
            #'pymc>=5.6.1',
            'statsmodels>=0.14.1',
            'matplotlib>=3.8.3',
            'jax>=0.4.26',
            'numpyro>=0.14.0'
            'arviz>=0.17.1',
            'mudata>=0.2.3'
            'scanpy>=1.9.8',
            'scirpy>=0.13.0',
            ],

)
