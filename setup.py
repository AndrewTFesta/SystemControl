"""
https://packaging.python.org/tutorials/packaging-projects/

name
    The distribution name of your package. This can be any name as long as only contains letters,
    numbers, _ , and -. It also must not already taken on pypi.org. Be sure to update this with your username,
    as this ensures you won’t try to upload a package with the same name as one which already exists when
    you upload the package.

version
    The package version see PEP 440 for more details on versions.

author
author_email
    used to identify the author of the package.

description
    a short, one-sentence summary of the package.

long_description
    a detailed description of the package. This is shown on the package detail package on the Python Package Index.
    In this case, the long description is loaded from README.md which is a common pattern.

long_description_content_type
    tells the index what type of markup is used for the long description. In this case, it’s Markdown.

url
    the URL for the homepage of the project. For many projects, this will just be a link to GitHub,
    GitLab, Bitbucket, or similar code hosting service.

packages
    a list of all Python import packages that should be included in the distribution package. Instead of
    listing each package manually, we can use find_packages() to automatically discover all packages and
    subpackages. In this case, the list of packages will be example_pkg as that’s the only package present.

classifiers
    gives the index and pip some additional metadata about your package. In this case, the package is only
    compatible with Python 3, is licensed under the MIT license, and is OS-independent. You should always
    include at least which version(s) of Python your package works on, which license your package is
    available under, and which operating systems your package will work on. For a complete list of
    classifiers, see https://pypi.org/classifiers/.
"""

import setuptools

with open('README.md', 'r') as desc_file:
    long_description = desc_file.read()

with open('requirements.txt', 'r') as req_file:
    requirements_list = req_file.readlines()

setuptools.setup(
    name='SystemControl',
    version='0.1',

    description='Provides control system, using OpenBCI Ganglion, for reading and classifying motor-imagery events',
    ong_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/adrang/SystemControl',

    author='Andrew Festa',
    author_email='andrew@thefestas.com',

    packages=setuptools.find_packages(),

    python_requires='>=3.7',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    install_requires=requirements_list,
    dependency_links=[
        'https://github.com/pyvista/pyvista/tarball/master',
    ],
)
