import setuptools

from SystemControl import name, version

with open('README.md', 'r') as desc_file:
    long_description = desc_file.read()

with open('requirements.txt', 'r') as req_file:
    requirements_list = req_file.readlines()

short_description = """
A library for performing data collection, preprocessing, and classification for the task of motor imagery
classification.
"""

setuptools.setup(
    name=name,
    version=version,

    description=short_description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/adrang/SystemControl',

    author='Andrew Festa',
    author_email='andrew@thefestas.com',

    python_requires='>=3.7',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    packages=setuptools.find_packages(),
    install_requires=requirements_list,
    dependency_links=[
        'https://github.com/pyvista/pyvista/tarball/master',
    ],
)
