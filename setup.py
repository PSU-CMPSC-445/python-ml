from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['tensorflow==2.4.1', 'numpy==1.19.5']

setup(
    name='lego-classifier',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Lego training application package.'
)