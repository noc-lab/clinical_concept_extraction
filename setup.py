from setuptools import setup

package_list = ['scipy', 'numpy']
try:
    import tensorflow
except ImportError:  # There is no one, declare dependency
    package_list.append('tensorflow>=1.8.0')


setup(
    name='clinical_concept_extraction',
    version='0.1.1',
    author="Henghui Zhu",
    url='https://github.com/noc-lab/clinical_concept_extraction',
    author_email="henghuiz@bu.edu",
    packages=['clinical_concept_extraction',],
    license='',
    description="A clinical concept extraction tools",
    install_requires=package_list,
)
