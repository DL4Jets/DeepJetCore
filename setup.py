from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='DeepJetCore',
      version='0.0.1',
      description='The DeepJetCore Library: Deep Learning \
      for High-energy Physics',
      url='https://github.com/SwapneelM/DeepJetCore',
      author='CERN - CMS Group (EP-CMG-PS)',
      author_email='swapneel.mehta@cern.ch',
      license='Apache',
      packages=['DeepJetCore'],
      install_requires=[
          'cycler==0.10.0',
          'funcsigs==1.0.2',
          'functools32==3.2.3.post2',
          'h5py==2.6.0',
          'tensorflow==1.0.1',
          'Keras==2.0.0',
          'matplotlib==2.0.0',
          'mock==2.0.0',
          'pbr==2.0.0',
          'protobuf==3.2.0',
          'pyparsing==2.2.0',
          'python-dateutil==2.6.0',
          'pytz==2016.10',
          'PyYAML==3.12',
          'subprocess32==3.2.7'
      ],
      include_package_data=True,
      zip_safe=False)
