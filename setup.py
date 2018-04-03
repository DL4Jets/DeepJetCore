import os
from setuptools import setup, Extension
from setuptools.command.install import install
# from distutils.command.build_py import build_py
from setuptools.command.build_ext import build_ext
# from setuptools.command.build_py import build_py
# from subprocess import call
from multiprocessing import cpu_count

BASEPATH = os.path.dirname(os.path.abspath(__file__))
print "\nBasepath: ", BASEPATH

DEEPJETCORE = os.path.join(BASEPATH, 'DeepJetCore')
print "\nDeepjetcore: ", DEEPJETCORE

COMPILEPATH = os.path.join(DEEPJETCORE, 'compiled')
print "\Compile Path: ", COMPILEPATH

CONDA_PREFIX = os.environ('CONDA_PREFIX')

PYCONFIG_PATH = os.path.join(CONDA_PREFIX, 'lib',
                             'python' + str(os.environ('PYTHON_VERSION')),
                             'config')

# declare `make` command
cmd = [
    'make',
]
try:
    cmd.append('-j%d' % cpu_count())
except NotImplementedError:
    print 'Unable to determine number of CPUs. \
    Using single threaded make.'
options = [
    '--directory=' + COMPILEPATH,
    '--makefile=Makefile',
]
cmd.extend(options)
print "\n\n" + str(cmd) + "\n\n"


class DeepJetCoreBuildExt(build_ext):
    def run(self):
        # run original build code
        print "\n\n\n*****running original DeepJetCore build_py*****\n\n\n"
        build_ext.run(self)
        # print "\n\n*********running custom build_py***********\n\n"
        # call(cmd, cwd=DEEPJETCORE)


class DeepJetCoreInstall(install):
    def run(self):
        # if BUILDFLAG==0:
        # print "\n\n\n*****running custom DeepJetCore install*****\n\n\n"
        # call(cmd, cwd=DEEPJETCORE)
        # run original install code
        print "\n\n\n*****running original DeepJetCore install*****\n\n\n"
        install.run(self)


def retrieveReadmeContent():
    with open(os.path.join(BASEPATH, 'README.rst')) as f:
        return f.read()


quicklz = Extension('DeepJet.compiled.quicklz',
                    include_dirs=[os.path.join(COMPILEPATH, 'interface')],
                    sources=['quicklzpy.c'])

cpp_compiler_args = ['-O2', '-fPIC', '-c']

cpp_indata = Extension('DeepJetCore.compiled.indata')

boost_include_dirs = [os.path.join(CONDA_PREFIX, 'include'), ]
module_lib_dirs = [os.path.join(CONDA_PREFIX, 'lib'), PYCONFIG_PATH,
                   os.path.join(COMPILEPATH, 'interface')]
module_compiler_flags = ['-fPIC', '-Wl', '--export-dynamic', ]

c_meanNormZeroPad = Extension('DeepJet.compiled.c_meanNormZeroPad',
                              # define_macros=module_macros,
                              extra_compile_args=module_compiler_flags,
                              sources=[os.path.join(COMPILEPATH, 'src',
                                                    'c_meanNormZeroPad.c')],
                              include_dirs=module_lib_dirs,
                              runtime_library_dirs=[],
                              libraries=['boost_python', 'python2.7'])

c_makePlots = Extension('DeepJet.compiled.c_makePlots',
                        extra_compile_args=module_compiler_flags,
                        sources=[os.path.join(COMPILEPATH, 'src',
                                              'c_makePlots.c')],
                        include_dirs=module_lib_dirs,
                        runtime_library_dirs=[],
                        libraries=['boost_python', 'python2.7'])

c_makeROCs = Extension('DeepJet.compiled.c_makeROCs',
                       extra_compile_args=module_compiler_flags,
                       sources=[os.path.join(COMPILEPATH, 'src',
                                             'c_makeROCs.c')],
                       include_dirs=module_lib_dirs,
                       runtime_library_dirs=[],
                       libraries=['boost_python', 'python2.7'])

c_readArrThreaded = Extension('DeepJet.compiled.c_readArrThreaded',
                              extra_compile_args=module_compiler_flags,
                              sources=[os.path.join(COMPILEPATH, 'src',
                                                    'c_readArrThreaded.c')],
                              include_dirs=module_lib_dirs,
                              runtime_library_dirs=[],
                              libraries=['boost_python', 'python2.7'])

c_randomSelect = Extension('DeepJet.compiled.c_randomSelect',
                           extra_compile_args=module_compiler_flags,
                           sources=[os.path.join(COMPILEPATH, 'src',
                                                 'c_randomSelect.c')],
                           include_dirs=module_lib_dirs,
                           runtime_library_dirs=[],
                           libraries=['boost_python', 'python2.7'])

setup(name='DeepJetCore',
      version='0.0.4',
      description='The DeepJetCore Library: Deep Learning \
      for High-energy Physics',
      url='https://github.com/DL4J/DeepJetCore',
      author='CERN - CMS Group (EP-CMG-PS)',
      author_email='swapneel.mehta@cern.ch',
      license='Apache',
      long_description=retrieveReadmeContent(),
      packages=['DeepJetCore', 'DeepJetCore.preprocessing',
                'DeepJetCore.training', 'DeepJetCore.evaluation',
                'DeepJetCore.compiled'],
      scripts=['DeepJetCore/bin/plotLoss.py',
               'DeepJetCore/bin/plotLoss.py',
               'DeepJetCore/bin/batch_conversion.py',
               'DeepJetCore/bin/check_conversion.py',
               'DeepJetCore/bin/convertFromRoot.py',
               'DeepJetCore/bin/predict.py',
               'DeepJetCore/bin/addPredictionLabels.py',
               'DeepJetCore/bin/convertDCtoNumpy.py',
               'DeepJetCore/bin/convertToTF.py'],
      python_requires='~=2.7',
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
      zip_safe=False,
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: Apache Software License',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Operating System :: Unix',
          'Programming Language :: Python :: 2.7',
      ],
      keywords='deep-learning physics jets cern cms',
      project_urls={
          'Documentation': 'https://github.com/SwapneelM/DeepJetCore/wiki',
          'Source': 'https://github.com/SwapneelM/DeepJetCore',
      },
      cmdclass={
          'build_ext': DeepJetCoreBuildExt,
      },
      ext_modules=[quicklz, c_randomSelect, c_makeROCs, c_meanNormZeroPad,
                   c_makePlots, c_readArrThreaded],
      )
