import os
from setuptools import setup, Extension
from setuptools.command.install import install
# from distutils.command.build_py import build_py
from setuptools.command.build_py import build_py
# from setuptools.command.build_ext import build_ext
from subprocess import call, check_output
from multiprocessing import cpu_count

# os.environ["CXX"] = "g++"
BASEPATH = os.path.dirname(os.path.abspath(__file__))
print "\nBasepath: ", BASEPATH

# Path to the `DeepJetCore` folder within the package
# DEEPJETCORE = os.path.join(BASEPATH, 'DeepJetCore')
DEEPJETCORE = './DeepJetCore'
print "\nDeepjetcore: ", DEEPJETCORE

# Path to the `compiled` folder within `DeepJetCore`
COMPILEPATH = os.path.join(DEEPJETCORE, 'compiled')
print "\nCompile Path: ", COMPILEPATH

# Path to the `interface` folder within `DeepJetCore/compiled`
INTERFACEPATH = os.path.join(COMPILEPATH, 'interface')
print "\nCompile Path: ", INTERFACEPATH

# Exporting the value to a variable for use to set paths
CONDA_PREFIX = os.environ['CONDA_PREFIX']

# Path to configuration files required to build some of the extensions
PYCONFIG_PATH = os.path.join(
    CONDA_PREFIX, 'lib', 'python2.7', 'config')
'''
if os.environ['PYTHON_VERSION']:
    PYCONFIG_PATH = os.path.join(
        CONDA_PREFIX, 'lib', 'python' +
        str(os.environ['PYTHON_VERSION']), 'config')
'''

# declare command to run manual `make` using Makefile in COMPILEPATH
# --to be deprecated in DeepJetCore version 0.0.6
cmd = [
    'make',
]
try:
    cmd.append('-j%d' % cpu_count())
except NotImplementedError:
    print 'Unable to determine number of CPUs. \
    Using single threaded make.'
options = [
#    '--directory=' + COMPILEPATH,
    '--makefile=Makefile',
]
cmd.extend(options)
# print "\n\n" + str(cmd) + "\n\n"


class DeepJetCoreBuild(build_py):
    '''
    Override the default `build` command
    to implement custom commands
    -- deprecated in DeepJetCorev0.0.5
    '''
    def run(self):
        # run original build_py code
	os.environ["CC"] = "g++"
        call(cmd, cwd=COMPILEPATH)
        # print "\n\n\n*****running original DeepJetCore build_py*****\n\n\n"
        build_py.run(self)

        # print "\n\n*********running custom build_py***********\n\n"
        # call(cmd)

class DeepJetCoreInstall(install):
    '''
    Override the default `install` command
    to implement custom commands
    -- deprecated in DeepJetCorev0.0.5
    '''
    def run(self):
        # if BUILDFLAG==0:
        # print "\n\n\n*****running custom DeepJetCore install*****\n\n\n"
        # call(cmd, cwd=DEEPJETCORE)
        # run original install code
        # print "\n\n\n*****running original DeepJetCore install*****\n\n\n"
        install.run(self)

def retrieveReadmeContent():
    '''
    Retrieve the description for the package
    from the README.rst file in BASEPATH
    '''
    with open(os.path.join(BASEPATH, 'README.rst')) as f:
        return f.read()
'''
root_flags = [
	'-pthread',
	'-std=c++11',
	'-Wno-deprecated-declarations',
	'-m64',
	'-lCore',
	'-lRIO',
	'-lNet',
	'-lHist',
	'-lGraf',
	'-lGraf3d',
	'-lGpad',
	'-lTree',
	'-lRint',
	'-lPostscript',
	'-lMatrix',
	'-lPhysics',
	'-lMathCore',
	'-lThread',
	'-lm',
	'-ldl',
	'-rdynamic',
]
'''

root_flags = check_output(['root-config', '--cflags', '--libs', '--glibs']).replace('\n','').split(' ')
cpp_compiler_flags = root_flags + ['-O2', '-fPIC', '-c']

cpp_lib_dirs = [
    os.path.join(CONDA_PREFIX, 'lib'),
    os.path.join(COMPILEPATH, 'interface'),
    '/usr/local/lib/',
]
cpp_include_dirs = [
    os.path.join(CONDA_PREFIX, 'include'),
    '/usr/local/include'
]

module_lib_dirs = [
    os.path.join(CONDA_PREFIX, 'lib'), PYCONFIG_PATH,
    os.path.join(COMPILEPATH, 'interface'),
]
boost_include_dirs = [
	os.path.join(CONDA_PREFIX, 'include'),
]
module_compiler_flags = root_flags + ['-fPIC', '-Wl,--export-dynamic']

quicklz = Extension(
    'libquicklz',
    include_dirs=[os.path.join(COMPILEPATH, 'interface')],
    sources=[os.path.join(COMPILEPATH, 'quicklzpy.c')],
    language='c')

cpp_indata = Extension(
    'DeepJetCore.compiled.indata',
    extra_compile_args=cpp_compiler_flags,
    sources=[os.path.join(INTERFACEPATH,
        'indata_wrap.cxx')],
    include_dirs=cpp_lib_dirs,
    libraries=['python2.7'],
    language='c++')

cpp_helper = Extension(
    'DeepJetCore.compiled.helper',
    extra_compile_args=cpp_compiler_flags,
    sources=[os.path.join(INTERFACEPATH,
        'helper_wrap.cxx')],
    include_dirs=cpp_lib_dirs,
    libraries=['python2.7'],
    language='c++')

cpp_colorToTColor = Extension(
    'DeepJetCore.compiled.colorToTColor',
    extra_compile_args=cpp_compiler_flags,
    sources=[os.path.join(INTERFACEPATH,
        'colorToTColor_wrap.cxx')],
    include_dirs=cpp_lib_dirs,
    libraries=['python2.7'],
    language='c++')

cpp_rocCurve = Extension(
    'DeepJetCore.compiled.rocCurve',
    extra_compile_args=cpp_compiler_flags,
    sources=[os.path.join(INTERFACEPATH,'rocCurve_wrap.cxx')],
    include_dirs=cpp_lib_dirs,
    libraries=['python2.7'],
    language='c++')

cpp_rocCurveCollection = Extension(
    'DeepJetCore.compiled.rocCurveCollection',
    extra_compile_args=cpp_compiler_flags,
    sources=[os.path.join(INTERFACEPATH,
        'rocCurveCollection_wrap.cxx')],
    include_dirs=cpp_lib_dirs,
    libraries=['python2.7'],
    language='c++')

cpp_friendTreeInjector = Extension(
    'DeepJetCore.compiled.friendTreeInjector',
    extra_compile_args=cpp_compiler_flags,
    sources=[os.path.join(INTERFACEPATH,
        'friendTreeInjector_wrap.cxx')],
    include_dirs=cpp_lib_dirs,
    libraries=['python2.7'],
    language='c++')

c_meanNormZeroPad = Extension(
    'DeepJet.compiled.c_meanNormZeroPad',
    extra_compile_args=module_compiler_flags,
    sources=[os.path.join(COMPILEPATH, 'src',
        'c_meanNormZeroPad.C')],
    include_dirs=module_lib_dirs,
    runtime_library_dirs=[boost_include_dirs],
    libraries=['boost_python', 'python2.7'],
    language='c++')

c_makePlots = Extension(
    'DeepJet.compiled.c_makePlots',
    extra_compile_args=module_compiler_flags,
    sources=[os.path.join(COMPILEPATH, 'src',
        'c_makePlots.C')],
    include_dirs=module_lib_dirs,
    runtime_library_dirs=[boost_include_dirs],
    libraries=['boost_python', 'python2.7'],
    language='c++')

c_makeROCs = Extension(
    'DeepJet.compiled.c_makeROCs',
    extra_compile_args=module_compiler_flags,
    sources=[os.path.join(COMPILEPATH, 'src',
        'c_makeROCs.C')],
    include_dirs=module_lib_dirs,
    runtime_library_dirs=[boost_include_dirs],
    libraries=['boost_python', 'python2.7'],
    language='c++')

c_readArrThreaded = Extension(
    'DeepJet.compiled.c_readArrThreaded',
    extra_compile_args=module_compiler_flags,
    sources=[os.path.join(COMPILEPATH, 'src',
        'c_readArrThreaded.C')],
    include_dirs=module_lib_dirs,
    runtime_library_dirs=[boost_include_dirs],
    libraries=['boost_python', 'python2.7'],
    language='c++')

c_randomSelect = Extension(
    'DeepJet.compiled.c_randomSelect',
    extra_compile_args=module_compiler_flags,
    sources=[os.path.join(COMPILEPATH, 'src',
        'c_randomSelect.C')],
    include_dirs=module_lib_dirs,
    runtime_library_dirs=[boost_include_dirs],
    libraries=['boost_python', 'python2.7'],
    language='c++')

# To Do: modify package DeepJetCore to allow find_packages()
# to find all the subpackages recursively by adding this to __init__.py
# __path__ = __import__('pkgutil').extend_path(__path__, __name__)
setup(name='DeepJetCore',
      version='0.0.5',
      description='The DeepJetCore Library: Deep Learning \
      for High-energy Physics',
      url='https://github.com/DL4J/DeepJetCore',
      author='CERN - CMS Group (EP-CMG-PS)',
      author_email='swapneel.mehta@cern.ch',
      license='Apache',
      long_description=retrieveReadmeContent(),
      packages=['DeepJetCore',
		'DeepJetCore.preprocessing',
		'DeepJetCore.evaluation',
		'DeepJetCore.training'],
      scripts=['./DeepJetCore/bin/plotLoss.py',
               './DeepJetCore/bin/plotLoss.py',
               './DeepJetCore/bin/batch_conversion.py',
               './DeepJetCore/bin/check_conversion.py',
               './DeepJetCore/bin/convertFromRoot.py',
               './DeepJetCore/bin/predict.py',
               './DeepJetCore/bin/addPredictionLabels.py',
               './DeepJetCore/bin/convertDCtoNumpy.py',
               './DeepJetCore/bin/convertToTF.py'],
      python_requires='~=2.7',
      install_requires=[
		'cycler',
		'funcsigs',
		'functools32',
		'h5py',
		'tensorflow',
		'Keras',
		'matplotlib',
		'mock',
		'pbr',
		'protobuf',
		'pyparsing',
		'python-dateutil',
		'pytz',
		'PyYAML',
	        'subprocess32'
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
          	'build_py': DeepJetCoreBuild,
      },
      ext_modules=[
	])

'''
quicklz,
cpp_indata,
cpp_helper,
cpp_friendTreeInjector,
cpp_rocCurve,
cpp_rocCurveCollection,
c_makePlots,
c_makeROCs,
c_meanNormZeroPad,
c_randomSelect,
c_readArrThreaded,
'''
