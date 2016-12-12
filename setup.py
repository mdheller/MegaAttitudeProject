from setuptools import setup

setup(name='MegaAttitudeProject',
      version='1.1dev0',
      description='Ordinal and multiview factor analysis models of projection',
      url='http://github.com/aaronstevenwhite/MegaAttitudeProject',
      author='Aaron Steven White',
      author_email='aswhite@jhu.edu',
      license='MIT',
      packages=['projectionmodel'],
      install_requires=['numpy',
                        'scipy',
                        'pandas',
                        'pymc',
                        'theano'],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
