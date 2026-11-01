# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""setup.py for varoptic."""

import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand

class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.verbose = True

    def run_tests(self):
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)

setup(name='photonicslab',
      version='0.50.4',
      description='Laser beam shaping and thermal lensing analysis',
      long_description=open('README.md').read(),
      url='https://github.com/NLisa/photonicslab',
      download_url='https://github.com/NLisa/photonicslab.git',
      author='Nyameko Lisa',
      author_email='lisa.nyameko@gmail.com',
      license='GNU GPLv3',
      packages=['photonicslab'],
      scripts = [],
      keywords=['beam', 'shaping', 'laser', 'gain', 'thermal lensing', 'beam quality', 'beam shaping'],
      tests_require=['pytest'],
      zip_safe=False)
