from __future__ import absolute_import, division, print_function
from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = ''  # use '' for first of series, number for 1 and above
_version_extra = 'dev'
# _version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = "pyTotalActivation: sparse spatio-temporal deconvolution for functional MRI"
# Long description will go up on the pypi page
long_description = """

pyTotalActivation
========

License
=======
``pyTotalActivation`` is licensed under the terms of the MIT license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.

All trademarks referenced herein are property of their respective holders.

Copyright (c) 2017--, Karol Jurek, Marcel Falkiewicz, Isik Karahanoglu, 
                      Dimitri Van De Ville, Younes Farouj, Bolton Thomas
"""

NAME = "pyTotalActivation"
MAINTAINER = "Karol Jurek"
MAINTAINER_EMAIL = "kjurek@gmail.com"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "http://github.com/kjurek/pyTotalActivation"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "Karol Jurek, Marcel Falkiewicz, Isik Karahanoglu, Dimitri Van De Ville, Younes Farouj, Bolton Thomas"
AUTHOR_EMAIL = "kjurek@gmail.com, mfalkiewicz@gmail.com"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {'pyTotalActivation': [pjoin('data', '*')]}
REQUIRES = ["numpy", "nilearn", "PyWavelets", "scipy"]
