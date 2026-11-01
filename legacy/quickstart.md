

# Quick Start

The Photonics LABoratory project consists of the development of a framework to model
and analyse the thermal, mechanical and optical properties of laser beams
systems propagating through various optical media, in an effort to realise novel
beam shaping strategies geared towards additive manufacturing technologies and
applications.

Both metal and polymer additive manufacturing (AM) technologies, have been
identified by the South African government as not only disruptive but also
enabling technologies, that form an integral part of the national strategic
objectives to grow and diversify the economy and alleviate poverty. Recent
forecasts predict an annual global market in excess of $50 billion by the year
2020 for the sale of AM products and services. Apart from the academic
applications to modal decomposition, wavefront reconstruction, multimode and
quality analysis, (laser physics&#x2026;?) etc&#x2026; and in line with the Republic's
strategic objectives, in efforts to secure a portion of this market share for
sovereign benefit we intend to realise a real-time, inline device capable of not
only identifying, classifying and categorising (AM) melt pool and laser beam
profiles (and their thermal loads), but also pre-emptively predict and
compensate for imperfections and potential build errors using a system of
machine learning and adaptive optics.


# Installation

Where possible to development and simulation ecosystem will strive to be as
[Pythonic](https://www.python.org/dev/peps/pep-0008/) as possible. However, we do accommodate MatLab appliances and
interfaces in many instances.


## Anaconda

Most of the above dependencies can be satisfied by installing the Python 3.6.
version of [Anaconda 5.0.1](https://conda.io/docs/user-guide/install/download.html). However with regards to the Python Bindings for HDF5
and Pandoc, you'll need to run the following within your Anaconda environment.

    $ conda install h5py
    $ conda install scikit-image
    $ conda install scikit-cuda


# Usage

A good resource and basic introduction to the scientific Python stack and it's
uses is [Tim Fuller's FEM with Python](https://github.com/tjfulle/fem-with-python).


## Use Git to get source-code and Explore Jupyter Notebooks

Once you've [Created your Bitbucket Account](https://bitbucket.org/account/signup/) and added your [public ssh-key](https://confluence.atlassian.com/bitbucketserver/creating-ssh-keys-776639788.html) to the
private [Bitbucket NLC Novel Lasers](https://bitbucket.org/account/user/nlcnl/ssh-keys/) repository by emailing it to the [admin](mailto:Nlisa@csir.co.za), you
can get access to the source by:

    $ cd your_desired_working_folder
    $ git clone git@bitbucket.org:nlcnl/photonicslab.git your_desired_checkout_name

    # Or without ssh-keys
    $ git clone https://bitbucket.org/nlcnl/photonicslab.git your_desired_checkout_name

The developer or end-user can learn to use the software through a series of
[Jupyter Notebooks](http://ipython.org/notebook.html), which are included with the [src](https://bitbucket.org/nlcnl/varoptic/src). These lessons provide a set
of interactive literate programming tools that are [very effective](https://drivendata.github.io/cookiecutter-data-science/) for
exploratory data analysis. The sub-folder `notebooks/reports` contains more
polished work that is *\`near-publication-ready'*. Generally speaking these serve
as high level functional, systems or acceptance tests. First change directory
into the notebooks folder and then start the notebook server:

    $ cd notebooks
    $ jupyter notebook

This will start the interactive Python session in your browser and from there
you can start the Lessons.
