Getting started
===============

This package requires Python 3.8 or above. Please follow the instructions
below to install the package. Depending on your needs, you can choose between
two types of installation.

Installing dectree through pip (Windows and Linux users)
--------------------------------------------------------

To install the latest version of the package you need to run the following
command::

  pip install dectree
  # or
  pip3 install dectree

To install a specific version of the package you would issue the following
command::

  pip install dectree==0.1.0
  # or
  pip3 install dectree==0.1.0

Developer installation
----------------------

First, clone the package. Then, enter the newly created dectree repository. Finally, develop the setup.py file, and you are good to go ! ::

  git clone https://github.com/Dessia-tech/dectree.git

  cd dectree

  python3 setup.py develop --user
  # or whatever version you are using :
  python3.x setup.py develop --user

Requirements
------------

The installation of dectree requires the installation of other packages listed
in the file setup.py and in the table below. These libraries will be
automatically installed when you install dectree.

=============  ===============  ===========
Dependency     Minimum Version  Usage
=============  ===============  ===========
numpy          latest           computation
matplotlib     latest           display
=============  ===============  ===========

Troubleshooting
---------------

If the installation is successful but your IDE don't recognize the package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this case you may have several versions of Python installed on your
computer. Make sure the `pip` command points to the right Python version, or
that you have selected the desired Python version in your IDE.
You can force the installation of the package on a given Python version by
executing this command::

  python -m pip install dectree

You have to specify the Python version you are working with by replacing
`python` by the Python of your choice. For example, `python3`, `python3.8`,
`python3.9`, etc.
