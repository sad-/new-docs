Containers
==========

This section covers the classes to construct neural network models.

.. currentmodule:: mxnet.gluon.nn

Blocks
------

The :py:mod:`mxnet.gluon.nn` module provides three classes to construct basic
blocks for a neural network model:

.. autosummary::
   :nosignatures:
   :toctree: .

   Block
   HybridBlock
   SymbolBlock

The difference between these three classes:

- :py:class:`Block`: the base class for any neural network layers and models.
  `A look under the hood of gluon`_ and
  `Gluon - Neural network building blocks`_ are tutorials that provide details
  on how to use this class of container.
- :py:class:`HybridBlock`: a subclass of :py:class:`Block` that allows to
  hybridize a model. It constraints operations can be run in the ``forward``
  method, e.g. the `print` function doesn't work any more.
- :py:class:`SymbolBlock`: a subclass of :py:class:`Block` that is able to wrap
  a :py:class:`mxnet.symbol.Symbol` instance into a :py:class:`Block`
  instance.


Sequential containers
---------------------

Besides inheriting :py:class:`mxnet.gluon.nn.Block` to create a neural network
models, :py:mod:`mxnet.gluon.nn` provides two classes to construct a model by
stacking layers sequentially. Refer to the tutorials list in the
`Further Reading`_ section on how to use them.

.. currentmodule:: mxnet.gluon.nn

.. autosummary::
    :toctree: _autogen
    :nosignatures:

    Sequential
    HybridSequential


Concurrent containers
---------------------

The :py:mod:`mxnet.gluon.contrib.nn` package provides two additional containers
to construct models with more than one path, such as the Residual block in
ResNet and Inception block in GoogLeNet.


.. currentmodule:: mxnet.gluon.contrib.nn

.. autosummary::
    :toctree: _autogen
    :nosignatures:

    Concurrent
    HybridConcurrent


Further Reading
---------------

- `A look under the hood of gluon`_
- `Gluon - Neural network building blocks`_


.. disqus::


.. _A look under the hood of gluon: https://gluon.mxnet.io/chapter03_deep-neural-networks/plumbing.html
.. _Gluon - Neural network building blocks: https://mxnet.incubator.apache.org/tutorials/gluon/gluon.html
