API Reference
=============

Samplers
--------

.. autosummary::
   :toctree: generated/

   geovalidate.PointSampler
   geovalidate.ConstantClassSampler
   geovalidate.StratifiedClassSampler
   geovalidate.MultinomialSampler
   geovalidate.PoissonSampler

Cross-validators
----------------

.. autosummary::
   :toctree: generated/

   geovalidate.HilbertKFold
   geovalidate.BallKFold
   geovalidate.LeaveBallOut
   geovalidate.ClusterStratifiedKFold
   geovalidate.LocalBootstrap
   geovalidate.LocalPermutation

Range detection
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   geovalidate.correlogram_range
   geovalidate.knn_range

Metrics
-------

.. autosummary::
   :toctree: generated/

   geovalidate.area_of_applicability
   geovalidate.gearygram
