.. _changelog:

Changelog
=========

v3.4 (Development)
------------------

Version 3.4.0.dev0 is the current development release of MSMBuilder.

API Changes
~~~~~~~~~~~

- Range-based slicing on dataset objects is no longer allowed. Keys in the
  dataset object don't have to be continuous. The empty slice, e.g. ``ds[:]``
  loads all trajectories in a list (#610).


New Features
~~~~~~~~~~~~

- Dataset objects have a method, ``fit_transform_with`` to simplify the common
  pattern of applying an estimator to a dataset object to produce a new dataset
  object (#610).


Improvements
~~~~~~~~~~~~



v3.3 (August 27, 2015)
----------------------

We're pleased to announce the release of MSMBuilder v3.3.0. The focus of this
release is a completely re-written module for constructing HMMs as well as bug
fixes and incremental improvements.

API Changes
~~~~~~~~~~~

- ``FeatureUnion`` is an estimator that deprecates the functionality of
  ``UnionDataset``. Passing a list of paths to ``dataset()`` will no longer
  automatically yield a ``UnionDataset``. This behavior is still available by
  specifying ``fmt="dir-npy-union"``, but is deprecated (#611).
- The command line flag for featurizers ``--out`` (deprecated in 3.2) now saves
  the featurizer as a pickle file (#546). Please use ``--transformed`` for the
  old behavior. This is consistent with other command-line commands.
- The default number of timescales in ``MarkovStateModel`` is now one less than
  the number of states (was 10). This addresses some bugs with
  ``implied_timescales`` and PCCA(+) (#603).

New Features
~~~~~~~~~~~~

- ``GaussianHMM`` and ``VonMisesHMM`` is rewritten to feature higher code reuse
  and code quality (#583, #582, #584, #572, #570).
- ``KDTree`` can find n nearest points to e.g. a cluster center (#599).
- ``Slicer`` featurizer can slice feature arrays as part of a pipeline
  (#567).

Improvements
~~~~~~~~~~~~

- ``PCCAPlus`` is compatible with scipy 0.16 (#620).
- Documentation improvements (#618, #608, #604, #602)
- Test improvements, especially for Windows (#593, #590, #588, #579, #578,
  #577, #576)
- Bug fix: ``MarkovStateModel.sample()`` produced trajectories of incorrect
  length. This function is still deprecated (#556).
- Bug fix: The muller example dataset did not respect users' specifications for
  initial coordinates (#631).
- ``MarkovStateModel.draw_samples`` failed if discrete trajectories did not
  contain every possible state (#638). Function can now accept a single
  trajectory, as well as a list of them.
- ``SuperposeFeaturizer`` now respects the topology argument when loading the
  reference trajectory (#555).

v3.2 (April 14, 2015)
---------------------

- ``tICA`` ignores too-short trajectories during fitting instead of raising
  an exception
- New methods for sampling from MSM models
- Datasets can be opened in "append" mode
- Compatibility with scipy 0.16
- ``utils.dump`` saves using the pickle protocol. ``utils.load`` is backwards
  compatible.
- The command line flag for featurizers ``--out`` is deprecated. Use
  ``--transformed`` instead. This is consistent with other command-line
  commands.
- Bug fixes

v3.1 (Feb 27, 2015)
-------------------

- Numerous improvements to ``ContinuousTimeMSM`` optimization
- Switch ``ContinuousTimeMSM.score`` to transmat-style GMRQ
- New example dataset with Muller potential
- Assorted bug fixes in the command line layer

v3.0.1 (January 9, 2015)
------------------------

- Fix missing file on PyPI.


v3.0.0 (January 9, 2015)
------------------------

MSMBuilder 3.0 is a complete rewrite of our `previous work
<https://github.com/msmbuilder/msmbuilder-legacy>`_. The focus is on power
and extensibility, with a much wider class of estimators and models
supported throughout the codebase. All users are encouraged to switch to
MSMBuilder 3.0.  Pre-release versions of MSMBuilder 3.0 were called
mixtape.
