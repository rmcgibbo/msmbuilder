# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.

from __future__ import print_function, division, absolute_import

import numpy as np
import scipy.linalg
from ..utils import list_of_1d
from scipy.sparse import csgraph, csr_matrix, coo_matrix
from sklearn.base import TransformerMixin

__all__ = [
    '_MappingTransformMixin', '_dict_compose', '_strongly_connected_subgraph',
    '_transition_counts',
    '_solve_msm_eigensystem',
]


class _MappingTransformMixin(TransformerMixin):
    def transform(self, sequences, mode='clip'):
        r"""Transform a list of sequences to internal indexing

        Recall that `sequences` can be arbitrary labels, whereas ``transmat_``
        and ``countsmat_`` are indexed with integers between 0 and
        ``n_states - 1``. This methods maps a set of sequences from the labels
        onto this internal indexing.

        Parameters
        ----------
        sequences : list of array-like
            List of sequences, or a single sequence. Each sequence should be a
            1D iterable of state labels. Labels can be integers, strings, or
            other orderable objects.
        mode : {'clip', 'fill'}
            Method by which to treat labels in `sequences` which do not have
            a corresponding index. This can be due, for example, to the ergodic
            trimming step.

           ``clip``
               Unmapped labels are removed during transform. If they occur
               at the beginning or end of a sequence, the resulting transformed
               sequence will be shorted. If they occur in the middle of a
               sequence, that sequence will be broken into two (or more)
               sequences. (Default)
           ``fill``
               Unmapped labels will be replaced with NaN, to signal missing
               data. [The use of NaN to signal missing data is not fantastic,
               but it's consistent with current behavior of the ``pandas``
               library.]

        Returns
        -------
        mapped_sequences : list
            List of sequences in internal indexing
        """
        if mode not in ['clip', 'fill']:
            raise ValueError('mode must be one of ["clip", "fill"]: %s' % mode)
        sequences = list_of_1d(sequences)

        f = np.vectorize(lambda k: self.mapping_.get(k, np.nan),
                         otypes=[np.float])

        result = []
        for y in sequences:
            a = f(y)
            if mode == 'fill':
                if np.all(np.mod(a, 1) == 0):
                    result.append(a.astype(int))
                else:
                    result.append(a)
            elif mode == 'clip':
                result.extend([a[s].astype(int) for s in
                               np.ma.clump_unmasked(np.ma.masked_invalid(a))])
            else:
                raise RuntimeError()

        return result

    def inverse_transform(self, sequences):
        """Transform a list of sequences from internal indexing into
        labels

        Parameters
        ----------
        sequences : list
            List of sequences, each of which is one-dimensional array of
            integers in ``0, ..., n_states_ - 1``.

        Returns
        -------
        sequences : list
            List of sequences, each of which is one-dimensional array
            of labels.
        """
        sequences = list_of_1d(sequences)
        inverse_mapping = {v: k for k, v in self.mapping_.items()}
        f = np.vectorize(inverse_mapping.get)

        result = []
        for y in sequences:
            uq = np.unique(y)
            if not np.all(np.logical_and(0 <= uq, uq < self.n_states_)):
                raise ValueError('sequence must be between 0 and n_states-1')

            result.append(f(y))
        return result


def _solve_msm_eigensystem(transmat, k):
    """Find the dominant eigenpairs of an MSM transition matrix

    Parameters
    ----------
    transmat : np.ndarray, shape=(n_states, n_states)
        The transition matrix
    k : int
        The number of eigenpairs to find.

    Notes
    -----
    Normalize the left (:math:`\phi`) and right (:math:``\psi``) eigenfunctions
    according to the following criteria.
      * The first left eigenvector, \phi_1, _is_ the stationary
        distribution, and thus should be normalized to sum to 1.
      * The left-right eigenpairs should be biorthonormal:
        <\phi_i, \psi_j> = \delta_{ij}
      * The left eigenvectors should satisfy
        <\phi_i, \phi_i>_{\mu^{-1}} = 1
      * The right eigenvectors should satisfy <\psi_i, \psi_i>_{\mu} = 1

    Returns
    -------
    eigvals : np.ndarray, shape=(k,)
        The largest `k` eigenvalues
    lv : np.ndarray, shape=(n_states, k)
        The normalized left eigenvectors (:math:`\phi`) of ``transmat``
    rv :  np.ndarray, shape=(n_states, k)
        The normalized right eigenvectors (:math:`\psi`) of ``transmat``
    """
    u, lv, rv = scipy.linalg.eig(transmat, left=True, right=True)
    order = np.argsort(-np.real(u))
    u = np.real_if_close(u[order[:k]])
    lv = np.real_if_close(lv[:, order[:k]])
    rv = np.real_if_close(rv[:, order[:k]])
    return _normalize_eigensystem(u, lv, rv)


def _normalize_eigensystem(u, lv, rv):
    """Normalize the eigenvectors of a reversible Markov state model according
    to our preferred scheme.
    """
    # first normalize the stationary distribution separately
    lv[:, 0] = lv[:, 0] / np.sum(lv[:, 0])

    for i in range(1, lv.shape[1]):
        # the remaining left eigenvectors to satisfy
        # <\phi_i, \phi_i>_{\mu^{-1}} = 1
        lv[:, i] = lv[:, i] / np.sqrt(np.dot(lv[:, i], lv[:, i] / lv[:, 0]))

    for i in range(rv.shape[1]):
        # the right eigenvectors to satisfy <\phi_i, \psi_j> = \delta_{ij}
        rv[:, i] = rv[:, i] / np.dot(lv[:, i], rv[:, i])

    return u, lv, rv


def _strongly_connected_subgraph(counts, weight=1, verbose=True):
    """Trim a transition count matrix down to its maximal
    strongly ergodic subgraph.

    From the counts matrix, we define a graph where there exists
    a directed edge between two nodes, `i` and `j` if
    `counts[i][j] > weight`. We then find the nodes belonging to the largest
    strongly connected subgraph of this graph, and return a new counts
    matrix formed by these rows and columns of the input `counts` matrix.

    Parameters
    ----------
    counts : np.array, shape=(n_states_in, n_states_in)
        Input set of directed counts.
    weight : float
        Threshold by which ergodicity is judged in the input data. Greater or
        equal to this many transition counts in both directions are required
        to include an edge in the ergodic subgraph.
    verbose : bool
        Print a short statement

    Returns
    -------
    counts_component :
        "Trimmed" version of ``counts``, including only states in the
        maximal strongly ergodic subgraph.
    mapping : dict
        Mapping from "input" states indices to "output" state indices
        The semantics of ``mapping[i] = j`` is that state ``i`` from the
        "input space" for the counts matrix is represented by the index
        ``j`` in counts_component
    """
    n_states_input = counts.shape[0]
    n_components, component_assignments = csgraph.connected_components(
        csr_matrix(counts >= weight), connection="strong")
    populations = np.array(counts.sum(0)).flatten()
    component_pops = np.array([populations[component_assignments == i].sum() for
                               i in range(n_components)])
    which_component = component_pops.argmax()

    def cpop(which):
        csum = component_pops.sum()
        return 100 * component_pops[which] / csum if csum != 0 else np.nan

    if verbose:
        print("MSM contains %d strongly connected component%s "
              "above weight=%.2f. Component %d selected, with "
              "population %f%%" % (n_components, 's' if (n_components != 1) else '',
                                   weight, which_component, cpop(which_component)))


    # keys are all of the "input states" which have a valid mapping to the output.
    keys = np.arange(n_states_input)[component_assignments == which_component]

    if n_components == n_states_input and counts[np.ix_(keys, keys)] == 0:
        # if we have a completely disconnected graph with no self-transitions
        return np.zeros((0, 0)), {}

    # values are the "output" state that these guys are mapped to
    values = np.arange(len(keys))
    mapping = dict(zip(keys, values))
    n_states_output = len(mapping)

    trimmed_counts = np.zeros((n_states_output, n_states_output), dtype=counts.dtype)
    trimmed_counts[np.ix_(values, values)] = counts[np.ix_(keys, keys)]
    return trimmed_counts, mapping


def _transition_counts(sequences, lag_time=1, sliding_window=True):
    """Count the number of directed transitions in a collection of sequences
    in a discrete space.

    Parameters
    ----------
    sequences : list of array-like
        List of sequences, or a single sequence. Each sequence should be a
        1D iterable of state labels. Labels can be integers, strings, or
        other orderable objects.
    lag_time : int
        The time (index) delay for the counts.

    Returns
    -------
    counts : array, shape=(n_states, n_states)
        ``counts[i][j]`` counts the number of times a sequences was in state
        `i` at time t, and state `j` at time `t+self.lag_time`, over the
        full set of trajectories.
    mapping : dict
        Mapping from the items in the sequences to the indices in
        ``(0, n_states-1)`` used for the count matrix.

    Examples
    --------
    >>> sequence = [0, 0, 0, 1, 1]
    >>> counts, mapping = _transition_counts([sequence])
    >>> print counts
    [[2, 1],
     [0, 1]]
    >>> print mapping
    {0: 0, 1: 1}

    >>> sequence = [100, 200, 300]
    >>> counts, mapping = _transition_counts([sequence])
    >>> print counts
    [[ 0.  1.  0.]
     [ 0.  0.  1.]
     [ 0.  0.  0.]]
    >>> print mapping
    {100: 0, 200: 1, 300: 2}

    Notes
    -----
    `None` and `NaN` are recognized immediately as invalid labels. Therefore,
    transition counts from or to a sequence item which is NaN or None will not
    be counted. The mapping return value will not include the NaN or None.
    """
    if (not sliding_window) and lag_time > 1:
        return  _transition_counts([X[::lag_time] for X in sequences], lag_time=1)

    classes = np.unique(np.concatenate(sequences))
    contains_nan = (classes.dtype.kind == 'f') and np.any(np.isnan(classes))
    contains_none = any(c is None for c in classes)

    if contains_nan:
        classes = classes[~np.isnan(classes)]
    if contains_none:
        classes = [c for c in classes if c is not None]

    n_states = len(classes)

    mapping = dict(zip(classes, range(n_states)))
    mapping_is_identity = np.all(classes == np.arange(n_states))
    mapping_fn = np.vectorize(mapping.get, otypes=[np.int])
    none_to_nan = np.vectorize(lambda x: np.nan if x is None else x,
                               otypes=[np.float])

    counts = np.zeros((n_states, n_states), dtype=float)
    for y in sequences:
        y = np.asarray(y)
        from_states = y[: -lag_time: 1]
        to_states = y[lag_time::1]

        if contains_none:
            from_states = none_to_nan(from_states)
            to_states = none_to_nan(to_states)

        if contains_nan or contains_none:
            # mask out nan in either from_states or to_states
            mask = ~(np.isnan(from_states) + np.isnan(to_states))
            from_states = from_states[mask]
            to_states = to_states[mask]

        if (not mapping_is_identity) and len(from_states) > 0 and len(to_states) > 0:
            from_states = mapping_fn(from_states)
            to_states = mapping_fn(to_states)

        transitions = np.row_stack((from_states, to_states))
        C = coo_matrix((np.ones(transitions.shape[1], dtype=int), transitions),
            shape=(n_states, n_states))
        counts = counts + np.asarray(C.todense())

    return counts / float(lag_time), mapping


def _dict_compose(dict1, dict2):
    """
    Example
    -------
    >>> dict1 = {'a': 0, 'b': 1, 'c': 2}
    >>> dict2 = {0: 'A', 1: 'B'}
    >>> _dict_compose(dict1, dict2)
    {'a': 'A', 'b': 'b'}
    """
    return {k: dict2.get(v) for k, v in dict1.items() if v in dict2}

