"""An absorbing-state variant of the tree AR-HMM.

`models/tarhmm.py` is shared by every pipeline in this repo, so nothing here
modifies it.  Instead this subclasses the transition component and the model,
which is all that is needed: `tARHMM.m_step` dispatches to
`self.transition_component.m_step`, so swapping the component is enough to
change the constraint.

The constraint: one designated state has its transition row pinned to
one-hot self-transition, i.e. P_std[absorbing, absorbing] = 1 and every other
entry in that row is 0.  Once posterior mass enters that state it cannot
leave -- there is no dead -> alive transition.

Three consequences worth knowing:

  * The absorbing state is designated BY INDEX (the last state, k-1, by
    convention), not by phenotype.  EM decides what lands there.  Whether the
    phenotype that lands there is actually death is a separate, post-hoc
    question -- `fit_dead_state_hmm.py` scores it with the same label-blind
    rule used for every other state and records whether the two agree.

  * Cells that are ALREADY dead when their track starts are reachable through
    the initial distribution, which stays free.  A purely transition-based
    detector cannot represent them at all; this is why the already-dead cell
    in the annotations scored zero in every `cancer_death_hmm` arm.

  * The readout changes.  With an absorbing state, "fraction of cells that
    ever entered" is well defined and is the quantity the downstream
    per-condition comparison actually wants, rather than frame occupancy.
"""

import jax.numpy as jnp

from models.tarhmm import tARHMM, TreeTransitions


class AbsorbingTreeTransitions(TreeTransitions):
    """Standard tree transitions with one row pinned to self-transition 1.0."""

    def __init__(self, num_states, absorbing_state, concentration=1.1,
                 stickiness=0.0):
        """
        Args:
            num_states (int): number of latent states.
            absorbing_state (int): index of the state to make absorbing.
            concentration (float): Dirichlet concentration for the free rows.
            stickiness (float): extra diagonal mass for the free rows.
        """
        super().__init__(num_states, concentration=concentration,
                         stickiness=stickiness)
        self.absorbing_state = absorbing_state

    def m_step(self, params, props, batch_stats, m_step_state):
        """Run the usual M-step, then re-pin the absorbing row.

        Pinning happens after the unconstrained update rather than by
        modifying the sufficient statistics, so the free rows are estimated
        exactly as they would be without the constraint.
        """
        params, m_step_state = super().m_step(params, props, batch_stats,
                                              m_step_state)
        row = jnp.zeros(self.num_states).at[self.absorbing_state].set(1.0)
        pinned = params.transition_matrix.at[self.absorbing_state].set(row)
        return params._replace(transition_matrix=pinned), m_step_state


class AbsorbingTARHMM(tARHMM):
    """tARHMM whose `absorbing_state` cannot transition to any other state."""

    def __init__(self, num_states, emission_dim, num_lags=1,
                 absorbing_state=None, initial_probs_concentration=1.1,
                 transition_matrix_concentration=1.1,
                 transition_matrix_stickiness=0.0):
        """
        Args:
            absorbing_state (int | None): index of the absorbing state.  None
                gives the unconstrained model, so the same class can fit both
                arms of the constrained/unconstrained comparison.
        """
        super().__init__(num_states, emission_dim, num_lags=num_lags,
                         initial_probs_concentration=initial_probs_concentration,
                         transition_matrix_concentration=transition_matrix_concentration,
                         transition_matrix_stickiness=transition_matrix_stickiness)
        self.absorbing_state = absorbing_state
        if absorbing_state is not None:
            self.transition_component = AbsorbingTreeTransitions(
                num_states, absorbing_state,
                concentration=transition_matrix_concentration,
                stickiness=transition_matrix_stickiness)


def pin_absorbing_row(transition_matrix, absorbing_state):
    """Pin one row of an initial transition matrix to self-transition 1.0.

    The M-step pins the row on every iteration, but the FIRST E-step runs on
    whatever matrix initialization produced.  Without this the model spends
    its first iteration with a leaky absorbing state.

    Args:
        transition_matrix: (num_states, num_states) array.
        absorbing_state (int | None): row to pin; None returns the input.

    Returns:
        The matrix with the designated row set to one-hot.
    """
    if absorbing_state is None:
        return transition_matrix
    num_states = transition_matrix.shape[0]
    row = jnp.zeros(num_states).at[absorbing_state].set(1.0)
    return transition_matrix.at[absorbing_state].set(row)
