r"""
This subpackage implements the core linear algebra required for the
:class:`QuasisepSolver`, and a few extras. An intro to these matrices can be
found in `Matrix Computations and Semiseparable Matrices: Linear Systems
<https://muse.jhu.edu/book/16537>`_.

There exist a range of definitions for *quasiseparable matrices* in the
literature, so to be explicit, let's select the one that we will consider in all
that follows. The most suitable definition for our purposes is nearly identical
to the one used by `Eidelman & Gohberg (1999)
<https://link.springer.com/article/10.1007%2FBF01300581>`_, with some small
modifications.

Let's start by considering an :math:`N \times N` *square quasiseparable matrix*
:math:`M` with lower quasiseparable order :math:`m_l` and upper quasiseparable
order :math:`m_u`. We represent this matrix :math:`M` as:

.. math::

  M_{ij} = \left \{ \begin{array}{ll}
    d_i\quad, & \mbox{if }\, i = j \\ {p}_i^T\,\left ( \prod_{k=i-1}^{j+1} {A}_k
    \right )\,{q}_j\quad,   & \mbox{if }\, i > j \\ {h}_i^T\,\left (
    \prod_{k=i+1}^{j-1} {B}_k^T \right )\,{g}_j\quad, & \mbox{if }\, i < j \\
  \end{array}\right .

where

- :math:`i` and :math:`j` both range from :math:`1` to :math:`N`,
- :math:`d_i` is a scalar,
- :math:`{p}_i` and :math:`{q}_j` are both vectors with :math:`m_l` elements,
- :math:`{A}_k` is an :math:`m_l \times m_l` matrix,
- :math:`{g}_j` and :math:`{h}_i` are both vectors with :math:`m_u` elements,
  and
- :math:`{B}_k` is an :math:`m_u \times m_u` matrix.

Comparing this definition to the one from Eidelman & Gohberg, you may notice
that we have swapped the labels of :math:`{g}_j` and :math:`{h}_i`, and that
we've added an explicit transpose to :math:`{B}_k^T`. These changes simplify the
notation and implementation for symmetric matrices where, with our definition,
:math:`{g} = {p}`, :math:`{h} = {q}`, and :math:`{B} = {A}`.

In our definition, the product notation is a little sloppy so, to be more
explicit, this is how the products expand:

.. math::

  \prod_{k=i-1}^{j+1} {A}_k \equiv
  {A}_{i-1}\,{A}_{i-2}\cdots{A}_{j+2}\,{A}_{j+1}

and

.. math::

  \prod_{k=i+1}^{j-1} {B}_k^T \equiv
  {B}_{i+1}^T\,{B}_{i+2}^T\cdots{B}_{j-2}^T\,{B}_{j-1}^T

It is often useful see a concrete example matrix. For example, under our
definition, the general :math:`4 \times 4` quasiseparable matrix is:

.. math::

    M = \left(\begin{array}{cccc}
      d_1 & {h}_1^T{g}_2 & {h}_1^T{B}_2^T{g}_3 & {h}_1^T{B}_2^T{B}_3^T{g}_4 \\
      {p}_2^T{q}_1 & d_2 & {h}_2^T{g}_3 & {h}_2^T{B}_3^T{g}_4 \\
      {p}_3^T{A}_2{q}_1 & {p}_3^T {q}_2 & d_3 & {h}_3^T{g}_4 \\
      {p}_4^T{A}_3{A}_2{q}_1 & {p}_4^T{A}_3{q}_2 & {p}_4^T{q}_3 & d_4 \\
    \end{array}\right)

These matrices allow linear scaling for most basic linear algebra operations
(that's why we like them!), and this subpackage implements many of these
building blocks.
"""

__all__ = ["QuasisepSolver"]

from tinygp.solvers.quasisep.solver import QuasisepSolver
