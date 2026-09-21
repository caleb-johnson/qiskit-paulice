# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Implementation of :meth:`.CheckedCircuit.dope`."""

from __future__ import annotations

import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Clifford, PauliList


def dope_circuit(
    circuit: QuantumCircuit,
    check_qubits: tuple[int, ...],
    check_support: tuple[tuple[int, ...], ...],
    num_sites: int | None,
    wires: str,
    angle: float | None,
    seed: int | np.random.Generator | None,
) -> tuple[QuantumCircuit, list[tuple[int, int | None]]]:
    """Dope ``circuit``; see :meth:`.CheckedCircuit.dope` for the arguments.

    Returns:
        The doped circuit and its doped wires as ``(qubit, after_instruction)`` pairs,
        sorted by circuit position.
    """
    if wires not in ("all", "after_entangling", "before_entangling"):
        raise ValueError(
            f"wires must be 'all', 'after_entangling', or 'before_entangling', not {wires!r}."
        )
    site_qubits = set(range(circuit.num_qubits)) - set(check_qubits)
    positions, qubits, gens, full_clifford = _sweep_wire_segments(circuit, site_qubits, wires)

    # The back-propagation of a generator P to the input is C^dag P C for the whole-circuit
    # Clifford C; a diagonal image stabilizes |0^n>, so the rotation only applies a phase.
    input_diagonal = ~gens.evolve(full_clifford, frame="h").x.any(axis=1)
    output_diagonal = ~gens.x.any(axis=1)

    # A wire preserves a check iff Z there commutes with the check's back-cumulant. By
    # conjugation through the suffix, that is the commutation of the forward-propagated
    # generator with the check's syndrome operator, a Z-product on its support.
    active = np.ones(len(gens), dtype=bool)
    for support in check_support:
        active &= gens.x[:, list(support)].sum(axis=1) % 2 == 0
    _prune_to_fixpoint(gens, input_diagonal, output_diagonal, active)
    valid = [int(i) for i in np.flatnonzero(active)]

    if num_sites is None:
        chosen_idx = valid
    elif not 0 <= num_sites <= len(valid):
        raise ValueError(
            f"num_sites ({num_sites}) must be between 0 and the number of valid doping "
            f"sites ({len(valid)})."
        )
    else:
        # A subset of a fixed point need not be one: re-prune each draw and top up.
        rng = np.random.default_rng(seed)
        pool = [valid[i] for i in rng.permutation(len(valid))]
        selected = np.zeros(len(gens), dtype=bool)
        while need := num_sites - int(selected.sum()):
            if not pool:
                raise ValueError(
                    f"Could only draw {int(selected.sum())} irreducible doping sites of "
                    f"the requested {num_sites}; request fewer sites or pass "
                    "num_sites=None."
                )
            selected[pool[:need]] = True
            del pool[:need]
            _prune_to_fixpoint(gens, input_diagonal, output_diagonal, selected)
        chosen_idx = [int(i) for i in np.flatnonzero(selected)]

    chosen = sorted((int(positions[i]), int(qubits[i])) for i in chosen_idx)
    inserts: dict[int, list[int]] = {}
    for position, qubit in chosen:
        inserts.setdefault(position, []).append(qubit)
    angles = iter(ParameterVector("dope", len(chosen))) if angle is None else None
    doped = circuit.copy_empty_like()
    for position in range(len(circuit.data) + 1):
        for qubit in inserts.get(position, ()):
            doped.rz(angle if angles is None else next(angles), qubit)
        if position < len(circuit.data):
            doped.append(circuit.data[position])

    return doped, [(qubit, position - 1 if position else None) for position, qubit in chosen]


def _sweep_wire_segments(
    circuit: QuantumCircuit, site_qubits: set[int], wires: str
) -> tuple[np.ndarray, np.ndarray, PauliList, Clifford]:
    """Enumerate candidate wire segments with their generators conjugated to the output.

    A backward pass keeps the Clifford of all gates after the current one; stabilizer row
    ``q`` of its tableau is the image of ``Z_q``. One boundary represents each gate-free
    segment. Segments past a terminal measurement are skipped; ``wires`` restricts to those
    directly following (``"after_entangling"``) or preceding (``"before_entangling"``) a
    multi-qubit gate.

    Returns:
        Time-sorted ``(positions, qubits, gens, full_clifford)``: a rotation at candidate ``i``
        precedes ``circuit.data[positions[i]]`` on ``qubits[i]`` with output image ``gens[i]``.

    Raises:
        ValueError: on a non-Clifford instruction or a non-terminal measurement.
    """
    data = circuit.data
    suffix = Clifford.from_label("I" * circuit.num_qubits)
    touched: set[int] = set()
    positions: list[int] = []
    qubits: list[int] = []
    x_rows: list[np.ndarray] = []
    z_rows: list[np.ndarray] = []

    def _record(position: int, qubit: int) -> None:
        positions.append(position)
        qubits.append(qubit)
        x_rows.append(suffix.stab_x[qubit].copy())
        z_rows.append(suffix.stab_z[qubit].copy())

    for position in range(len(data) - 1, -1, -1):
        inst = data[position]
        name = inst.operation.name
        if name == "barrier":
            continue
        qargs = [circuit.find_bit(qubit).index for qubit in inst.qubits]
        if name == "measure":
            # Sweeping backward, a gate already seen on this qubit lies after the measurement.
            if qargs[0] in touched:
                raise ValueError(
                    f"Qubit {qargs[0]} is used after its measurement; only terminal "
                    "measurements are supported."
                )
            continue
        touched.update(qargs)
        entangling = len(qargs) > 1
        if wires == "all" or (wires == "after_entangling" and entangling):
            for qubit in qargs:
                if qubit in site_qubits:
                    _record(position + 1, qubit)
        try:
            suffix = suffix.dot(inst.operation, qargs=qargs)
        except QiskitError as exc:
            raise ValueError(f"Non-Clifford instruction in circuit: {name!r}") from exc
        if wires == "before_entangling" and entangling:
            # The suffix now includes this gate, as seen by a rotation directly before it.
            for qubit in qargs:
                if qubit in site_qubits:
                    _record(position, qubit)
    if wires == "all":
        for qubit in sorted(site_qubits):
            _record(0, qubit)

    num_qubits = circuit.num_qubits
    gens = PauliList.from_symplectic(
        np.asarray(z_rows[::-1], dtype=bool).reshape(len(z_rows), num_qubits),
        np.asarray(x_rows[::-1], dtype=bool).reshape(len(x_rows), num_qubits),
    )
    return np.asarray(positions[::-1], dtype=int), np.asarray(qubits[::-1], dtype=int), gens, suffix


def _prune_to_fixpoint(
    gens: PauliList,
    input_diagonal: np.ndarray,
    output_diagonal: np.ndarray,
    active: np.ndarray,
) -> None:
    """Iterate the pruning rewrites of :meth:`.CheckedCircuit.dope` on ``active`` in place.

    Candidates must be time-sorted; "previous" and "following" refer to active candidates.
    """
    changed = True
    while changed:
        changed = False
        for i in np.flatnonzero(active & (input_diagonal | output_diagonal)):
            row = ~gens.commutes(gens[i])
            if (input_diagonal[i] and not (row[:i] & active[:i]).any()) or (
                output_diagonal[i] and not (row[i + 1 :] & active[i + 1 :]).any()
            ):
                active[i] = False
                changed = True
        groups: dict[bytes, list[int]] = {}
        for i in np.flatnonzero(active):
            groups.setdefault(gens.x[i].tobytes() + gens.z[i].tobytes(), []).append(int(i))
        for members in groups.values():
            # Equal generators share anticommutation rows, so one row serves the whole group.
            row = ~gens.commutes(gens[members[0]])
            anchor = members[0]
            for member in members[1:]:
                if (row[anchor + 1 : member] & active[anchor + 1 : member]).any():
                    anchor = member
                else:
                    active[member] = False
                    changed = True
