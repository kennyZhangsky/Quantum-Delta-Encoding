import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
import math



# noise comparison of QDE and BRQI and CE-NGQR, denoted as state, state2 and state3. detailed circuit construction please refer to their references.

# font size setting
fontsize_title = 26
fontsize_xlabel = 24
fontsize_ylabel = 24
fontsize_ticks = 20
fontsize_legend = 20



# qde state init
state = np.array([1 / math.sqrt(8), 0, 1 / math.sqrt(8), 0, 1 / math.sqrt(8), 1 / math.sqrt(8), 0, 0,
                  0, 1 / math.sqrt(8), 0, 1 / math.sqrt(8), 0, 0, 1 / math.sqrt(8), 1 / math.sqrt(8)])

# 
n_qubits1 = 4   
n_qubits2 = 8   
n_qubits3 = 8
# 
dev1 = qml.device("default.mixed", wires=n_qubits1)
dev2 = qml.device("default.mixed", wires=n_qubits2)
dev3 = qml.device("default.mixed", wires=n_qubits3)
#  state circuit
@qml.qnode(dev1)
def encode_state():
    qml.AmplitudeEmbedding(state, wires=range(n_qubits1), normalize=True)
    return qml.state()

# state2 circuit
@qml.qnode(dev2)
def encode_state2():
    for i in range(1, 7):
        qml.Hadamard(wires=i)
    # n1: '001100'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 1, 1, 0, 0])
    # n2: '001010'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 1, 0, 1, 0])
    # n3: '000110'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 0, 1, 1, 0])
    # n4: '001110'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 1, 1, 1, 0])
    # n5: '000001'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 0, 0, 0, 1])
    # n6: '001001'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 1, 0, 0, 1])
    # n7: '000101'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 0, 1, 0, 1])

    # n02: '100000'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 0, 0, 0, 0])
    # n12: '101100'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 1, 1, 0, 0])
    # n22: '101010'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 1, 0, 1, 0])
    # n32: '100110'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 0, 1, 1, 0])
    # n42: '101110'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 1, 1, 1, 0])
    # n52: '100001'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 0, 0, 0, 1])
    # n62: '101001'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 1, 0, 0, 1])
    # n72: '100101'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 0, 1, 0, 1])

    # n03: '011000'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 1, 0, 0, 0])
    # n13: '011100'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 1, 1, 0, 0])
    # n23: '011010'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 1, 0, 1, 0])
    # n33: '010110'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 0, 1, 1, 0])
    # n43: '011110'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 1, 1, 1, 0])
    # n53: '010001'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 0, 0, 0, 1])
    # n63: '011001'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 1, 0, 0, 1])
    # n73: '010101'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 0, 1, 0, 1])

    # n04: '110000'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 0, 0, 0, 0])
    # n041: '111000'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 1, 0, 0, 0])
    # n14: '111100'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 1, 1, 0, 0])
    # n24: '111010'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 1, 0, 1, 0])
    # n34: '110110'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 0, 1, 1, 0])
    # n44: '111110'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 1, 1, 1, 0])
    # n54: '110001'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 0, 0, 0, 1])
    # n64: '111001'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 1, 0, 0, 1])
    # n74: '110101'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 0, 1, 0, 1])


    # n1: '001100'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 0, 1, 1, 0, 0])
    # n2: '001010'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 0, 1, 0, 1, 0])
    # n3: '000110'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 0, 0, 1, 1, 0])
    # n4: '001110'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 0, 1, 1, 1, 0])
    # n5: '000001'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 0, 0, 0, 0, 1])
    # n6: '001001'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 0, 1, 0, 0, 1])
    # n7: '000101'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 0, 0, 1, 0, 1])

    # n02: '100000'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 0, 0, 0, 0, 0])
    # n12: '101100'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 0, 1, 1, 0, 0])
    # n22: '101010'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 0, 1, 0, 1, 0])
    # n32: '100110'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 0, 0, 1, 1, 0])
    # n42: '101110'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 0, 1, 1, 1, 0])
    # n52: '100001'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 0, 0, 0, 0, 1])
    # n62: '101001'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 0, 1, 0, 0, 1])
    # n72: '100101'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 0, 0, 1, 0, 1])

    # n03: '011000'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 1, 1, 0, 0, 0])
    # n13: '011100'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 1, 1, 1, 0, 0])
    # n23: '011010'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 1, 1, 0, 1, 0])
    # n33: '010110'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 1, 0, 1, 1, 0])
    # n43: '011110'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 1, 1, 1, 1, 0])
    # n53: '010001'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 1, 0, 0, 0, 1])
    # n63: '011001'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 1, 1, 0, 0, 1])
    # n73: '010101'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 1, 0, 1, 0, 1])

    # n04: '110000'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 1, 0, 0, 0, 0])
    # n041: '111000'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 1, 1, 0, 0, 0])
    # n14: '111100'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 1, 1, 1, 0, 0])
    # n24: '111010'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 1, 1, 0, 1, 0])
    # n34: '110110'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 1, 0, 1, 1, 0])
    # n44: '111110'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 1, 1, 1, 1, 0])
    # n54: '110001'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 1, 0, 0, 0, 1])
    # n64: '111001'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 1, 1, 0, 0, 1])
    # n74: '110101'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 1, 0, 1, 0, 1])
    return qml.state()

#  state3 的电路
@qml.qnode(dev3)
def encode_state3():
    for i in range(1, 6):
        qml.Hadamard(wires=i)
    # n1: '001100'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 1, 1, 0, 0])
    # n2: '001010'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 1, 0, 1, 0])
    # n3: '000110'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 0, 1, 1, 0])
    # n4: '001110'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 1, 1, 1, 0])
    # n5: '000001'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 0, 0, 0, 1])
    # n6: '001001'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 1, 0, 0, 1])
    # n7: '000101'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 0, 1, 0, 1])

    # n02: '100000'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 0, 0, 0, 0])
    # n12: '101100'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 1, 1, 0, 0])
    # n22: '101010'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 1, 0, 1, 0])
    # n32: '100110'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 0, 1, 1, 0])
    # n42: '101110'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 1, 1, 1, 0])
    # n52: '100001'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 0, 0, 0, 1])
    # n62: '101001'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 1, 0, 0, 1])
    # n72: '100101'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 0, 1, 0, 1])

    # n03: '011000'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 1, 0, 0, 0])
    # n13: '011100'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 1, 1, 0, 0])
    # n23: '011010'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 1, 0, 1, 0])
    # n33: '010110'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 0, 1, 1, 0])
    # n43: '011110'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 1, 1, 1, 0])
    # n53: '010001'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 0, 0, 0, 1])
    # n63: '011001'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 1, 0, 0, 1])
    # n73: '010101'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 0, 1, 0, 1])

    # n04: '110000'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 0, 0, 0, 0])
    # n041: '111000'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 1, 0, 0, 0])
    # n14: '111100'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 1, 1, 0, 0])
    # n24: '111010'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 1, 0, 1, 0])
    # n34: '110110'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 0, 1, 1, 0])
    # n44: '111110'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 1, 1, 1, 0])
    # n54: '110001'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 0, 0, 0, 1])
    # n64: '111001'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 1, 0, 0, 1])
    # n74: '110101'
    qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 0, 1, 0, 1])
    qml.RY(np.pi/2,wires=[7])
    return qml.state()

# flatten
original_state = encode_state().flatten()
original_state2 = encode_state2().flatten()
original_state3 = encode_state3().flatten()

# noise range
noise_levels = np.linspace(0, 1, 20)

# 
phase_shift_fidelity_state1 = []
phase_shift_fidelity_state2 = []
phase_shift_fidelity_state3 = []

amplitude_damping_fidelity_state1 = []
amplitude_damping_fidelity_state2 = []
amplitude_damping_fidelity_state3 = []

depolarizing_fidelity_state1 = []
depolarizing_fidelity_state2 = []
depolarizing_fidelity_state3 = []

# ======================
# Phase Shift 
# ======================
for level in noise_levels:
    #  state1 (dev1)
    @qml.qnode(dev1)
    def phase_shift_noise_state1():
        qml.AmplitudeEmbedding(state, wires=range(n_qubits1), normalize=True)
        for i in range(n_qubits1):
            qml.PhaseShift(level, wires=i)
        return qml.state()
    noisy_state1 = phase_shift_noise_state1().flatten()
    fidelity1 = np.abs(np.dot(np.conj(original_state), noisy_state1)) ** 2
    phase_shift_fidelity_state1.append(fidelity1)

    # 
    @qml.qnode(dev2)
    def phase_shift_noise_state2():
        for i in range(1, 7):
            qml.Hadamard(wires=i)
        # n1: '001100'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 1, 1, 0, 0])
        # n2: '001010'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 1, 0, 1, 0])
        # n3: '000110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 0, 1, 1, 0])
        # n4: '001110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 1, 1, 1, 0])
        # n5: '000001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 0, 0, 0, 1])
        # n6: '001001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 1, 0, 0, 1])
        # n7: '000101'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 0, 1, 0, 1])

        # n02: '100000'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 0, 0, 0, 0])
        # n12: '101100'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 1, 1, 0, 0])
        # n22: '101010'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 1, 0, 1, 0])
        # n32: '100110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 0, 1, 1, 0])
        # n42: '101110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 1, 1, 1, 0])
        # n52: '100001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 0, 0, 0, 1])
        # n62: '101001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 1, 0, 0, 1])
        # n72: '100101'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 0, 1, 0, 1])

        # n03: '011000'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 1, 0, 0, 0])
        # n13: '011100'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 1, 1, 0, 0])
        # n23: '011010'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 1, 0, 1, 0])
        # n33: '010110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 0, 1, 1, 0])
        # n43: '011110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 1, 1, 1, 0])
        # n53: '010001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 0, 0, 0, 1])
        # n63: '011001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 1, 0, 0, 1])
        # n73: '010101'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 0, 1, 0, 1])

        # n04: '110000'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 0, 0, 0, 0])
        # n041: '111000'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 1, 0, 0, 0])
        # n14: '111100'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 1, 1, 0, 0])
        # n24: '111010'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 1, 0, 1, 0])
        # n34: '110110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 0, 1, 1, 0])
        # n44: '111110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 1, 1, 1, 0])
        # n54: '110001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 0, 0, 0, 1])
        # n64: '111001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 1, 0, 0, 1])
        # n74: '110101'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 0, 1, 0, 1])

        # n1: '001100'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 0, 1, 1, 0, 0])
        # n2: '001010'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 0, 1, 0, 1, 0])
        # n3: '000110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 0, 0, 1, 1, 0])
        # n4: '001110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 0, 1, 1, 1, 0])
        # n5: '000001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 0, 0, 0, 0, 1])
        # n6: '001001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 0, 1, 0, 0, 1])
        # n7: '000101'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 0, 0, 1, 0, 1])

        # n02: '100000'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 0, 0, 0, 0, 0])
        # n12: '101100'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 0, 1, 1, 0, 0])
        # n22: '101010'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 0, 1, 0, 1, 0])
        # n32: '100110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 0, 0, 1, 1, 0])
        # n42: '101110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 0, 1, 1, 1, 0])
        # n52: '100001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 0, 0, 0, 0, 1])
        # n62: '101001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 0, 1, 0, 0, 1])
        # n72: '100101'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 0, 0, 1, 0, 1])

        # n03: '011000'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 1, 1, 0, 0, 0])
        # n13: '011100'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 1, 1, 1, 0, 0])
        # n23: '011010'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 1, 1, 0, 1, 0])
        # n33: '010110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 1, 0, 1, 1, 0])
        # n43: '011110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 1, 1, 1, 1, 0])
        # n53: '010001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 1, 0, 0, 0, 1])
        # n63: '011001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 1, 1, 0, 0, 1])
        # n73: '010101'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 1, 0, 1, 0, 1])

        # n04: '110000'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 1, 0, 0, 0, 0])
        # n041: '111000'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 1, 1, 0, 0, 0])
        # n14: '111100'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 1, 1, 1, 0, 0])
        # n24: '111010'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 1, 1, 0, 1, 0])
        # n34: '110110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 1, 0, 1, 1, 0])
        # n44: '111110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 1, 1, 1, 1, 0])
        # n54: '110001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 1, 0, 0, 0, 1])
        # n64: '111001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 1, 1, 0, 0, 1])
        # n74: '110101'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 1, 0, 1, 0, 1])
        for i in range(n_qubits2):
            qml.PhaseShift(level, wires=i)
        return qml.state()
    noisy_state2 = phase_shift_noise_state2().flatten()
    fidelity2 = np.abs(np.dot(np.conj(original_state2), noisy_state2)) ** 2
    phase_shift_fidelity_state2.append(fidelity2)

    #  state3 (dev2) 
    @qml.qnode(dev3)
    def phase_shift_noise_state3():
        for i in range(1, 6):
            qml.Hadamard(wires=i)
        # n1: '001100'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 1, 1, 0, 0])
        # n2: '001010'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 1, 0, 1, 0])
        # n3: '000110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 0, 1, 1, 0])
        # n4: '001110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 1, 1, 1, 0])
        # n5: '000001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 0, 0, 0, 1])
        # n6: '001001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 1, 0, 0, 1])
        # n7: '000101'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 0, 1, 0, 1])

        # n02: '100000'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 0, 0, 0, 0])
        # n12: '101100'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 1, 1, 0, 0])
        # n22: '101010'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 1, 0, 1, 0])
        # n32: '100110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 0, 1, 1, 0])
        # n42: '101110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 1, 1, 1, 0])
        # n52: '100001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 0, 0, 0, 1])
        # n62: '101001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 1, 0, 0, 1])
        # n72: '100101'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 0, 1, 0, 1])

        # n03: '011000'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 1, 0, 0, 0])
        # n13: '011100'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 1, 1, 0, 0])
        # n23: '011010'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 1, 0, 1, 0])
        # n33: '010110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 0, 1, 1, 0])
        # n43: '011110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 1, 1, 1, 0])
        # n53: '010001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 0, 0, 0, 1])
        # n63: '011001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 1, 0, 0, 1])
        # n73: '010101'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 0, 1, 0, 1])

        # n04: '110000'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 0, 0, 0, 0])
        # n041: '111000'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 1, 0, 0, 0])
        # n14: '111100'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 1, 1, 0, 0])
        # n24: '111010'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 1, 0, 1, 0])
        # n34: '110110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 0, 1, 1, 0])
        # n44: '111110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 1, 1, 1, 0])
        # n54: '110001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 0, 0, 0, 1])
        # n64: '111001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 1, 0, 0, 1])
        # n74: '110101'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 0, 1, 0, 1])
        qml.RY(np.pi / 2, wires=[7])
        for i in range(n_qubits3):
            qml.PhaseShift(level, wires=i)
        return qml.state()
    noisy_state3 = phase_shift_noise_state3().flatten()
    fidelity3 = np.abs(np.dot(np.conj(original_state3), noisy_state3)) ** 2
    phase_shift_fidelity_state3.append(fidelity3)

# ======================
# Amplitude Damping 
# ======================
for level in noise_levels:
    #  state1 (dev1)
    @qml.qnode(dev1)
    def amplitude_damping_noise_state1():
        qml.AmplitudeEmbedding(state, wires=range(n_qubits1), normalize=True)
        for i in range(n_qubits1):
            qml.AmplitudeDamping(level, wires=i)
        return qml.state()
    noisy_state1 = amplitude_damping_noise_state1().flatten()
    fidelity1 = np.abs(np.dot(np.conj(original_state), noisy_state1)) ** 2
    amplitude_damping_fidelity_state1.append(fidelity1)

    #  state2 (dev2)
    @qml.qnode(dev2)
    def amplitude_damping_noise_state2():
        for i in range(1, 7):
            qml.Hadamard(wires=i)
        # n1: '001100'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 1, 1, 0, 0])
        # n2: '001010'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 1, 0, 1, 0])
        # n3: '000110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 0, 1, 1, 0])
        # n4: '001110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 1, 1, 1, 0])
        # n5: '000001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 0, 0, 0, 1])
        # n6: '001001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 1, 0, 0, 1])
        # n7: '000101'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 0, 1, 0, 1])

        # n02: '100000'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 0, 0, 0, 0])
        # n12: '101100'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 1, 1, 0, 0])
        # n22: '101010'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 1, 0, 1, 0])
        # n32: '100110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 0, 1, 1, 0])
        # n42: '101110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 1, 1, 1, 0])
        # n52: '100001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 0, 0, 0, 1])
        # n62: '101001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 1, 0, 0, 1])
        # n72: '100101'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 0, 1, 0, 1])

        # n03: '011000'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 1, 0, 0, 0])
        # n13: '011100'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 1, 1, 0, 0])
        # n23: '011010'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 1, 0, 1, 0])
        # n33: '010110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 0, 1, 1, 0])
        # n43: '011110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 1, 1, 1, 0])
        # n53: '010001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 0, 0, 0, 1])
        # n63: '011001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 1, 0, 0, 1])
        # n73: '010101'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 0, 1, 0, 1])

        # n04: '110000'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 0, 0, 0, 0])
        # n041: '111000'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 1, 0, 0, 0])
        # n14: '111100'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 1, 1, 0, 0])
        # n24: '111010'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 1, 0, 1, 0])
        # n34: '110110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 0, 1, 1, 0])
        # n44: '111110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 1, 1, 1, 0])
        # n54: '110001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 0, 0, 0, 1])
        # n64: '111001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 1, 0, 0, 1])
        # n74: '110101'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 0, 1, 0, 1])

        # n1: '001100'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 0, 1, 1, 0, 0])
        # n2: '001010'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 0, 1, 0, 1, 0])
        # n3: '000110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 0, 0, 1, 1, 0])
        # n4: '001110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 0, 1, 1, 1, 0])
        # n5: '000001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 0, 0, 0, 0, 1])
        # n6: '001001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 0, 1, 0, 0, 1])
        # n7: '000101'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 0, 0, 1, 0, 1])

        # n02: '100000'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 0, 0, 0, 0, 0])
        # n12: '101100'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 0, 1, 1, 0, 0])
        # n22: '101010'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 0, 1, 0, 1, 0])
        # n32: '100110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 0, 0, 1, 1, 0])
        # n42: '101110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 0, 1, 1, 1, 0])
        # n52: '100001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 0, 0, 0, 0, 1])
        # n62: '101001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 0, 1, 0, 0, 1])
        # n72: '100101'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 0, 0, 1, 0, 1])

        # n03: '011000'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 1, 1, 0, 0, 0])
        # n13: '011100'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 1, 1, 1, 0, 0])
        # n23: '011010'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 1, 1, 0, 1, 0])
        # n33: '010110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 1, 0, 1, 1, 0])
        # n43: '011110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 1, 1, 1, 1, 0])
        # n53: '010001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 1, 0, 0, 0, 1])
        # n63: '011001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 1, 1, 0, 0, 1])
        # n73: '010101'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 1, 0, 1, 0, 1])

        # n04: '110000'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 1, 0, 0, 0, 0])
        # n041: '111000'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 1, 1, 0, 0, 0])
        # n14: '111100'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 1, 1, 1, 0, 0])
        # n24: '111010'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 1, 1, 0, 1, 0])
        # n34: '110110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 1, 0, 1, 1, 0])
        # n44: '111110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 1, 1, 1, 1, 0])
        # n54: '110001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 1, 0, 0, 0, 1])
        # n64: '111001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 1, 1, 0, 0, 1])
        # n74: '110101'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 1, 0, 1, 0, 1])
        for i in range(n_qubits2):
            qml.AmplitudeDamping(level, wires=i)
        return qml.state()
    noisy_state2 = amplitude_damping_noise_state2().flatten()
    fidelity2 = np.abs(np.dot(np.conj(original_state2), noisy_state2)) ** 2
    amplitude_damping_fidelity_state2.append(fidelity2)

    # 针对 state3 (dev2)
    @qml.qnode(dev3)
    def amplitude_damping_noise_state3():
        for i in range(1, 6):
            qml.Hadamard(wires=i)
        # n1: '001100'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 1, 1, 0, 0])
        # n2: '001010'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 1, 0, 1, 0])
        # n3: '000110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 0, 1, 1, 0])
        # n4: '001110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 1, 1, 1, 0])
        # n5: '000001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 0, 0, 0, 1])
        # n6: '001001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 1, 0, 0, 1])
        # n7: '000101'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 0, 1, 0, 1])

        # n02: '100000'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 0, 0, 0, 0])
        # n12: '101100'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 1, 1, 0, 0])
        # n22: '101010'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 1, 0, 1, 0])
        # n32: '100110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 0, 1, 1, 0])
        # n42: '101110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 1, 1, 1, 0])
        # n52: '100001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 0, 0, 0, 1])
        # n62: '101001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 1, 0, 0, 1])
        # n72: '100101'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 0, 1, 0, 1])

        # n03: '011000'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 1, 0, 0, 0])
        # n13: '011100'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 1, 1, 0, 0])
        # n23: '011010'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 1, 0, 1, 0])
        # n33: '010110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 0, 1, 1, 0])
        # n43: '011110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 1, 1, 1, 0])
        # n53: '010001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 0, 0, 0, 1])
        # n63: '011001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 1, 0, 0, 1])
        # n73: '010101'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 0, 1, 0, 1])

        # n04: '110000'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 0, 0, 0, 0])
        # n041: '111000'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 1, 0, 0, 0])
        # n14: '111100'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 1, 1, 0, 0])
        # n24: '111010'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 1, 0, 1, 0])
        # n34: '110110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 0, 1, 1, 0])
        # n44: '111110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 1, 1, 1, 0])
        # n54: '110001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 0, 0, 0, 1])
        # n64: '111001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 1, 0, 0, 1])
        # n74: '110101'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 0, 1, 0, 1])
        qml.RY(np.pi / 2, wires=[7])
        for i in range(n_qubits3):
            qml.AmplitudeDamping(level, wires=i)
        return qml.state()
    noisy_state3 = amplitude_damping_noise_state3().flatten()
    fidelity3 = np.abs(np.dot(np.conj(original_state3), noisy_state3)) ** 2
    amplitude_damping_fidelity_state3.append(fidelity3)

# ======================
# Depolarizing Channel 
# ======================
#noise range rescale 0.5
noise_levels1 = np.linspace(0, 0.5, 20)
for level in noise_levels1:
    #  state1 (dev1)
    @qml.qnode(dev1)
    def depolarizing_noise_state1():
        qml.AmplitudeEmbedding(state, wires=range(n_qubits1), normalize=True)
        for i in range(n_qubits1):
            qml.DepolarizingChannel(level, wires=i)
        return qml.state()
    noisy_state1 = depolarizing_noise_state1().flatten()
    fidelity1 = np.abs(np.dot(np.conj(original_state), noisy_state1)) ** 2
    depolarizing_fidelity_state1.append(fidelity1)

    #  state2 (dev2)
    @qml.qnode(dev2)
    def depolarizing_noise_state2():
        for i in range(1, 7):
            qml.Hadamard(wires=i)
        # n1: '001100'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 1, 1, 0, 0])
        # n2: '001010'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 1, 0, 1, 0])
        # n3: '000110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 0, 1, 1, 0])
        # n4: '001110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 1, 1, 1, 0])
        # n5: '000001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 0, 0, 0, 1])
        # n6: '001001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 1, 0, 0, 1])
        # n7: '000101'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 0, 1, 0, 1])

        # n02: '100000'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 0, 0, 0, 0])
        # n12: '101100'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 1, 1, 0, 0])
        # n22: '101010'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 1, 0, 1, 0])
        # n32: '100110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 0, 1, 1, 0])
        # n42: '101110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 1, 1, 1, 0])
        # n52: '100001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 0, 0, 0, 1])
        # n62: '101001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 1, 0, 0, 1])
        # n72: '100101'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 0, 1, 0, 1])

        # n03: '011000'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 1, 0, 0, 0])
        # n13: '011100'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 1, 1, 0, 0])
        # n23: '011010'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 1, 0, 1, 0])
        # n33: '010110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 0, 1, 1, 0])
        # n43: '011110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 1, 1, 1, 0])
        # n53: '010001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 0, 0, 0, 1])
        # n63: '011001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 1, 0, 0, 1])
        # n73: '010101'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 0, 1, 0, 1])

        # n04: '110000'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 0, 0, 0, 0])
        # n041: '111000'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 1, 0, 0, 0])
        # n14: '111100'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 1, 1, 0, 0])
        # n24: '111010'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 1, 0, 1, 0])
        # n34: '110110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 0, 1, 1, 0])
        # n44: '111110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 1, 1, 1, 0])
        # n54: '110001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 0, 0, 0, 1])
        # n64: '111001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 1, 0, 0, 1])
        # n74: '110101'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 0, 1, 0, 1])

        # n1: '001100'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 0, 1, 1, 0, 0])
        # n2: '001010'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 0, 1, 0, 1, 0])
        # n3: '000110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 0, 0, 1, 1, 0])
        # n4: '001110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 0, 1, 1, 1, 0])
        # n5: '000001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 0, 0, 0, 0, 1])
        # n6: '001001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 0, 1, 0, 0, 1])
        # n7: '000101'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 0, 0, 1, 0, 1])

        # n02: '100000'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 0, 0, 0, 0, 0])
        # n12: '101100'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 0, 1, 1, 0, 0])
        # n22: '101010'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 0, 1, 0, 1, 0])
        # n32: '100110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 0, 0, 1, 1, 0])
        # n42: '101110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 0, 1, 1, 1, 0])
        # n52: '100001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 0, 0, 0, 0, 1])
        # n62: '101001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 0, 1, 0, 0, 1])
        # n72: '100101'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 0, 0, 1, 0, 1])

        # n03: '011000'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 1, 1, 0, 0, 0])
        # n13: '011100'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 1, 1, 1, 0, 0])
        # n23: '011010'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 1, 1, 0, 1, 0])
        # n33: '010110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 1, 0, 1, 1, 0])
        # n43: '011110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 1, 1, 1, 1, 0])
        # n53: '010001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 1, 0, 0, 0, 1])
        # n63: '011001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 1, 1, 0, 0, 1])
        # n73: '010101'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[0, 1, 0, 1, 0, 1])

        # n04: '110000'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 1, 0, 0, 0, 0])
        # n041: '111000'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 1, 1, 0, 0, 0])
        # n14: '111100'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 1, 1, 1, 0, 0])
        # n24: '111010'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 1, 1, 0, 1, 0])
        # n34: '110110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 1, 0, 1, 1, 0])
        # n44: '111110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 1, 1, 1, 1, 0])
        # n54: '110001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 1, 0, 0, 0, 1])
        # n64: '111001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 1, 1, 0, 0, 1])
        # n74: '110101'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[7], control_values=[1, 1, 0, 1, 0, 1])
        for i in range(n_qubits2):
            qml.DepolarizingChannel(level, wires=i)
        return qml.state()
    noisy_state2 = depolarizing_noise_state2().flatten()
    fidelity2 = np.abs(np.dot(np.conj(original_state2), noisy_state2)) ** 2
    depolarizing_fidelity_state2.append(fidelity2)

    #  state3 (dev2)
    @qml.qnode(dev3)
    def depolarizing_noise_state3():
        for i in range(1, 6):
            qml.Hadamard(wires=i)
        # n1: '001100'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 1, 1, 0, 0])
        # n2: '001010'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 1, 0, 1, 0])
        # n3: '000110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 0, 1, 1, 0])
        # n4: '001110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 1, 1, 1, 0])
        # n5: '000001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 0, 0, 0, 1])
        # n6: '001001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 1, 0, 0, 1])
        # n7: '000101'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 0, 0, 1, 0, 1])

        # n02: '100000'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 0, 0, 0, 0])
        # n12: '101100'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 1, 1, 0, 0])
        # n22: '101010'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 1, 0, 1, 0])
        # n32: '100110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 0, 1, 1, 0])
        # n42: '101110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 1, 1, 1, 0])
        # n52: '100001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 0, 0, 0, 1])
        # n62: '101001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 1, 0, 0, 1])
        # n72: '100101'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 0, 0, 1, 0, 1])

        # n03: '011000'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 1, 0, 0, 0])
        # n13: '011100'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 1, 1, 0, 0])
        # n23: '011010'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 1, 0, 1, 0])
        # n33: '010110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 0, 1, 1, 0])
        # n43: '011110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 1, 1, 1, 0])
        # n53: '010001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 0, 0, 0, 1])
        # n63: '011001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 1, 0, 0, 1])
        # n73: '010101'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[0, 1, 0, 1, 0, 1])

        # n04: '110000'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 0, 0, 0, 0])
        # n041: '111000'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 1, 0, 0, 0])
        # n14: '111100'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 1, 1, 0, 0])
        # n24: '111010'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 1, 0, 1, 0])
        # n34: '110110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 0, 1, 1, 0])
        # n44: '111110'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 1, 1, 1, 0])
        # n54: '110001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 0, 0, 0, 1])
        # n64: '111001'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 1, 0, 0, 1])
        # n74: '110101'
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=[0], control_values=[1, 1, 0, 1, 0, 1])
        qml.RY(np.pi / 2, wires=[7])
        for i in range(n_qubits3):
            qml.DepolarizingChannel(level, wires=i)
        return qml.state()
    noisy_state3 = depolarizing_noise_state3().flatten()
    fidelity3 = np.abs(np.dot(np.conj(original_state3), noisy_state3)) ** 2
    depolarizing_fidelity_state3.append(fidelity3)

# ======================
# saving Phase Shift pics
# ======================
plt.figure(figsize=(10, 6))
plt.plot(noise_levels, phase_shift_fidelity_state1, label="QDE", marker='o', linestyle='-', color='#264653')
plt.plot(noise_levels, phase_shift_fidelity_state2, label="PE-NGQR", marker='s', linestyle='-', color='#E66F51')
plt.plot(noise_levels, phase_shift_fidelity_state3, label="MQIR", marker='d', linestyle='-', color='#2A9D8E')
plt.xlabel("Noise Strength", fontsize=fontsize_xlabel)
plt.ylabel("Fidelity", fontsize=fontsize_ylabel)
plt.title("Phase Shift on QDE, MQIR and PE-NGQR", fontsize=fontsize_title)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
plt.legend(fontsize=fontsize_legend)
plt.savefig("phase_shift_noise.png", dpi=300, bbox_inches='tight')
plt.show()

# ======================
# saving Amplitude Damping pics
# ======================
plt.figure(figsize=(10, 6))
plt.plot(noise_levels, amplitude_damping_fidelity_state1, label="QDE", marker='o', linestyle='-', color='#264653')
plt.plot(noise_levels, amplitude_damping_fidelity_state2, label="PE-NGQR", marker='s', linestyle='-', color='#E66F51')
plt.plot(noise_levels, amplitude_damping_fidelity_state3, label="MQIR", marker='d', linestyle='-', color='#2A9D8E')
plt.xlabel("Noise Strength", fontsize=fontsize_xlabel)
plt.ylabel("Fidelity", fontsize=fontsize_ylabel)
plt.title("Amplitude Damping on QDE, MQIR and PE-NGQR", fontsize=fontsize_title)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
plt.legend(fontsize=fontsize_legend)
plt.savefig("amplitude_damping_noise.png", dpi=300, bbox_inches='tight')
plt.show()

# ======================
# saving Depolarizing Channel pics
# ======================
plt.figure(figsize=(10, 6))
plt.plot(noise_levels1, depolarizing_fidelity_state1, label="QDE", marker='o', linestyle='-', color='#264653')
plt.plot(noise_levels1, depolarizing_fidelity_state2, label="PE-NGQR", marker='s', linestyle='-', color='#E66F51')
plt.plot(noise_levels1, depolarizing_fidelity_state3, label="MQIR", marker='d', linestyle='-', color='#2A9D8E')
plt.xlabel("Noise Strength", fontsize=fontsize_xlabel)
plt.ylabel("Fidelity", fontsize=fontsize_ylabel)
plt.title("Depolarizing Channel on QDE, MQIR and PE-NGQR", fontsize=fontsize_title)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
plt.legend(fontsize=fontsize_legend)
plt.savefig("depolarizing_noise.png", dpi=300, bbox_inches='tight')
plt.show()
