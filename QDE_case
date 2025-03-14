import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer
from qiskit.circuit.library.standard_gates import  MCXGate,ZGate
simulator = Aer.get_backend('qasm_simulator')
qreg = QuantumRegister(4)
creg = ClassicalRegister(4)
qc = QuantumCircuit(qreg, creg)

n1= MCXGate(3, None, '011')
n2= MCXGate(3, None, '100')
n3= MCXGate(3, None, '110')
n4= MCXGate(3, None, '111')
qc.h(1)
qc.h(2)
qc.h(3)

qc.append(n1, [1,2,3,0])
qc.append(n2, [1,2,3,0])
qc.append(n3, [1,2,3,0])
qc.append(n4, [1,2,3,0])


figure =qc.draw("mpl")#,filename='non.png'
figure.savefig('case_circuit.png', dpi=300)
plt.show()
qc.measure(qreg,creg)

job = execute(qc, simulator, shots=50000)
result = job.result()
counts = result.get_counts(qc)

qde = plot_histogram(counts ,figsize=(20,20))#filename='origin.png'
qde.subplots_adjust(left=0.215, right=0.46, top=0.71, bottom=0.535, hspace=0.2, wspace=0.2)
qde.savefig('case_result.png', dpi=300)
plt.show()
