import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute, Aer
from qiskit.circuit.library.standard_gates import  MCXGate
simulator = Aer.get_backend('qasm_simulator')
qreg = QuantumRegister(6)
creg = ClassicalRegister(6)
qc = QuantumCircuit(qreg, creg)

n1= MCXGate(3, None, '000')
n2= MCXGate(3, None, '001')
n3= MCXGate(3, None, '100')
n4= MCXGate(3, None, '101')
n5= MCXGate(3, None, '110')
n6= MCXGate(3, None, '111')

n7= MCXGate(2, None, '00')
n8= MCXGate(2, None, '11')
qc.h(3)
qc.h(4)
qc.h(5)


qc.append(n1, [3,4,5,0])
qc.append(n2, [3,4,5,0])
qc.append(n3, [3,4,5,1])
qc.append(n4, [3,4,5,1])
qc.append(n5, [3,4,5,0])
qc.append(n5, [3,4,5,1])
qc.append(n6, [3,4,5,0])
qc.append(n6, [3,4,5,1])


# image operation
qc.barrier()
qc.swap(0,2)
qc.barrier()
qc.x(4)

figure =qc.draw("mpl")
figure.savefig('colorimage_circui.png', dpi=300)
plt.show()
qc.measure(qreg,creg)

job = execute(qc, simulator, shots=50000)
result = job.result()
counts = result.get_counts(qc)

qde = plot_histogram(counts ,figsize=(20,20))#filename='origin.png'
qde.subplots_adjust(left=0.215, right=0.46, top=0.71, bottom=0.535, hspace=0.2, wspace=0.2)
qde.savefig('colorimage_result_test.png', dpi=300)
plt.show()
