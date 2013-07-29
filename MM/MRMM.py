from mrjob.job import MRJob
from mrjob.util import read_input
from mrjob.protocol import PickleValueProtocol, JSONProtocol
import numpy as np
from StringIO import *
import json

def jsonify(a, b):
    assert a.shape[1] == b.shape[0]
    it = np.nditer(a, flags=['multi_index'])
    while not it.finished:
	yield json.dumps((0,)+it.multi_index)+'\t'+json.dumps(int(it[0]))
	it.iternext()
    it = np.nditer(b, flags=['multi_index'])
    while not it.finished:
	yield json.dumps((1,)+it.multi_index)+'\t'+json.dumps(int(it[0]))
	it.iternext()

class MRMM(MRJob):
    INPUT_PROTOCOL = JSONProtocol
    INTERNAL_PROTOCOL = JSONProtocol
    OUTPUT_PROTOCOL = JSONProtocol
    def configure_options(self):
	super(MRMM, self).configure_options()
	self.add_passthrough_option('--Idim', type='int')
	self.add_passthrough_option('--Kdim', type='int')
	self.add_passthrough_option('--Jdim', type='int')
	self.add_passthrough_option('--IBdim', type='int', default=2)
	self.add_passthrough_option('--KBdim', type='int', default=2)
	self.add_passthrough_option('--JBdim', type='int', default=2)

    def load_options(self, args):
	super(MRMM, self).load_options(args)
	self.I = self.options.Idim
	self.K = self.options.Kdim
	self.J = self.options.Jdim
	self.IB = self.options.IBdim
	self.KB = self.options.KBdim
	self.JB = self.options.JBdim
	self.NIB = (self.I-1)/(self.IB)
	self.NKB = (self.K-1)/(self.KB)
	self.NJB = (self.J-1)/(self.JB)

    def mapper(self, key, value):
	if key[0] == 0:
	    matrix, i, k = key
	    for jb in range(self.NJB+1):
		yield (i/self.IB, k/self.KB, jb), ((i % self.IB), (k % self.KB), value, matrix)
	elif key[0] == 1:
	    matrix, k, j = key
	    for ib in range(self.NIB+1):
		yield (ib, k/self.KB, j/self.JB), ((k % self.KB), (j % self.JB), value, matrix)

    def reducer(self, key, value):
	ib, kb, jb = key
	self.imax, self.kmax, self.jmax = 0, 0, 0
	self.A = np.zeros((self.IB, self.KB))
	self.A.fill(np.nan)
	self.B = np.zeros((self.KB, self.JB))
	self.B.fill(np.nan)
	for item in value:
	    if item[-1] == 0:
		i, k, v, m = item
		if i > self.imax: self.imax = i
		if k > self.kmax: self.kmax = k
		self.A[i,k] = v
	    elif item[-1] == 1:
		k, j, v, m = item
		if k > self.kmax: self.kmax = k
		if j > self.jmax: self.jmax = j
		self.B[k,j] = v
	ibase = ib*self.IB
	jbase = jb*self.JB
	for i in range(self.imax+1):
	    for j in range(self.jmax+1):
		sum = 0
		for k in range(self.kmax+1): 
		    sum += self.A[i,k]*self.B[k,j]
		if not np.isnan(sum):
		    yield ((ibase+i, jbase+j), sum)

def MRdot(A, B, target='inline'):
    assert target in ['inline', 'local', 'emr']
    assert A.shape[1] == B.shape[0]
    mr_job = MRMM(args=['-r', target, '--Idim', str(A.shape[0]), '--Kdim', str(A.shape[1]), '--Jdim', str(B.shape[1])])
    mr_job.stdin = StringIO('\n'.join(list(jsonify(A,B))))
    output = np.zeros((A.shape[0],B.shape[1]))
    with mr_job.make_runner() as runner:
	runner.run()
	for line in runner.stream_output():
	    key, value = mr_job.parse_output_line(line)
	    output[tuple(key)] += value
    return output

if __name__ == '__main__':
    MRMM.run()
