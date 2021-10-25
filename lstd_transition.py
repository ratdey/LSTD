import numpy as np
import glob
import json
import sys
import os


'''
[github.com/ratdey]

LSTD(lambda) based on given transition tree

run:	- add -d argument to run using default parameter values
			- run from the location containing script
		- run without arguments for manual input
		- STATEDIM and envName are environment specific, can not be changed through CLI
output:	- prints result and beta vector
		- saves beta vector in 'beta.txt'
'''


'''
PARAMETERS
'''
# np.set_printoptions(threshold=sys.maxsize)
GAMMA = 1
LAMBDA = 0.9

F_BASE = 3 	# basis vector will have length F_BASE * stateDimension
STATEDIM = 4 	# for 2x2 state feature matrix
EPCOUNT = 1	# number of episodes to run

DIRPATH = os.getcwd() + "/rl_trees/" # data folder path wrt script directory
BETAFILEPATH = "./beta.txt"

'''
Environment class
	- reads all json files from DIRPATH
	- feature matrices are flattened into lists
	- file structure : 	- node_id : int
						- state_features : 2x2 double [* in some files 'state features', workaround added]
						- actions : list
							- features : 2x2 double
							- reward : int
							- result_node : int
'''

class Environment:
	def __init__(self):
		self.stateDim = STATEDIM
		self.numFiles = 0
		self.files = []
		# data to be read from file
		self.stateTransition = {}
		self.transitionReward = {}
		self.stateData = {}
		#procedures
		self.getFiles()

	# generate list of files to read
	def getFiles(self):
		for fileName in glob.glob(os.path.join(DIRPATH, '*.json')):
			self.files.append(fileName)
			self.numFiles += 1
		if self.numFiles < 1:
			sys.exit('No file found')

	def cleanPrevData(self):
		self.stateTransition = {}
		self.transitionReward = {}
		self.stateData = {}
			
	# load data from a file
	def parseFile(self, fileNum):
		self.cleanPrevData()
		'''
		workaround code for feature_id selection
		'''
		featureID = 'state_features'
		isFIDUnset = True

		file = open(self.files[fileNum], 'r')
		while True:
			line = file.readline()
			if not line:
				break
			data = json.loads(line)
			nodeId = data['node_id']

			if isFIDUnset:
				if featureID not in data:
					featureID = 'state features'
				isFIDUnset = False

			stateFeatures = np.array(data[featureID]).flatten()
			self.stateData[nodeId] = stateFeatures
			self.stateTransition[nodeId] = []
			self.transitionReward[nodeId] = []
			for action in data['actions']:
				self.stateTransition[nodeId].append(action['result_node'])
				self.transitionReward[nodeId].append(action['reward'])
		file.close()


'''
Agent class
'''

class LSTD:
	def __init__(self, stateDim=4):
		self.basis = F_BASE**stateDim
		self.A = np.zeros((self.basis, self.basis))
		self.B = np.zeros(self.basis)
		self.E = np.zeros(self.basis)
		self.beta = None
		'''
		generate matrix for fourier transform with base F_BASE
		e.g. 	0000
				1000
				0100
				1100
		'''
		self.FMat = np.zeros((self.basis, stateDim))
		self.FRow = np.zeros(stateDim)
		for x in range(self.basis):
			self.FMat[x] = self.FRow
			self.getNextFRow(stateDim)

	def getNextFRow(self, stateDim):
		for i in range(stateDim):
			self.FRow[i] += 1
			if (self.FRow[i] <= F_BASE - 1):
				break
			self.FRow[i] = 0
		return

	# returns phi(s)
	def getFVec(self, state):
		return np.cos(np.dot(self.FMat, state) * np.pi)

	# simulates one step of the chain
	def update(self, curState, reward, nextState):
		curFVec = self.getFVec(curState)
		self.E = GAMMA * LAMBDA * self.E + curFVec
		if nextState is not None:
			nextFVec = self.getFVec(nextState)
			self.A += np.outer(self.E, (curFVec - GAMMA * nextFVec))
		else:
			self.A += np.outer(self.E, curFVec)
		self.B += self.E * reward
		return

	# reset
	def newEpisode(self):
		self.E = np.zeros(self.basis)
		return

	# returns Beta vector
	def getBeta(self):
		return np.dot(np.linalg.pinv(self.A), self.B)
	
	# calculates state value
	def estimateVal(self, state):
		if self.beta is None:
			self.beta = np.dot(np.linalg.pinv(self.A), self.B)
		state = np.array(state).flatten()
		phi = self.getFVec(state)
		return np.dot(self.beta, phi)



# input
def manual():
	print('Set parameters. Press enter to keep default value ->')
	global DIRPATH, LAMBDA, GAMMA, F_BASE
	print('Enter data directory (absolute path)')
	inp = input()
	if inp:
		DIRPATH = inp
	print('Set Lambda : default value is ', LAMBDA)
	inp = input()
	if inp:
		LAMBDA = float(inp)
	print('Set Gamma : default value is ', GAMMA)
	inp = input()
	if inp:
		GAMMA = float(inp)
	print('Set Fourier basis parameter F in integer. Vector length will be F^state_dimension. Default value ', F_BASE)
	inp = input()
	if inp:
		F_BASE = int(inp)
	print('Set number of episodes to run : Default value is ', EPCOUNT)
	inp = input()
	if inp:
		EPCOUNT = int(inp)
	return


def check():
	if not isinstance(F_BASE, int) or F_BASE < 1:
		sys.exit("Fourier basis parameter must be int > 0")
	if LAMBDA < 0 or LAMBDA > 1:
		sys.exit("invalid value of Lambda")
	if GAMMA < 0 or GAMMA > 1:
		sys.exit("invalid value of Gamma")

def runEnv(envName):
	env = None
	if envName is "json":
		env = Environment()
	else:
		print("Select proper environment")
		return
	td = LSTD(env.stateDim)
	result = np.zeros(EPCOUNT)
	for ep in range(EPCOUNT):
		print('running ep ', ep + 1 , ' of ', EPCOUNT)
		td.newEpisode()
		for fileNum in range(env.numFiles):
			result[ep] = 0
			curGamma = 1
			env.parseFile(fileNum)
			for curState, transition in env.stateTransition.items():
				for index, nextState in enumerate(transition):
					reward = env.transitionReward[curState][index]
					result[ep] += (curGamma * reward)
					curGamma *= GAMMA
					if nextState in env.stateData:
						td.update(env.stateData[curState], reward, env.stateData[nextState])
					else:
						td.update(env.stateData[curState], reward, None)

	# output
	np.savetxt(BETAFILEPATH, td.getBeta())
	
	print('\nResult:\n')
	print(result)
	print('\nBeta value:\n')
	print(td.getBeta())
	return
	

if __name__ == '__main__':
	if len(sys.argv) == 1:
		manual()
	check()
	runEnv("json")


