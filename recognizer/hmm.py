import numpy as np

# default HMM
WORDS = { 
'name':['sil','oh','zero','one','two','three','four','five','six','seven','eight','nine'],
'size':[1,    3,   15,    12,   6,    9,      9,     9,     12,   15,     6,      9 ],
'gram':[100,  100, 100,   100,  100,  100,    100,   100,   10,   100,    100,    100] }


# HILFSFUNKTIONEN
def limLog( x ):
    MINLOG = 1e-100
    return np.log( np.maximum( x, MINLOG ) )

def column(ary,col):
    colAry = []
    for row in ary:
        colAry.append(row[col])
    return colAry

def viterbi( logLike, logPi, logA ):

    viterbiMatrix = np.zeros( ( len(logLike),len(logPi) ) ) # (Beobachtungen, Zustände i)
    pathMatrix = np.zeros((len(logLike),len(logPi)))
    number_of_states = len(logPi)
    seqLen = len(logLike)

    for st in range(number_of_states):
        
        viterbiMatrix[0][st] = logPi[st]+logLike[0][st]
        pathMatrix[0][st] = -1
    
    
    for t in range(1,seqLen):
        for j in range(number_of_states):
            viterbiMatrix[t][j] = max( np.array(viterbiMatrix[t-1]) + np.array(column(logA,j)) ) + logLike[t][j]
            pathMatrix[t][j] = np.argmax( np.array(viterbiMatrix[t-1]) + np.array(column(logA,j)) ) # Zustandsindex logPi
    
    observationProbability = max(viterbiMatrix[-1])
    lastState = int (np.argmax( np.array(viterbiMatrix[-1]) ) )
    
    stateSequence = []
    lastElement = lastState

    for ot in range(seqLen-1,-1,-1):

        stateSequence.append(lastElement)

        lastElement = int( pathMatrix[ot][lastElement] )

    stateSequence = np.array(stateSequence)
    return np.flip(stateSequence)


class HMM:  

    words = {}
    initPi = []
    A = []
    logPi = []
    logA = []
    startStates = []
    endStates = []

    def __init__(self, words = WORDS):

        self.words = words
        states = sum(self.words['size'])
        self.initPi = np.zeros( states )
        c = 0
        self.initPi[c] = 1/12
        self.A = np.zeros( (states,states) )
        for k in range(states-1):
            self.A[k][k:k+2] = 0.5
        
        self.A[-1][-1] = 0.5

        self.startStates = []
        self.startStates.append(c)

        for i in range(len(self.words['size'])-1):    
            c += self.words['size'][i]
            self.startStates.append(c)
            self.initPi[c] = 1/12 #np.log(1/12)
            sst = 0
            for j in range(len(self.words['size'])):
                self.A[c-1][sst] = 0.5*(1/12)
                sst += self.words['size'][j]

        w_beg = 0
        for s in range(len(self.words['size'])):
            self.A[-1][w_beg] = 0.5*(1/12)
            #self.A[0][w_beg] += (1/11)*(1/24)
            w_beg += self.words['size'][s]


        for s in range(1,len(self.startStates)):
            self.endStates.append(self.startStates[s]-1)
        self.endStates.append(105)
        

        #self.A[0] = self.A[0]*2   # - wie ist der erste Zustand zu handhaben ?
        self.A[0][0] += 0.5
        #self.A[0][0] = 0.5
        self.logPi = limLog(np.array(self.initPi))
        self.logA = limLog(np.array(self.A))
        
    def isIN(self,seq,idx):
        result = 1
        try: hypothesis = seq.index(idx)
        except ValueError: result = 0
        return result

    def getTranscription(self,ssq):

        preState = ssq[0]
        hypothesis = -1
        decode = []


        for idx, state in enumerate(ssq):



            if self.isIN(self.startStates,preState):
                hypothesis = self.startStates.index(preState)


            if self.A[preState][state] > 0: 

                if( self.isIN(self.endStates,state) and (hypothesis > 0.1) ):

                    decode.append(self.words['name'][hypothesis])
                    hypothesis = -1
            else: hypothesis = -1

            preState = state

        return decode


    def get_num_states(self):
        """
        Returns the total number of states of the defined HMM.

        :return: number of states.
        """
        return  sum(self.words['size'])


    def input_to_state(self, input):
        """
        Returns the state sequenze for a word.

        :param input: word of the defined HMM.
        :return: states of the word as a sequence.
        """
        if input not in self.words['name']:
            raise Exception('Undefined word/phone: {}'.format(input))

        # start index of each word
        start_idx = np.insert(np.cumsum(self.words['size']), 0, 0)

        # returns index for input's last state
        idx = self.words['name'].index(input) + 1

        start_state = start_idx[idx - 1]
        end_state = start_idx[idx]

        return [ n for n in range(start_state, end_state) ]
    
    def posteriors_to_transcription(self, posteriors):
        stateSequence = viterbi( limLog(posteriors), self.logPi, self.logA)
        return self.getTranscription(stateSequence)
    
    def posteriors_to_stateSequence(self, posteriors):
        return viterbi( limLog(posteriors), self.logPi, self.logA)
 

