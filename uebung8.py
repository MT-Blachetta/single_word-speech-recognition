import recognizer.hmm as HMM
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    # default HMM
    hmm = HMM.HMM()

    statesequence = [0, 0, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 0, 31, 32, 33, 34, 35, 36, 0, 37, 38, 39, 40, 41, 42, 43, 44, 45, 0]

    words = hmm.getTranscription(statesequence)
    print(words) # ['ONE', 'TWO', 'THREE']

    X_0 = [0, 1, 1, 2, 2, 3]
    print('X_0: ' + str(hmm.getTranscription(X_0)) )
    X_020 = [1, 2, 3, 3,31,32, 33, 34, 35, 36, 37, 38, 38, 1, 2, 3, 0]
    print('X_020: ' + str(hmm.getTranscription(X_020)) )
    X_2002 = [31,32, 33, 34, 35, 36, 37, 38, 0, 1, 2, 3, 1, 2, 3, 33, 34, 35, 36, 37, 38]
    print('X_2002: ' + str(hmm.getTranscription(X_2002)))

    plt.imshow(np.exp(hmm.logA))
    plt.xlabel('from state i')
    plt.ylabel('to state j')
    plt.colorbar()
    plt.show()
