"""Übung 7: Verständnis des Viterbi-Algorithmus mit einem Spielzeugbeispiel."""

import numpy as np

if __name__ == "__main__":
    
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
        print('number_of_states: ' + str(number_of_states) )
        seqLen = len(logLike)
        print('seqLen: '+ str(seqLen) )
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
        print('lastState: '+ str(lastState))
        for ot in range(seqLen-1,-1,-1):
            print('ot: ' + str(ot) )
            stateSequence.append(lastElement)
            print('stateSequence: ' + str(stateSequence) )
            lastElement = int( pathMatrix[ot][lastElement] ) 
            print('lastElement: '+ str(lastElement))
        print('OBSERVATION PROBABILITY: ' + str(observationProbability))
        print('leaving function...')
        stateSequence = np.array(stateSequence)
        return np.flip(stateSequence)
        #return(str(pathMatrix)) 
    
    # Vektor der initialen Zustandswahrscheinlichkeiten
    logPi = limLog([ 0.9, 0, 0.1 ])

    # Matrix der Zustandsübergangswahrscheinlichkeiten
    logA  = limLog([
      [ 0.8,   0, 0.2 ], 
      [ 0.4, 0.4, 0.2 ], 
      [ 0.3, 0.2, 0.5 ] 
    ]) 

    # Beobachtungswahrscheinlichkeiten für "Regen", "Sonne", "Schnee" 
    # B = [
    #     {  2: 0.1,  3: 0.1,  4: 0.2,  5: 0.5,  8: 0.1 },
    #     { -1: 0.1,  1: 0.1,  8: 0.2, 10: 0.2, 15: 0.4 },
    #     { -3: 0.2, -2: 0.0, -1: 0.8,  0: 0.0 }
    # ]




    # gemessene Temperaturen (Beobachtungssequenz): [ 2, -1, 8, 8 ]
    # p(xi|o)
    # ergibt folgende Zustands-log-Likelihoods
    logLike = limLog([
      [ 0.1,   0,   0 ],
      [   0, 0.1, 0.8 ],
      [ 0.1, 0.2,   0 ],
      [ 0.1, 0.2,   0 ]
    ])
    

    # erwartetes Ergebnis: [0, 2, 1, 1], -9.985131541576637
    print( viterbi( logLike, logPi, logA ) )


    # verlängern der Beobachtungssequenz um eine weitere Beobachung 
    # mit der gemessenen Temperatur 4
    # neue Beobachtungssequenz: [ 2, -1, 8, 8, 4 ]
    logLike = np.vstack( ( logLike, limLog([ 0.2, 0, 0 ]) ) )

    # erwartetes Ergebnis: 0, 2, 0, 0, 0][, -12.105395077776727
    print( viterbi( logLike, logPi, logA ) )
