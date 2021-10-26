# LSTD

Implementation of Least squared TD learning algorithm

lstd.py
- uses mountain car as the test domain
- other test domains can be added later
- run: python lstd.py

lstd_transition.py
- computes beta from a given set of S,A,R,S' transition
- run: 
  - using manual input > python lstd_transition.py
  - using specified default parameters > python lstd_transition.py -d 

value_estimator.py
- reads beta vector from file and calculates value of a state
- demonstrates how the class can be used
