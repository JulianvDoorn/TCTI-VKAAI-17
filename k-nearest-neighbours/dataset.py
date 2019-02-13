import numpy as np
import warnings

data = np.genfromtxt("../data/dataset1.csv", delimiter=";", usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})

dates = np.genfromtxt("../data/dataset1.csv", delimiter=";", usecols=[0])
labels = []
for label in dates:
  if label < 20000301:
    labels.append("winter")
  elif 20000301 <= label < 20000601:
    labels.append("lente") 
  elif 20000601 <= label < 20000901:
    labels.append("zomer") 
  elif 20000901 <= label < 20001201: 
    labels.append("herfst")
  else:
    labels.append("winter")

validation_data = np.genfromtxt("../data/validation1.csv", delimiter=";", usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
validation_dates = np.genfromtxt("../data/validation1.csv", delimiter=";", usecols=[0], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
validation_labels = []
for label in validation_dates:
  if label < 20010301:
    validation_labels.append("winter")
  elif 20010301 <= label < 20010601:
    validation_labels.append("lente") 
  elif 20010601 <= label < 20010901:
    validation_labels.append("zomer") 
  elif 20010901 <= label < 20011201: 
    validation_labels.append("herfst")
  else: # from 01-12 to end of year 
    validation_labels.append("winter")

random_days = np.genfromtxt("../data/days.csv", delimiter=";", usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
