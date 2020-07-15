import pickle

filename = '/home/tpatten/Data/Hands/HO3D/train/ABF10/meta/0000.pkl'
with open(filename, 'rb') as f:
    try:
        in_data = pickle.load(f, encoding='latin1')
    except:
        in_data = pickle.load(f)
        
for key in in_data.keys():
    print(key)
    print(in_data[key])
