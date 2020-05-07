import pickle

filename = '/home/tpatten/v4rtemp/datasets/HandTracking/HO3D_v2/train/ABF11/meta/0000.pkl'
with open(filename, 'rb') as f:
    try:
        in_data = pickle.load(f, encoding='latin1')
    except:
        in_data = pickle.load(f)
        
for key in in_data.keys():
    print(key)
    print(in_data[key])
