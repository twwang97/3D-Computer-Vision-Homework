
import pandas as pd

def partition_array():

	#############
	#
	# train.pkl
	#
	#############
	large_train_pkl = pd.read_pickle("./train.pkl")

	# Delete the last few rows
	large_train_pkl_1 = large_train_pkl.drop(large_train_pkl.index[300000:682468])
	# Delete the first 3000000 rows
	large_train_pkl_2 = large_train_pkl.drop(large_train_pkl.index[range(300000)])

	print(type(large_train_pkl)) # <class 'pandas.core.frame.DataFrame'>
	print(large_train_pkl.shape)
	print(large_train_pkl_1.shape)
	print(large_train_pkl_2.shape)

	large_train_pkl_1.to_pickle("./train_1.pkl")  
	large_train_pkl_2.to_pickle("./train_2.pkl") 

	#############
	#
	# point_desc.pkl
	#
	############# 

	large_point_desc_pkl = pd.read_pickle("./point_desc.pkl")
	large_point_desc_pkl_1 = large_point_desc_pkl.drop(large_point_desc_pkl.index[400000:large_point_desc_pkl.shape[0]])
	large_point_desc_pkl_2 = large_point_desc_pkl.drop(large_point_desc_pkl.index[800000:large_point_desc_pkl.shape[0]])
	large_point_desc_pkl_2 = large_point_desc_pkl_2.drop(large_point_desc_pkl_2.index[range(400000)])
	large_point_desc_pkl_3 = large_point_desc_pkl.drop(large_point_desc_pkl.index[range(800000)])

	print(large_point_desc_pkl.shape) # (1234458, 4)
	print(large_point_desc_pkl_1.shape)
	print(large_point_desc_pkl_2.shape)
	print(large_point_desc_pkl_3.shape)

	large_point_desc_pkl_1.to_pickle("./point_desc_1.pkl") 
	large_point_desc_pkl_2.to_pickle("./point_desc_2.pkl") 
	large_point_desc_pkl_3.to_pickle("./point_desc_3.pkl") 

def concatenate_pandas():

	#############
	#
	# train.pkl
	#
	#############
	large_train_pkl_1 = pd.read_pickle("data/train_1.pkl")
	large_train_pkl_2 = pd.read_pickle("data/train_2.pkl")
	large_train_pkl = pd.concat([large_train_pkl_1, large_train_pkl_2], axis=0)
	print(large_train_pkl.shape)
	large_train_pkl.to_pickle("data/train.pkl") 
	print('\ntrain.pkl is saved')

	#############
	#
	# point_desc.pkl
	#
	############# 

	large_point_desc_pkl_1 = pd.read_pickle("data/point_desc_1.pkl")
	large_point_desc_pkl_2 = pd.read_pickle("data/point_desc_2.pkl")
	large_point_desc_pkl_3 = pd.read_pickle("data/point_desc_3.pkl")
	large_point_desc_pkl = pd.concat([large_point_desc_pkl_1, large_point_desc_pkl_2], axis=0)
	large_point_desc_pkl = pd.concat([large_point_desc_pkl, large_point_desc_pkl_3], axis=0)
	print(large_point_desc_pkl.shape)
	large_point_desc_pkl.to_pickle("data/point_desc.pkl") 
	print('\npoint_desc.pkl is saved')

if __name__ == '__main__':
	# partition_array()
	concatenate_pandas()
