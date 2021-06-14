# xgboost macos seg fault when libomp==12.0.0
# passes with libomp==11.1.0

"""
	verify libomp version
		brew list --version libomp
		libomp 12.0.0
	to install libomp 11.1.0
		wget https://raw.githubusercontent.com/chenrui333/homebrew-core/0094d1513ce9e2e85e07443b8b5930ad298aad91/Formula/libomp.rb
        brew unlink libomp
        brew install --build-from-source ./libomp.rb
        brew list --version libomp

"""



# Press ⇧F10 to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import logging
import numpy as np
import xgboost as xgb


def xgboost_unit_tests():
	feature_arr = np.random.uniform(size=(100,5))
	label_arr = np.random.randint(low=0, high=10, size=(100,1))
	data_arr = np.concatenate((feature_arr, label_arr), axis=1)
	#Run Attribute Rank with -O 2
	run_xgboost(data_arr,data_arr)
	logging.info("XGBoost complete")
	return


def run_xgboost(trainarr, valarr):
	trainfeats = trainarr[:,:-1]
	trainlabels = trainarr[:,-1]
	# first seg fault
	dtrain = xgb.DMatrix(trainfeats, label=trainlabels)
	valfeats = valarr[:,:-1]
	vallabels = valarr[:,-1]
	# second seg fault
	dval = xgb.DMatrix(valfeats, label=vallabels)
	return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
	xgboost_unit_tests()

