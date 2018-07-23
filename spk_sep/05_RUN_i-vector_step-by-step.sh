#!/bin/bash

# 1. I-vector Normalization
	echo "Normalize i-vectors"
	bin/IvNorm --config cfg/ivNorm.cfg &> log/IvNorm.log
	echo "		done, see log/IvNorm.log for details"

# 2. PLDA Training
	echo "Train Probabilistic Linear Discriminant Analysis model"
	bin/PLDA --config cfg/Plda.cfg &> log/Plda.log
	echo "		done, see log/Plda.log for details"

# 3. PLDA Testing
	echo "Compare models to test segments using PLDA native scoring"
	bin/IvTest --config cfg/ivTest_Plda.cfg &> log/IvTest_Plda.log
	echo "		done, see log/IvTest_Plda.log for details"
