#!/bin/bash

# 1. UBM training
	echo "Train Universal Background Model by EM algorithm"
	bin/TrainWorld --config cfg/TrainWorld.cfg &> log/TrainWorld.log
	echo "		done, see log/TrainWorld.log for details"

# 2. Total Variability matrix Estimation
	echo "Train TotalVariability matrix"
	bin/TotalVariability --config cfg/TotalVariability_fast.cfg &> log/TotalVariability.log
	echo "		done, see log/TotalVariability.log for details"

# 3. I-vector extraction
	echo "Extract i-vectors"
	bin/IvExtractor --config cfg/ivExtractor_fast_l.cfg &> log/IvExtractor.log
	echo "		done, see log/IvExtractor.log for details"

