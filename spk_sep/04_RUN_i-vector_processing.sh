#!/bin/bash

# 1. Cosine Scoring with WCCN normalization
       echo "Compare models to test segments using Cosine scoring and WCCN"
       bin/IvTest --config cfg/ivTest_WCCN_Cosine.cfg &> log/IvTest_WCCN_Cosine.log
       echo "          done, see log/IvTest_WCCN_Cosine.log for details"

# 2. Mahalanobis scoring with EFR normalization of i-vectors
	echo "Compare models to test segments using Mahalanobis Distance with EFR normalization"
       bin/IvTest --config cfg/ivTest_EFR_Mahalanobis.cfg &> log/IvTest_EFR_Mahalanobis.log
       echo "          done, see log/IvTest_EFR_Mahalanobis.log for details"

# 3. 2-Covariance scoring with SphNorm
       echo "Compare models to test segments using 2-Covariance model scoring"
       bin/IvTest --config cfg/ivTest_SphNorm_2Cov.cfg &> log/IvTest_SphNorm_2Cov.log
       echo "          done, see log/IvTest_SphNorm_2Cov.log for details"

# 4. PLDA Testing including PLDA model training and i-vector normalization
       echo "Compare models to test segments using PLDA native scoring"
       bin/IvTest --config cfg/ivTest_SphNorm_Plda.cfg &> log/IvTest_SphNorm_Plda.log
       echo "          done, see log/IvTest_SphNorm_Plda.log for details"









