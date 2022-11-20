from utils.group_estimators import GroupEstimatorNumerical, GroupEstimatorHierarchical
from utils.data_generation import generate_one_component_simple

data, tru_cov = generate_one_component_simple()
estimator = GroupEstimatorHierarchical(n_clusters=3)
estimator.fit(data)
print(estimator.predicted_groups)

