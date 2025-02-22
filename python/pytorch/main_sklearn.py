import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

from dataset import ModelNet40graph
from autosklearn.regression import AutoSklearnRegressor
import numpy as np

if __name__=="__main__":
    print(np.__version__)
    train_dataset = ModelNet40graph(root="data_train")
    test_dataset = ModelNet40graph(root="data_test")

    automl = AutoSklearnRegressor(
        time_left_for_this_task=120,
        per_run_time_limit=30,
        tmp_folder="/tmp/autosklearn_multioutput_regression_example_tmp",
    )


    #x = np.cat((train_dataset.data[0].pos, train_dataset.data[1].pos))
    #print(x)
    #y = train_dataset.data[1].y
    #automl.fit(x, y, dataset_name="synthetic")