import os
import mlflow
from mlflow.models import infer_signature

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X, y = datasets.load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

params = {
    "solver": "lbfgs",
    "max_iter": 2000,
    "random_state": 301,
}

lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

mlflow.set_tracking_uri(uri=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))

mlflow.set_experiment("Finding best random state")

with mlflow.start_run():
    mlflow.log_params(params)

    mlflow.log_metric("accuracy", accuracy)

    signature = infer_signature(X_train, lr.predict(X_train))

    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        name="iris_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-iris-model",
    )

    mlflow.set_logged_model_tags(
        model_info.model_id, {"Training Info": "Basic LR model for iris data"}
    )
