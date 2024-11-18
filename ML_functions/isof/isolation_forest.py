import os
import pandas as pd
import numpy as np
import traceback
import json

from xpms_file_storage.file_handler import XpmsResource, LocalResource
from xpms_helper.model.data_schema import DatasetFormat, DatasetConvertor
from xpms_helper.model import model_utils
from xpms_storage.db_handler import DBProvider


from sklearn.ensemble import IsolationForest

IDENTIFIER_FEATURES = ['UNIQUE_RECORD_ID', 'SOURCE_CLAIM_ID', 'MEMBER_ID']
OTHER_NON_REQUIRED_FEATURES = ['clustering_prediction']


class Util:
    @staticmethod
    def download_minio_file(file_path):
        ext = os.path.splitext(file_path)[-1]
        local_file_path = "/tmp/" + file_path.split('/')[-1]
        xrm = XpmsResource()
        mr = xrm.get(urn=file_path)
        lr = LocalResource(key=local_file_path)
        mr.copy(lr)
        return local_file_path

    @staticmethod
    def read_file(file_path):
        df = None
        extn = file_path.split(".")[-1]
        if extn.lower() in ['xls', 'xlsx']:
            df = pd.read_excel(file_path, sheet_name=0, engine='openpyxl')
        elif extn.lower() in ['csv']:
            df = pd.read_csv(file_path)
        return df

    @staticmethod
    def upload_to_minio(solution_id, local_file_path, doc_id='train'):
        minio_file_path = f'{solution_id}/sol_outputs/{doc_id}/isof_recommendations.csv'
        lr = LocalResource(key=local_file_path)
        mr = XpmsResource().get(key=minio_file_path)
        lr.copy(mr)
        lr.delete()
        return minio_file_path

    @staticmethod
    def get_doc_id(solution_id, dag_execution_id):
        obj = DBProvider.get_instance(db_name=solution_id)
        filter_obj = {"execution_id": dag_execution_id}
        res = obj.find(table="dag_task_executions", filter_obj=filter_obj, multi_select=False)
        inputs = json.loads(res['inputs'][0])
        if 'document' in inputs:
            doc_id = inputs['document'][0]['doc_id']
        elif 'data' in inputs and 'doc_id' in inputs['data']:
            doc_id = inputs['data']['doc_id']
        else:
            doc_id = None
        return doc_id

    @staticmethod
    def get_anomaly_and_small_cluster_df(run_df):
        label_counts = run_df['clustering_prediction'].value_counts()
        label_counts = label_counts[label_counts.index != -1]
        max_cluster = int(label_counts.idxmax())
        filtered_df = run_df[run_df['clustering_prediction'] != max_cluster]
        return filtered_df


def train(datasets, config):
    dataset = DatasetConvertor.convert(datasets, DatasetFormat.DATA_FRAME, None)
    train_info = {"name": "iso_forest"}
    return train_info, dataset


def run(datasets, config):
    solution_id = config["context"]["solution_id"]
    dag_execution_id = config["context"]["dag_execution_id"]
    doc_id = Util.get_doc_id(solution_id, dag_execution_id) if dag_execution_id else config["context"]["doc_id"]

    dataset = DatasetConvertor.convert(datasets, DatasetFormat.DATA_FRAME, None)
    df = dataset["value"]

    run_df = Util.get_anomaly_and_small_cluster_df(df) if 'clustering_prediction' in df.columns else df


    column_to_sep = IDENTIFIER_FEATURES + OTHER_NON_REQUIRED_FEATURES if all(
        col in run_df.columns for col in OTHER_NON_REQUIRED_FEATURES) else IDENTIFIER_FEATURES
    df_identifier = run_df[column_to_sep]
    run_df = run_df.drop(columns=column_to_sep)

    model_name = "isolation_forest_isof.pkl"
    model_obj = model_utils.load(file_name=model_name, config=config)
    run_df = run_df[model_obj.feature_names_in_]
    predictions = model_obj.predict(run_df)
    decision_function = model_obj.decision_function(run_df)
    run_df["ISOLATION_FOREST_PREDICTION"] = predictions
    run_df["ISOLATION_FOREST_DECISION_FUNCTION"] = decision_function
    run_df = pd.concat([df_identifier, run_df], axis=1)
    columns_to_keep = IDENTIFIER_FEATURES + ['ISOLATION_FOREST_PREDICTION', 'ISOLATION_FOREST_DECISION_FUNCTION']
    run_df = run_df[columns_to_keep]

    # merge model output in main dataframe
    merged_df = pd.merge(df, run_df, on=IDENTIFIER_FEATURES, how='left')
    merged_df['ISOLATION_FOREST_PREDICTION'].fillna(1, inplace=True)
    merged_df['ISOLATION_FOREST_DECISION_FUNCTION'].fillna(np.nan, inplace=True)


    local_file_path = f"/tmp/iso_forest.csv"
    merged_df.to_csv(local_file_path, index=False)
    doc_id = doc_id if doc_id else "train"
    minio_path = Util.upload_to_minio(solution_id, local_file_path, doc_id=doc_id)
    result_df = pd.DataFrame(data=predictions, columns=['isolation_forest'])
    result_dataset = {"value": result_df.head(10), "data_format": "data_frame"}
    return result_dataset


def evaluate(datasets, config):
    dataset = DatasetConvertor.convert(datasets, DatasetFormat.DATA_FRAME, None)
    if "scorers" in config:
        scorers = config["scorers"]
    else:
        scorers = ["accuracy"]
    eval_df = dataset["value"]
    eval_df = eval_df.head(10)
    y = pd.Series(1, index=eval_df.index)
    model_output = run(datasets, config)
    y_pred = model_output["value"]['isolation_forest'].values
    score = model_utils.calculate_metrics(dataset["value"], scorers, y, y_pred, config)
    return score, model_output


def test_template():
    config = {}
    config["storage"] = "local"
    config["src_dir"] = os.getcwd()
    dataset_obj = json.load(open(os.path.join(os.getcwd(), "datasets_obj/dataset_obj.json")))
    dataset_format = dataset_obj["data_format"]
    if dataset_format != "list":
        dataset_obj["value"] = LocalResource(key=os.path.join(os.getcwd(), "datasets")).urn
    train(dataset_obj, config)
    run(dataset_obj, config)
    evaluate(dataset_obj, config)
