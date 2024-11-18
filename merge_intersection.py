import os
import json
import traceback
import pandas as pd
from uuid import uuid4


from xpms_file_storage.file_handler import XpmsResource, LocalResource
from xpms_helper.model.data_schema import DatasetFormat, DatasetConvertor
from xpms_storage.db_handler import DBProvider


class Util:
    @staticmethod
    def download_minio_file(file_path):
        ext = os.path.splitext(file_path)[-1]
        local_file_path = "/tmp/{}{}".format(str(uuid4()), ext)
        xrm = XpmsResource()
        mr = xrm.get(key=file_path)
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
    def upload_to_minio(solution_id, doc_id, local_file_path):
        minio_file_path = f'{solution_id}/sol_outputs/{doc_id}/models_ensemble.csv'
        lr = LocalResource(key=local_file_path)
        mr = XpmsResource().get(key=minio_file_path)
        lr.copy(mr)
        lr.delete()
        return mr.urn

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


def merge_intersection(config=None, **objects):
    try:
        solution_id = config["context"]["solution_id"]
        recommendations = list(objects.values())
        doc_id = recommendations[0]["recommendation"][0].get("doc_id", "")

        # create filepath and read df
        file_to_read = ["isof_recommendations", "cblof_recommendations", "gmm_recommendations"]
        dataframes = []
        for file in file_to_read:
            file_path = f"{solution_id}/sol_outputs/{doc_id}/{file}.csv"
            local_file_path = Util.download_minio_file(file_path)
            df = Util.read_file(local_file_path)
            dataframes.append(df)

        # get all common columns from all dfs
        common_columns = list(dataframes[0].columns)
        for df in dataframes[1:]:
            new_common_columns = []
            for column in common_columns:
                if column in df.columns:
                    new_common_columns.append(column)
            common_columns = new_common_columns

        # merge all dfs into single df
        merged_df = dataframes[0]

        for df in dataframes[1:]:
            merged_df = merged_df.merge(df, on=common_columns, how='outer')

        merged_df['Semisupervised_Recommendation'] = merged_df[
            ['ISOLATION_FOREST_PREDICTION', 'GMM_PREDICTION', 'CBLOF_PREDICTION']].apply(
            lambda row: 'Outlier' if all(value == -1 for value in row.values) else 'Non-Outliers', axis=1)
        merged_df['Semisupervised_Confidence'] = merged_df['Semisupervised_Recommendation'].apply(
            lambda status: 80 if status == 'Outlier' else 60)

        local_file_path = f"/tmp/{str(uuid4())}.csv"
        merged_df.to_csv(local_file_path, index=False)
        minio_path = Util.upload_to_minio(solution_id, doc_id, local_file_path)
        objects.update({"output_path": minio_path})
        return objects

        return

    except Exception as e:
        return traceback.format_exc()