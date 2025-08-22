from datasets import load_dataset, load_dataset_builder, Image, DatasetDict, Value, Sequence
import os

os.system(
    "huggingface-cli download liuwenhan/reasonrank_data_13k --repo-type=dataset --include=\"id_doc/*\" --local-dir ./reasonrank_data/")
os.system(
    "huggingface-cli download liuwenhan/reasonrank_data_13k --repo-type=dataset --include=\"id_query/*\" --local-dir ./reasonrank_data/")

import json
def read_json_files(directory):
    data = {}
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as f:
                content = json.load(f)
                data[filename[:-5]] = content
    return data


# read all json files in reasonrank_data/id_doc and reasonrank_data/id_query

id_doc_data = read_json_files('./reasonrank_data/id_doc')

id_query_data = read_json_files('./reasonrank_data/id_query')


def load_datasets():
    ds =  load_dataset("liuwenhan/reasonrank_data_13k")
    return ds


def transform_passages(entry):
    positive_passages = []
    negative_passages = []
    query_id = []
    query = []

    for qid, inital_list, relevant_docids, dataset in zip(entry['qid'], entry['initial_list'], entry['relevant_docids'], entry['dataset']):
        positive_document_ids = relevant_docids
        # set difference between initial_list and relevant_docids as negative document ids
        negative_document_ids = list(set(inital_list) - set(relevant_docids))
        if len(negative_document_ids) != len(inital_list) - len(relevant_docids):
            raise ValueError("some relevant document ids are not in the initial list")

        if len(negative_document_ids) == 0:
            # skip this query entirely
            print(f"Skipping query {qid} in dataset {dataset} because there are no negative document ids.")
            continue

        query_id.append(qid)
        query.append(id_query_data[dataset][qid])
        positives = []
        negatives = []
        for positive_document_id in positive_document_ids:
            docid = positive_document_id
            text = id_doc_data[dataset][docid]
            positives.append({"docid": docid, "text": text})

        for negative_document_id in negative_document_ids:
            docid = negative_document_id
            text = id_doc_data[dataset][docid]
            negatives.append({"docid": docid, "text": text})
        positive_passages.append(positives)
        negative_passages.append(negatives)


    return {
        "query_id": query_id,
        "query": query,
        "positive_passages": positive_passages,
        "negative_passages": negative_passages,
    }


def transform_dataset(ds):
    # remove data points that dataset == 'leetcode'
    # ds = ds.filter(lambda ex: ex["dataset"] != "leetcode", num_proc=8)

    trans_ds = ds.map(transform_passages, remove_columns=["dataset", "qid", "initial_list", "final_list", "reasoning", 'relevant_docids'], batched=True,
                      num_proc=8)

    # update old column attribute types
    # trans_ds = trans_ds.cast_column("query_id", Value("string")).cast_column("query", Value("string")).cast_column(
    #     "positive_passages", Sequence(DatasetDict("string"))).cast_column("negative_passages", Sequence(Value("string")))




    # reorder columns
    return trans_ds.select_columns(
        ['query_id', 'query','positive_passages', 'negative_passages'])


def upload_dataset(new_ds_dict):
    new_ds_dict.push_to_hub("ArvinZhuang/reasonrank-data-hn")


def main():
    ds_dict = load_datasets()
    print(ds_dict)
    ds_dict = {split: transform_dataset(ds_dict[split]) for split in ds_dict}
    # # perform dataset update
    upload_dataset(DatasetDict(ds_dict))
    # # verify feature
    print("-------------------")
    print(load_dataset_builder("ArvinZhuang/reasonrank-data-hn").info.features)


if __name__ == "__main__":
    main()