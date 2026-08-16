import pyarrow.parquet as pq

path = r"C:\Users\Aryan Shaikh\.cache\huggingface\hub\datasets--ai4bharat--MSMARCO-XI\snapshots\bf5cdc1f26e581e519018e434db14edd1b77602b\train\hintrain.parquet"

parquet_file = pq.ParquetFile(path)

for batch in parquet_file.iter_batches(batch_size=1):
    record = batch.to_pylist()[0]

    print(record)
    break