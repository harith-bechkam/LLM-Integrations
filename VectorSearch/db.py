from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

CLOUD_ENDPOINT = "in03-fa4be7617b28e70.serverless.aws-eu-central-1.cloud.zilliz.com"
CLOUD_USER = "db_fa4be7617b28e70"
CLOUD_PASSWORD = "Ut1%TZkmjPfHh49f"

MAIN_COLLECTION = "documents"
CHUNK_COLLECTION = "document_chunks"


def connect_milvus():
    connections.connect(
        alias="default",
        uri=f"https://{CLOUD_ENDPOINT}",  # Use URI for Zilliz Cloud Serverless
        user=CLOUD_USER,
        password=CLOUD_PASSWORD
    )
    print("Connected to Milvus Cloud")

    # -------- Main collection --------
    try:
        main_col = Collection(MAIN_COLLECTION)
        print("Loaded existing main collection")
    except Exception:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="full_embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
            FieldSchema(name="full_content", dtype=DataType.VARCHAR, max_length=65535),  # large text content
        ]
        schema = CollectionSchema(fields, description="Full document embeddings")
        main_col = Collection(MAIN_COLLECTION, schema=schema)
        print("Created main collection")

    # -------- Chunk collection --------
    try:
        chunk_col = Collection(CHUNK_COLLECTION)
        print("Loaded existing chunk collection")
    except Exception:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="file_id", dtype=DataType.INT64),
            FieldSchema(name="chunk_embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
            FieldSchema(name="chunk_content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
        ]
        schema = CollectionSchema(fields, description="Document chunks embeddings")
        chunk_col = Collection(CHUNK_COLLECTION, schema=schema)
        print("Created chunk collection")

    # -------- Create index if not exists --------
    if not main_col.has_index():
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 128}
        }
        main_col.create_index("full_embedding", index_params)
        print("Created index for main collection")

    if not chunk_col.has_index():
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 128}
        }
        chunk_col.create_index("chunk_embedding", index_params)
        print("Created index for chunk collection")

    # -------- Flush & Load --------
    main_col.flush()
    chunk_col.flush()
    main_col.load()
    chunk_col.load()
    print("Collections flushed and loaded")

    return main_col, chunk_col
