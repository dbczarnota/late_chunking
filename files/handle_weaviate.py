from dotenv import load_dotenv
import os
import weaviate
import weaviate.classes as wvc
import weaviate.classes.config as wvcc

def connect_to_weaviate(collection_name, delete_existing=False):
    """
    Connect to a Weaviate instance and create a collection.

    Parameters:
        collection_name (str): Name of the collection to create.
        delete_existing (bool): Whether to delete the collection if it already exists.

    Returns:
        client (weaviate.Client): The Weaviate client instance.
    """
    load_dotenv()

    # Connect to Weaviate cloud
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),                                
        auth_credentials=wvc.init.Auth.api_key(os.getenv("WEAVIATE_KEY"))
    )

    if client.collections.exists(collection_name):
        if delete_existing:
            client.collections.delete(collection_name)
        else:
            print(f"Collection '{collection_name}' already exists. No changes made.")
            return client

    # Create a new collection with no vectorizer
    client.collections.create(
        name=collection_name,
        vectorizer_config=wvc.config.Configure.Vectorizer.none(),
    )

    print(f"Collection '{collection_name}' created successfully.")
    return client






# Example usage:
client = connect_to_weaviate("test_late_chunking", delete_existing=False)
client.close()


