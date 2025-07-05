#!/usr/bin/env python3
"""
Migrate ChromaDB to fix compatibility issues
"""

import os
import sys
import json
import shutil
sys.path.append('.')

from environment import CHROMA_PERSIST_DIRECTORY

def backup_old_db():
    """Backup the old database"""
    backup_dir = CHROMA_PERSIST_DIRECTORY + "_backup"
    if os.path.exists(backup_dir):
        print(f"Backup already exists at {backup_dir}")
        return
    
    print(f"Backing up database from {CHROMA_PERSIST_DIRECTORY} to {backup_dir}")
    shutil.copytree(CHROMA_PERSIST_DIRECTORY, backup_dir)
    print("Backup completed")

def extract_documents():
    """Extract documents from the old database"""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
        
        # Get all collections
        collections = client.list_collections()
        print(f"Found {len(collections)} collections")
        
        extracted_data = {}
        for collection_info in collections:
            print(f"Processing collection: {collection_info.name}")
            
            try:
                collection = client.get_collection(collection_info.name)
                count = collection.count()
                print(f"  Collection has {count} documents")
                
                if count > 0:
                    # Extract all documents in batches
                    batch_size = 1000
                    all_documents = []
                    
                    for offset in range(0, count, batch_size):
                        limit = min(batch_size, count - offset)
                        batch = collection.get(limit=limit, offset=offset)
                        
                        if batch and batch.get('documents'):
                            for i, doc in enumerate(batch['documents']):
                                doc_data = {
                                    'id': batch['ids'][i],
                                    'document': doc,
                                    'metadata': batch['metadatas'][i] if batch.get('metadatas') else {}
                                }
                                all_documents.append(doc_data)
                        
                        print(f"  Extracted {offset + len(batch.get('documents', []))} / {count} documents")
                    
                    extracted_data[collection_info.name] = all_documents
                    print(f"  Successfully extracted {len(all_documents)} documents from {collection_info.name}")
                
            except Exception as e:
                print(f"  Error processing collection {collection_info.name}: {e}")
                continue
        
        # Save extracted data
        backup_file = "extracted_documents.json"
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, indent=2, ensure_ascii=False)
        
        print(f"Extracted data saved to {backup_file}")
        return extracted_data
        
    except Exception as e:
        print(f"Error extracting documents: {e}")
        import traceback
        traceback.print_exc()
        return None

def recreate_database():
    """Recreate the database with new ChromaDB version"""
    print("Recreating database...")
    
    # Remove old database with retry logic for Windows file locking
    if os.path.exists(CHROMA_PERSIST_DIRECTORY):
        print(f"Removing old database at {CHROMA_PERSIST_DIRECTORY}")
        
        # Try to remove with retries for Windows file locking
        max_retries = 3
        for attempt in range(max_retries):
            try:
                shutil.rmtree(CHROMA_PERSIST_DIRECTORY)
                print("Old database removed successfully")
                break
            except PermissionError as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed (file in use), retrying in 2 seconds...")
                    import time
                    time.sleep(2)
                else:
                    print(f"Failed to remove database after {max_retries} attempts: {e}")
                    print("Please close any running applications using the database (like MCP server) and try again.")
                    print("Or manually delete the 'chroma_db' folder and run the migration again.")
                    return None
            except Exception as e:
                print(f"Unexpected error removing database: {e}")
                return None
    
    # Create new database
    try:
        from vector_store import VectorStore
        print("Creating new VectorStore...")
        vs = VectorStore()
        
        print(f"New database created with collections: {list(vs.collections.keys())}")
        return vs
    except Exception as e:
        print(f"Failed to create new VectorStore: {e}")
        return None

def restore_documents(vs, extracted_data):
    """Restore documents to the new database"""
    print("Restoring documents...")
    
    for collection_name, documents in extracted_data.items():
        print(f"Restoring {len(documents)} documents to {collection_name}")
        
        # Convert to the format expected by VectorStore.add_documents
        formatted_docs = []
        for doc_data in documents:
            formatted_doc = {
                'content': doc_data['document'],
                'metadata': doc_data['metadata']
            }
            formatted_docs.append(formatted_doc)
        
        # Add documents in batches
        batch_size = 100
        for i in range(0, len(formatted_docs), batch_size):
            batch = formatted_docs[i:i+batch_size]
            try:
                vs.add_documents(collection_name, batch)
                print(f"  Added batch {i//batch_size + 1} ({len(batch)} documents)")
            except Exception as e:
                print(f"  Error adding batch {i//batch_size + 1}: {e}")

def main():
    print("ChromaDB Migration Tool")
    print("======================")
    
    # Check if extraction file already exists
    if os.path.exists("extracted_documents.json"):
        print("Found existing extraction file. Do you want to use it? (y/n)")
        response = input().lower()
        if response == 'y':
            with open("extracted_documents.json", 'r', encoding='utf-8') as f:
                extracted_data = json.load(f)
            print(f"Loaded {len(extracted_data)} collections from existing extraction")
        else:
            # Step 1: Backup
            backup_old_db()
            
            # Step 2: Extract documents
            extracted_data = extract_documents()
            if not extracted_data:
                print("Failed to extract documents. Aborting.")
                return
    else:
        # Step 1: Backup
        backup_old_db()
        
        # Step 2: Extract documents
        extracted_data = extract_documents()
        if not extracted_data:
            print("Failed to extract documents. Aborting.")
            return
    
    # Step 3: Recreate database
    vs = recreate_database()
    
    if vs is None:
        print("\nDatabase recreation failed. Please:")
        print("1. Close any running applications (MCP server, other Python processes)")
        print("2. Manually delete the 'chroma_db' folder")
        print("3. Run this migration script again")
        print("\nYour data is safely preserved in 'extracted_documents.json'")
        return
    
    # Step 4: Restore documents
    restore_documents(vs, extracted_data)
    
    print("\nMigration completed!")
    
    # Test the migrated database
    print("\nTesting migrated database...")
    for name, collection in vs.collections.items():
        count = collection.count()
        print(f"  {name}: {count} documents")
    
    # Test a query
    print("\nTesting query...")
    try:
        results = vs.query(
            collection_name="all_vendor_docs",
            query_text="RBAC role based access control",
            n_results=2
        )
        print(f"Query successful: {len(results.get('documents', [[]])[0])} results")
        if results.get('documents') and results['documents'][0]:
            print(f"First result preview: {results['documents'][0][0][:200]}...")
    except Exception as e:
        print(f"Query test failed: {e}")

if __name__ == "__main__":
    main() 