try:
    from sentence_transformers import SentenceTransformer
    print("Import successful!")
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
