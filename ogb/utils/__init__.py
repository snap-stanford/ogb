try:
    from ogb.mol import smiles2graph
except Exception as e:
    print("Error on imports, verify if all dependencies are installed correctly.")
    raise e